# 版权所有 (c) 2025 NVIDIA CORPORATION & AFFILIATES。保留所有权利。
# SPDX-License-Identifier: BSD-3-Clause

# 在满足以下条件的情况下，允许以源代码和二进制形式重新分发和使用，
# 无论是否进行修改：

# 1. 源代码的重新分发必须保留上述版权声明、
# 本条件列表和以下免责声明。

# 2. 二进制形式的重新分发必须在文档和/或随分发提供的
# 其他材料中复制上述版权声明、本条件列表和以下免责声明。

# 3. 未经特定的事先书面许可，不得使用版权持有者的名称
# 或其贡献者的名称来认可或推广从本软件衍生的产品。

# 本软件由版权持有者和贡献者"按原样"提供，
# 不提供任何明示或暗示的保证，包括但不限于
# 适销性和特定用途适用性的暗示保证。
# 在任何情况下，版权持有者或贡献者均不对任何直接、间接、
# 偶然、特殊、惩戒性或后果性损害（包括但不限于
# 替代商品或服务的采购；使用、数据或利润的损失；
# 或业务中断）承担责任，无论是基于合同、严格责任
# 还是侵权（包括疏忽或其他方式）的任何责任理论，
# 即使已被告知此类损害的可能性。

import argparse  # 导入命令行参数解析模块
import math  # 导入数学函数模块
import time  # 导入时间处理模块
from typing import Tuple, Type  # 导入类型提示模块

import cuda.bindings.driver as cuda  # 导入CUDA驱动绑定
import torch  # 导入PyTorch深度学习框架

import cutlass  # 导入CUTLASS库主模块
import cutlass.cute as cute  # 导入CuTe DSL模块
import cutlass.cute.testing as testing  # 导入CuTe测试工具
import cutlass.torch as cutlass_torch  # 导入CUTLASS的PyTorch集成
import cutlass.utils as utils  # 导入CUTLASS工具函数
from cutlass.cute.runtime import from_dlpack  # 从DLPack格式导入张量转换函数

"""
使用CUTE DSL在NVIDIA Ampere架构上实现的密集GEMM（C = A * B）示例。
- 矩阵A的形状为MxKxL，L是批次维度，A可以是行主序("K")或列主序("M")
- 矩阵B的形状为NxKxL，L是批次维度，B可以是行主序("N")或列主序("K")  
- 矩阵C的形状为MxNxL，L是批次维度，C可以是行主序("N")或列主序("M")

此GEMM内核支持以下特性：
    - 利用Ampere的张量核心进行矩阵乘加(MMA)运算
    - 线程块光栅化以提高数据重用
    - 支持多阶段流水线以重叠计算和内存访问
    - 实现用于尾声(epilogue)的共享内存缓冲以增加合并的全局内存访问

此GEMM的工作流程如下：
1. 使用异步拷贝将A和B矩阵从全局内存(GMEM)加载到共享内存(SMEM)。
2. 执行矩阵乘加(MMA)运算。
3. 将结果从寄存器(RMEM)存储到共享内存(SMEM)，然后到全局内存(GMEM)。

使用的Ampere张量核心指令的操作如下：
- 从SMEM读取矩阵A
- 从SMEM读取矩阵B
- 执行MMA运算并将结果存储在累加器(寄存器)中

运行此示例：

.. code-block:: bash

    python examples/ampere/tensorop_gemm.py                                  \\
      --mnkl 8192,8192,8192,1 --atom_layout_mnk 2,2,1                        \\
      --ab_dtype Float16                                                     \\
      --c_dtype Float16 --acc_dtype Float32                                  \\
      --a_major m --b_major n --c_major n

上述示例命令计算M=8192，N=8192，K=8192，
batch_count=1。原子布局的形状为2x2x1，输入、mma
累加器和输出数据类型分别设置为fp16、fp32和fp16。

使用NCU分析器收集性能数据：

.. code-block:: bash

    ncu python examples/ampere/tensorop_gemm.py                              \\
      --mnkl 8192,8192,8192,1 --atom_layout_mnk 2,2,1                        \\
      --ab_dtype Float16                                                     \\
      --c_dtype Float16 --acc_dtype Float32                                  \\
      --a_major m --b_major n --c_major n                                    \\
      --skip_ref_check --iterations 2

约束条件：
* 支持的输入和输出数据类型：fp16
* 支持的累加器数据类型：f32
* 默认分块形状设置为128x128x32
* 原子布局的MNK形状设置为使分块形状能被MMA指令形状整除
* A/B/C张量的连续维度必须至少16字节对齐，
  即元素数量是8的倍数
"""


class TensorOpGemm:
    """
    Ampere架构张量运算GEMM类
    
    实现了使用CUTLASS和CuTe DSL的高性能GEMM运算，
    专门针对NVIDIA Ampere架构的张量核心进行优化。
    """
    
    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],  # A和B矩阵的数据类型
        c_dtype: Type[cutlass.Numeric],   # C矩阵的数据类型
        acc_dtype: Type[cutlass.Numeric], # 累加器的数据类型
        atom_layout_mnk: Tuple[int, int, int],  # 原子布局的MNK维度
    ):
        """
        初始化TensorOpGemm类
        
        Args:
            ab_dtype: A和B矩阵的数据类型
            c_dtype: C矩阵的数据类型  
            acc_dtype: 累加器的数据类型
            atom_layout_mnk: 原子布局的(M, N, K)维度元组
        """
        self.ab_dtype = ab_dtype      # 存储A和B矩阵的数据类型
        self.c_dtype = c_dtype        # 存储C矩阵的数据类型
        self.acc_dtype = acc_dtype    # 存储累加器的数据类型
        self.cta_tiler = (128, 128, 32)  # CTA(Cooperative Thread Array)分块器，设置线程块的分块大小
        self.num_stages = 3           # 流水线阶段数，用于重叠计算和内存访问
        self.atom_layout_mnk = atom_layout_mnk  # 原子布局的MNK维度
        atom_lay_M, atom_lay_N, atom_lay_K = self.atom_layout_mnk  # 解包原子布局维度
        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32  # 计算线程数量

        self.bM, self.bN, self.bK = self.cta_tiler  # 解包CTA分块器的维度
        self.mma_inst_shape = (16, 8, 16)  # MMA指令的形状(M, N, K)
        mmaM, mmaN, mmaK = self.mma_inst_shape  # 解包MMA指令形状

        # 验证分块大小能被MMA指令形状整除
        assert (
            self.bM % (atom_lay_M * mmaM) == 0
        ), "bM必须能被MMA指令形状整除"
        assert (
            self.bN % (atom_lay_N * mmaN) == 0
        ), "bN必须能被MMA指令形状整除"
        assert atom_lay_K == 1, "此示例不支持原子布局K > 1"
        assert self.bK % mmaK == 0, "bK必须能被MMA指令形状整除"
        assert self.num_stages >= 3, "流水线阶段数必须大于等于3"

    @cute.jit  # 使用CuTe的JIT编译装饰器
    def __call__(
        self,
        mA: cute.Tensor,  # 输入矩阵A的CuTe张量
        mB: cute.Tensor,  # 输入矩阵B的CuTe张量
        mC: cute.Tensor,  # 输出矩阵C的CuTe张量
        epilogue_op: cutlass.Constexpr = lambda x: x,  # 尾声操作，默认为恒等函数
    ):
        """
        执行GEMM运算的主函数
        
        Args:
            mA: 输入矩阵A的CuTe张量
            mB: 输入矩阵B的CuTe张量
            mC: 输出矩阵C的CuTe张量
            epilogue_op: 尾声操作函数，用于在写回结果前进行额外处理
        """
        # 网格将问题的M、N和L维度按照分块形状(bM, bN, 1)的相应模式进行划分。
        # K维度在块内通过多阶段流水线处理。

        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)  # 获取矩阵A的主序模式
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)  # 获取矩阵B的主序模式
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)  # 获取矩阵C的主序模式

        # ///////////////////////////////////////////////////////////////////////////////
        # 共享内存布局：
        # ///////////////////////////////////////////////////////////////////////////////

        # 创建具有所提供分块大小和阶段数所需大小的布局
        # (阶段用于K维度)，该布局也被分割成64x8或8x32布局原子。
        # 设置混洗(swizzle)以便共享内存->寄存器拷贝的原子不会遇到存储体冲突

        # 假设输入是16B对齐的
        ab_copy_bits = 128  # A和B矩阵拷贝的位数
        sA_layout = self._make_smem_layout_AB(  # 创建A矩阵的共享内存布局
            mA.element_type,        # A矩阵的元素类型
            self.a_major_mode,      # A矩阵的主序模式
            ab_copy_bits,           # 拷贝位数
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),  # 分块大小和阶段数
        )
        sB_layout = self._make_smem_layout_AB(  # 创建B矩阵的共享内存布局
            mB.element_type,        # B矩阵的元素类型
            self.b_major_mode,      # B矩阵的主序模式
            ab_copy_bits,           # 拷贝位数
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),  # 分块大小和阶段数
        )

        # 创建类似的布局但不包含流水线阶段数或布局原子
        sC_layout = self._make_smem_layout_C(  # 创建C矩阵的共享内存布局
            mC.element_type,        # C矩阵的元素类型
            self.c_major_mode,      # C矩阵的主序模式
            ab_copy_bits,           # 拷贝位数
            (self.cta_tiler[0], self.cta_tiler[1]),  # 分块大小
        )

        # 为A、B操作分配的共享内存将在C操作时被覆写。
        # 这是为了通过减少每个块请求的共享内存大小来提高性能
        smem_size = max(  # 计算所需的共享内存大小
            cute.size_in_bytes(mC.element_type, sC_layout),  # C矩阵所需的共享内存
            cute.size_in_bytes(mA.element_type, sA_layout)   # A矩阵所需的共享内存
            + cute.size_in_bytes(mB.element_type, sB_layout),  # 加上B矩阵所需的共享内存
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # 分块拷贝：
        # tA/tB/tC的主序性遵循gA/gB/gC的主序性，
        # 使得能够合并访问全局内存，以便在全局内存和共享内存之间更快地传输数据。
        # ///////////////////////////////////////////////////////////////////////////////

        # 为全局内存到共享内存的异步拷贝创建拷贝原子
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(  # 全局内存到共享内存的拷贝操作
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL  # 使用全局缓存模式
            ),
            mA.element_type,      # 元素类型
            num_bits_per_copy=ab_copy_bits,  # 每次拷贝的位数
        )

        # 从拷贝原子创建用于分块拷贝的线程布局，
        # 其中线程布局简单地遵循张量的主维度
        tiled_copy_A = self._make_gmem_tiled_copy_AB(  # 创建A矩阵的全局内存分块拷贝
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(  # 创建B矩阵的全局内存分块拷贝
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits
        )

        # 为尾声创建同步拷贝原子和线程布局
        c_copy_bits = 128  # C矩阵拷贝的位数
        atom_sync_copy = cute.make_copy_atom(  # 创建同步拷贝原子
            cute.nvgpu.CopyUniversalOp(),  # 通用拷贝操作
            mC.element_type,      # C矩阵的元素类型
            num_bits_per_copy=c_copy_bits,  # 每次拷贝的位数
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(  # 创建C矩阵的全局内存分块拷贝
            atom_sync_copy, mC.element_type, self.c_major_mode, c_copy_bits
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # 分块MMA
        # ///////////////////////////////////////////////////////////////////////////////

        # 创建形状为16x8x16的MMA原子，对应MNK维度
        op = cute.nvgpu.warp.MmaF16BF16Op(  # 创建FP16/BF16 MMA操作
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )

        permutation_mnk = (  # 计算MNK的排列
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],  # M维度排列
            # 如果原子布局的N模式为1，为了利用最大的合并共享内存->寄存器拷贝，
            # 将分块mma的N模式设置为16
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,  # N维度排列
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],  # K维度排列
        )

        # 创建一个分块mma，根据指定布局对原子进行分块。
        # 对于2x2x1原子布局，mma原子被复制4次，
        # 在M方向复制两次，在N方向复制两次
        tC = cute.make_layout(self.atom_layout_mnk)  # 创建原子布局
        tiled_mma = cute.make_tiled_mma(  # 创建分块MMA
            op,                    # MMA操作
            tC,                    # 线程布局
            permutation_mnk=permutation_mnk,  # MNK排列
        )

        # 网格维度：((m + BLK_M - 1) // BLK_M, (n + BLK_N - 1) // BLK_N, l)
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))  # 计算网格维度

        # 添加线程块光栅化以提高数据重用
        raster_factor = 1  # 光栅化因子
        grid_dim_n = cute.size(grid_dim[1])  # 获取N维度的网格大小
        # 选择阈值以避免产生太多无操作CTA
        if grid_dim_n > 5:      # 如果N维度网格大小大于5
            raster_factor = 8   # 设置光栅化因子为8
        elif grid_dim_n > 2:    # 如果N维度网格大小大于2
            raster_factor = 4   # 设置光栅化因子为4
        elif grid_dim_n > 1:    # 如果N维度网格大小大于1
            raster_factor = 2   # 设置光栅化因子为2
        
        # 重新映射的网格维度用于光栅化
        rasterization_remap_grid_dim = (
            cute.size(grid_dim[0]) * raster_factor,  # 重新映射的X维度
            (cute.size(grid_dim[1]) + raster_factor - 1) // raster_factor,  # 重新映射的Y维度
            cute.size(grid_dim[2]),  # Z维度保持不变
        )

        # 启动内核
        self.kernel(
            mA,                    # 矩阵A
            mB,                    # 矩阵B
            mC,                    # 矩阵C
            sA_layout,             # A的共享内存布局
            sB_layout,             # B的共享内存布局
            sC_layout,             # C的共享内存布局
            tiled_copy_A,          # A的分块拷贝
            tiled_copy_B,          # B的分块拷贝
            tiled_copy_C,          # C的分块拷贝
            tiled_mma,             # 分块MMA
            raster_factor,         # 光栅化因子
            epilogue_op,           # 尾声操作
        ).launch(
            grid=rasterization_remap_grid_dim,  # 网格维度
            block=[self.num_threads, 1, 1],     # 块维度
            smem=smem_size,                     # 共享内存大小
        )

    @cute.kernel  # CuTe内核装饰器
    def kernel(
        self,
        mA: cute.Tensor,                    # 矩阵A
        mB: cute.Tensor,                    # 矩阵B
        mC: cute.Tensor,                    # 矩阵C
        sA_layout: cute.ComposedLayout,     # A的共享内存布局
        sB_layout: cute.ComposedLayout,     # B的共享内存布局
        sC_layout: cute.ComposedLayout,     # C的共享内存布局
        tiled_copy_A: cute.TiledCopy,       # A的分块拷贝
        tiled_copy_B: cute.TiledCopy,       # B的分块拷贝
        tiled_copy_C: cute.TiledCopy,       # C的分块拷贝
        tiled_mma: cute.TiledMma,           # 分块MMA
        rasterization_factor: cutlass.Int32, # 光栅化因子
        epilogue_op: cutlass.Constexpr = lambda x: x,  # 尾声操作
    ):
        """
        GEMM内核的主体函数
        
        这是在GPU上执行的内核函数，包含了完整的GEMM计算流程：
        1. 数据预取和流水线设置
        2. 主循环中的矩阵乘法计算
        3. 尾声处理和结果写回
        """
        # 获取线程索引和块索引
        tidx, _, _ = cute.arch.thread_idx()    # 线程在块内的索引
        bidx, bidy, bidz = cute.arch.block_idx()  # 块在网格中的索引
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))  # 计算网格维度
        
        # 应用光栅化映射
        offset_tile_x, offset_tile_y = self.raster_tile(
            bidx, bidy, rasterization_factor
        )
        
        # 如果CTA超出范围则提前退出
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass  # 无操作，直接返回
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)  # 分块器坐标

            # ///////////////////////////////////////////////////////////////////////////////
            # 为此线程块获取适当的分块。
            # gA: (BLK_M, BLK_N, k), gB: (BLK_N, BLK_K, k), gC: (BLK_M, BLK_N)
            # ///////////////////////////////////////////////////////////////////////////////
            # 对矩阵A进行局部分块
            gA = cute.local_tile(
                mA[None, None, bidz],    # 选择批次维度
                tiler=self.cta_tiler,    # 使用CTA分块器
                coord=tiler_coord,       # 分块坐标
                proj=(1, None, 1),       # 投影模式，保留K维度
            )
            # 对矩阵B进行局部分块
            gB = cute.local_tile(
                mB[None, None, bidz],    # 选择批次维度
                tiler=self.cta_tiler,    # 使用CTA分块器
                coord=tiler_coord,       # 分块坐标
                proj=(None, 1, 1),       # 投影模式，保留K维度
            )
            # 对矩阵C进行局部分块
            gC = cute.local_tile(
                mC[None, None, bidz],    # 选择批次维度
                tiler=self.cta_tiler,    # 使用CTA分块器
                coord=tiler_coord,       # 分块坐标
                proj=(1, 1, None),       # 投影模式，保留批次维度
            )

            # 默认情况下，如果张量k模式不能整除分块k大小，
            # 那么k维度的最后分块是不规则的。
            # 相反，当k不规则时，使第一个分块不规则。
            # 这允许我们首先处理不规则分块，避免在主循环内检查此条件。

            # residual_k是一个负数，表示在k维度上需要移动指针的量
            residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(
                gA, mode=[2]
            )

            # 在-k方向上移动gA/gB的指针
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB = cute.domain_offset((0, residual_k, 0), gB)
            # 输入是16B对齐的
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            # 为sA和sB构造单位布局(镜像全局张量，仅用于谓词判断)
            mcA = cute.make_identity_tensor(mA.layout.shape)  # 创建A的单位张量
            mcB = cute.make_identity_tensor(mB.layout.shape)  # 创建B的单位张量
            # 对单位张量A进行局部分块
            cA = cute.local_tile(
                mcA[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, None, 1),
            )
            # 对单位张量B进行局部分块
            cB = cute.local_tile(
                mcB[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(None, 1, 1),
            )

            # 应用相同的域偏移
            cA = cute.domain_offset((0, residual_k, 0), cA)
            cB = cute.domain_offset((0, residual_k, 0), cB)

            # ///////////////////////////////////////////////////////////////////////////////
            # 创建共享内存缓冲区并为此线程获取适当的片段。
            # sA:   (BLK_M, BLK_K, PIPE)       , sB:   (BLK_N, BLK_K, PIPE)
            # tAgA: (CPY, CPY_M, CPY_K, k)     , tBgB: (CPY, CPY_N, CPY_K, k)
            # tAsA: (CPY, CPY_M, CPY_K, PIPE)  , tBsB: (CPY, CPY_N, CPY_K, PIPE)
            # ///////////////////////////////////////////////////////////////////////////////
            # 共享内存缓冲区分配器
            smem = cutlass.utils.SmemAllocator()

            # 分配共享内存张量
            sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)  # A的共享内存张量
            sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)  # B的共享内存张量
            # C的共享内存张量重用A的内存空间
            sC = cute.make_tensor(
                cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout
            )

            # 获取线程级拷贝切片
            thr_copy_A = tiled_copy_A.get_slice(tidx)  # A的线程拷贝
            thr_copy_B = tiled_copy_B.get_slice(tidx)  # B的线程拷贝
            thr_copy_C = tiled_copy_C.get_slice(tidx)  # C的线程拷贝
            
            # 对全局和共享内存张量进行分区
            tAgA = thr_copy_A.partition_S(gA)    # A的全局内存分区
            tAsA = thr_copy_A.partition_D(sA)    # A的共享内存分区
            tBgB = thr_copy_B.partition_S(gB)    # B的全局内存分区
            tBsB = thr_copy_B.partition_D(sB)    # B的共享内存分区
            tCsC_epilogue = thr_copy_C.partition_S(sC)  # C的共享内存分区(尾声用)
            tCgC_epilogue = thr_copy_C.partition_D(gC)  # C的全局内存分区(尾声用)

            # 使用单位布局重复分区
            tAcA = thr_copy_A.partition_S(cA)    # A的单位张量分区
            tBcB = thr_copy_B.partition_S(cB)    # B的单位张量分区

            # ///////////////////////////////////////////////////////////////////////////////
            # 谓词：当problem_shape不是tile_shape的倍数时，标记需要拷贝的索引
            # ///////////////////////////////////////////////////////////////////////////////

            # 对于张量A(M/K)、B(N/K)和(在尾声中)C(M/N)的谓词判断，
            # 我们将以类似外积的方式计算它。沿一个维度的谓词判断
            # 被评估并存储在谓词张量中。然后，剩余维度的谓词判断
            # 稍后通过拷贝时的if/else分支处理。
            # 对于A和B，沿M/N的谓词布尔值存储在谓词张量中，
            # 沿K的谓词通过if/else分支处理。

            # 为M和N分配谓词张量。谓词检查在拷贝原子的粒度上进行，
            # 因此谓词张量不需要为拷贝原子内的单个元素
            # (例如，tAgA.shape[0][0]的元素)提供单独的布尔值。
            tApA = cute.make_fragment(  # A的谓词片段
                cute.make_layout(
                    (
                        tAgA.shape[0][1],      # 拷贝维度
                        cute.size(tAgA, mode=[1]),  # M维度大小
                        cute.size(tAgA, mode=[2]),  # K维度大小
                    ),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),  # 步长
                ),
                cutlass.Boolean,  # 布尔类型
            )
            tBpB = cute.make_fragment(  # B的谓词片段
                cute.make_layout(
                    (
                        tBsB.shape[0][1],      # 拷贝维度
                        cute.size(tBsB, mode=[1]),  # N维度大小
                        cute.size(tBsB, mode=[2]),  # K维度大小
                    ),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0),  # 步长
                ),
                cutlass.Boolean,  # 布尔类型
            )
            
            # 设置M/N边界的谓词
            for rest_v in range(tApA.shape[0]):     # 遍历拷贝维度
                for m in range(tApA.shape[1]):      # 遍历M维度
                    # 检查M维度是否在边界内
                    tApA[rest_v, m, 0] = cute.elem_less(
                        tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                    )
            for rest_v in range(tBpB.shape[0]):     # 遍历拷贝维度
                for n in range(tBpB.shape[1]):      # 遍历N维度
                    # 检查N维度是否在边界内
                    tBpB[rest_v, n, 0] = cute.elem_less(
                        tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                    )

            # ///////////////////////////////////////////////////////////////////////////////
            # 预取序言
            # ///////////////////////////////////////////////////////////////////////////////
            # 清空共享内存分块以处理谓词关闭的加载
            tAsA.fill(0)      # 将A的共享内存初始化为0
            tBsB.fill(0)      # 将B的共享内存初始化为0
            cute.arch.sync_threads()  # 线程同步

            # 开始第一个k分块的异步加载。这里我们通过沿k维度的if/else检查
            # 来处理k余数。因为我们将单位张量按residue_k偏移，
            # 并且因为单位张量是坐标张量，任何毒化的单位张量元素的值都小于-1
            num_smem_stages = cute.size(tAsA, mode=[3])  # 共享内存阶段数
            k_tile_count = cute.size(tAgA, mode=[3])     # k分块数量
            k_tile_index = cutlass.Int32(0)              # k分块索引

            # 预取第一个k分块的A矩阵
            for k in range(tApA.shape[2]):  # 遍历k维度
                if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):  # 检查k边界
                    cute.copy(  # 异步拷贝
                        tiled_copy_A,
                        tAgA[None, None, k, k_tile_index],  # 源：全局内存
                        tAsA[None, None, k, 0],             # 目标：共享内存
                        pred=tApA[None, None, k],           # 谓词
                    )
            # 预取第一个k分块的B矩阵
            for k in range(tBpB.shape[2]):  # 遍历k维度
                if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):  # 检查k边界
                    cute.copy(  # 异步拷贝
                        tiled_copy_B,
                        tBgB[None, None, k, k_tile_index],  # 源：全局内存
                        tBsB[None, None, k, 0],             # 目标：共享内存
                        pred=tBpB[None, None, k],           # 谓词
                    )
            k_tile_index = k_tile_index + 1  # 增加k分块索引
            cute.arch.cp_async_commit_group()  # 提交异步拷贝组

            # 开始其余k分块的异步加载
            for k_tile in range(1, num_smem_stages - 1):  # 遍历剩余的共享内存阶段
                if k_tile == k_tile_count:  # 如果超出k分块数量
                    tApA.fill(0)  # 清空A的谓词
                    tBpB.fill(0
整数
        except ValueError:
            raise argparse.ArgumentTypeError(
                "格式无效。期望逗号分隔的整数。"
            )

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="在GPU上使用CuTe进行多阶段块矩阵乘法的示例"
    )
    parser.add_argument(
        "--mnkl", type=parse_comma_separated_ints, default=(112, 136, 40, 1)
    )  # 矩阵维度参数
    parser.add_argument(
        "--atom_layout_mnk", type=parse_comma_separated_ints, default=(2, 2, 1)
    )  # 原子布局参数
    parser.add_argument(
        "--ab_dtype",
        type=cutlass.dtype,
        choices=[cutlass.Float16],  # 支持的A和B矩阵数据类型
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        choices=[cutlass.Float32],  # 支持的累加器数据类型
        default=cutlass.Float32,
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        choices=[cutlass.Float16],  # 支持的C矩阵数据类型
        default=cutlass.Float16,
    )
    parser.add_argument("--a_major", choices=["k", "m"], default="m")  # A矩阵主序
    parser.add_argument("--b_major", choices=["k", "n"], default="n")  # B矩阵主序
    parser.add_argument("--c_major", choices=["n", "m"], default="n")  # C矩阵主序
    parser.add_argument("--warmup_iterations", default=2, type=int)    # 预热迭代次数
    parser.add_argument("--iterations", default=100, type=int)         # 测试迭代次数
    parser.add_argument("--skip_ref_check", action="store_true")       # 是否跳过参考检查
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="使用循环缓冲张量集以确保L2缓存冷启动",  # 冷L2缓存帮助文本
    )

    args = parser.parse_args()  # 解析命令行参数
    run(  # 调用运行函数
        args.a_major,           # A矩阵主序
        args.b_major,           # B矩阵主序
        args.c_major,           # C矩阵主序
        args.ab_dtype,          # A和B矩阵数据类型
        args.c_dtype,           # C矩阵数据类型
        args.acc_dtype,         # 累加器数据类型
        args.mnkl,              # 矩阵维度
        args.atom_layout_mnk,   # 原子布局
        args.warmup_iterations, # 预热迭代次数
        args.iterations,        # 测试迭代次数
        args.skip_ref_check,    # 是否跳过参考检查
        args.use_cold_l2,       # 是否使用冷L2缓存
    )
    print("PASS")  # 打印成功信息