# SPH 平板泊肃叶流动模拟

基于**光滑粒子流体动力学 (Smoothed Particle Hydrodynamics, SPH)** 方法的二维平板泊肃叶流 (Poiseuille Flow) 数值模拟程序。采用 **MATLAB + C/MEX + OpenMP** 实现高性能计算。

## Tips

- **2026-02-21 备注：** 曾尝试将计算链路完全迁移到 GPU（含自定义 CUDA MEX），但实际性能与 CPU 占用未达预期，现已整体回滚到 GitHub 稳定版本。

## 物理问题

模拟两平行板之间的重力驱动层流（泊肃叶流动），该问题具有解析解，适合用于验证 SPH 方法的准确性。

**解析解：**

$$u(y) = \frac{g}{2\nu} y(H - y)$$

其中 $g$ 为体积力（等效压力梯度），$\nu = \mu/\rho_0$ 为运动粘度。

**默认模拟参数（config.ini）：**

| 参数 | 数值 | 说明 |
|------|------|------|
| 板间距 DH | 1.0 m | Y 方向流体域高度 |
| 通道长度 DL | 3.0 m | X 方向流体域长度（周期性边界） |
| 粒子间距 dp | 0.04 m | 空间分辨率（约 1875 粒子） |
| 参考密度 ρ₀ | 1.0 kg/m³ | 流体参考密度 |
| 动力粘度 μ | 0.1 Pa·s | 流体粘度 |
| 截面平均速度 U_bulk | 0.667 m/s | U_max = 1.5*U_bulk, Re = ρ*U_max*DH/μ |
| 声速因子 c_f | 10.0 | 人工声速 c = c_f × U_bulk |
| 仿真时间 | 5.0 s | 物理时间（通常 3-4s 达到稳态） |

## 数值方法

### 核心算法
- **核函数：** 2D Cubic Spline (Wendland C2)
- **邻居搜索：** Cell-Linked List（网格链表法），支持周期性边界
- **计算加速：** 模块化 MEX（邻居搜索 + 高层物理门面 + OpenMP 多线程）
- **时间积分：** 单层 CFL 时间步进（CFL = 0.3）
  - 统一时间步：`dt = 0.3 * h / (c_f + |u|_max)`，并按输出点与终止时间裁剪
- **密度计算：** 核函数求和 + 核梯度修正（Kernel Gradient Correction）
- **压力方程：** 弱可压缩状态方程 p = p₀(ρ/ρ₀ - 1)
- **粘性模型：** 基于核梯度修正的粘性力
- **传输修正：** Transport Velocity Correction（抑制数值扩散）
- **边界条件：**
  - X 方向：周期性边界（缓冲层宽度 4dp），fluid-wall 对同样做周期最小像距
  - Y 方向：单层 shell 壁面粒子（中面位于 `y=-0.5dp` 和 `y=DH+0.5dp`，厚度 `dp`）+ 唯一 no-slip + 壁面防穿透

### 架构特点
- **配置驱动：** 所有参数通过 `config.ini` 配置
- **单主文件壳层：** `SPH_Poiseuille.m` 负责配置、初始化、主循环框架、监控与后处理，保留 `local functions` 完成组织
- **Restart 机制：** 支持从断点续算（带配置签名验证）
- **粒子排序：** 定期按 Cell ID 排序提升缓存命中率
- **自动编译：** MEX 模块首次运行时自动检测并编译

## 模拟结果

![模拟结果](SPH_Poiseuille_result.png?v=2)

- **左图：** 速度剖面对比（黑线 = 解析解，红点 = SPH 结果）
- **右图：** 粒子分布与速度场（彩色 = 流体粒子，灰色 = 壁面虚粒子）

**L2 相对误差 < 1%**，SPH 结果与解析解高度吻合。

## 运行方式

### 环境要求

- MATLAB R2018b 或更高版本
- 支持 OpenMP 的 C 编译器（推荐：Windows 使用 MSVC，Linux/macOS 使用 GCC）

### 快速开始

**方式1：MATLAB 交互式运行**
```matlab
run('SPH_Poiseuille.m')
```

**方式2：命令行批处理模式**
```bash
matlab -batch "run('SPH_Poiseuille.m')"
```

程序将自动完成：
1. MEX 模块检测与编译（首次运行）
2. 从 `config.ini` 读取参数
3. 粒子初始化（流体粒子 + 壁面虚粒子）
4. SPH 主循环计算
5. 输出结果对比图 `SPH_Poiseuille_result.png`

### MEX 编译

程序首次运行时会**自动检测并编译** MEX 加速模块，编译输出存放在 `build/` 目录，无需手动操作。

**手动编译（可选）：**

```matlab
% Windows (MSVC)
mex -R2018a -O COMPFLAGS="$COMPFLAGS /openmp" -output sph_neighbor_search_mex -outdir build mex/sph_neighbor_search_mex.c
mex -R2018a -O COMPFLAGS="$COMPFLAGS /openmp" -output sph_physics_shell_mex -outdir build mex/sph_physics_mex.c

% Linux/macOS (GCC)
mex -R2018a -O CFLAGS="$CFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" -output sph_neighbor_search_mex -outdir build mex/sph_neighbor_search_mex.c
mex -R2018a -O CFLAGS="$CFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" -output sph_physics_shell_mex -outdir build mex/sph_physics_mex.c
```

**注意：**
- C 源文件位于 `mex/` 目录，编译产物存放在 `build/` 目录
- `mex/sph_physics_mex.c` 现在同时提供底层 mode 和高层门面 mode：`advance_shell_step`、`wall_shear_monitor`

### 参数配置

所有物理和数值参数通过 `config.ini` 配置，支持以下参数：

**物理参数 [physical]：**
- `DL`：流体域长度（X 方向，周期性边界）
- `DH`：流体域高度（Y 方向，壁面边界）
- `dp`：粒子间距（决定空间分辨率）
- `rho0`：参考密度
- `mu`：动力粘度
- `U_bulk`：截面平均速度（U_max = 1.5*U_bulk）
- `c_f`：声速因子（人工声速 = c_f × U_bulk）

**仿真控制 [simulation]：**
- `end_time`：仿真终止时间（秒）
- `output_interval`：输出间隔（秒）
- `sort_interval`：粒子排序间隔（步数）
- `restart_from_file`：是否从 restart 文件续算（0=从零开始，1=续算）

### 环境变量控制

默认运行仍直接读取项目根目录下的 `config.ini` 并输出到默认结果文件。为测试或批处理隔离目录，主脚本支持以下可选环境变量覆盖：

- `SPH_CONFIG_OVERRIDE`
- `SPH_RESTART_PATH_OVERRIDE`
- `SPH_RESULT_PNG_OVERRIDE`
- `SPH_PROFILE_PNG_OVERRIDE`

不设置这些环境变量时，行为与原先保持一致。

## 文件说明

| 文件/目录 | 说明 |
|------|------|
| `SPH_Poiseuille.m` | 单主文件 MATLAB 壳层，负责配置、初始化、主循环框架、监控与后处理 |
| `build_shell_wall_particles.m` | 单层 shell 壁面粒子几何生成函数 |
| `config.ini` | 参数配置文件（物理参数 + 仿真控制） |
| `mex/sph_neighbor_search_mex.c` | 邻居搜索 MEX（C+OpenMP）：Cell-Linked List + 周期性边界 |
| `mex/sph_physics_mex.c` | 物理计算 MEX（C+OpenMP）：底层物理 mode + 高层门面 `advance_shell_step` / `wall_shear_monitor` |
| `build/` | MEX 编译输出目录（自动生成，已在 `.gitignore` 中忽略） |
| `restart.mat` | Restart 状态文件（运行过程中自动生成，用于续算） |
| `SPH_Poiseuille_result.png` | 模拟结果可视化 |

## 性能优化建议

### 当前性能瓶颈
- **粒子分辨率**：dp=0.025 → 约 6000 粒子，计算复杂度 O(N×邻居数)
- **仿真时间**：3.0 秒（泊肃叶流通常 1-1.5 秒达到稳态）

### 优化方案（按效果排序）

| 配置 | dp | end_time | 粒子数 | 预期提速 | 适用场景 |
|------|----|---------:|-------:|---------:|----------|
| **快速调试** | 0.05 | 1.0 | ~1500 | **4-9x** | 算法验证、快速迭代 |
| **平衡配置** | 0.0333 | 1.5 | ~2700 | **2-3x** | 日常计算、合理精度 |
| **高精度** | 0.025 | 1.5 | ~6000 | **1.5-2x** | 发表级结果 |

**修改方式：** 编辑 `config.ini` 中的 `dp` 和 `end_time` 参数。

## 技术特性

### 1. 单层 CFL 时间步进（Single CFL Stepping）
- **统一时间步**：`dt = 0.3 * h / (c_f + |u|_max)`
- **时间裁剪**：每步按 `output_interval` 和 `end_time` 自动截断，保证输出点对齐
- **推进顺序**：每步依次执行密度修正、粘性力、传输修正、两阶段积分和边界处理

### 2. 核梯度修正（Kernel Gradient Correction）
- 自动计算修正矩阵 B（2×2）
- 修正后的核梯度：∇W_corrected = B · ∇W_original
- 提升边界附近的计算精度

### 3. 传输修正（Transport Velocity Correction）
- 抑制数值扩散，保持粒子分布均匀性
- 基于粒子数密度梯度计算修正速度
- 参数：`transport_coeff = 0.25`

### 4. 单层 Shell 壁面（Single-Layer Shell Wall）
- 上下壁面各只有一层中面粒子
- 壁面中面位置固定在 `y=-0.5dp` 和 `y=DH+0.5dp`
- 每个壁面粒子显式携带厚度 `wall_thickness=dp`
- 有效 wall 体积通过 `measure × thickness` 进入密度、粘性、压力和连续方程离散

### 5. 周期性边界（Periodic Boundary）
- X 方向周期性边界，缓冲层宽度 4dp
- 自动包裹粒子位置：x = mod(x, DL)
- 邻居搜索支持周期性镜像

### 6. Restart 机制
- 支持从断点续算，保存完整状态（位置、速度、密度、压力等）
- 配置签名验证：确保 restart 文件与当前配置一致
- 文件格式：MAT-file v7.3（支持大数据）

### 7. 粒子排序优化
- 定期按 Cell ID 排序（`sort_interval = 100` 步）
- 提升缓存命中率，加速邻居搜索
- 排序后自动重建邻居列表

## 参考文献

- Morris, J. P., Fox, P. J., & Zhu, Y. (1997). Modeling low Reynolds number incompressible flows using SPH. *Journal of Computational Physics*, 136(1), 214-226.
- Liu, G. R., & Liu, M. B. (2003). *Smoothed Particle Hydrodynamics: A Meshfree Particle Method*. World Scientific.
- Zhang, C., et al. (2020). SPHinXsys: An open-source multi-physics and multi-resolution library based on smoothed particle hydrodynamics. *Computer Physics Communications*, 267, 108066.

## 许可证

本项目仅供学习交流使用。
