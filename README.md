# SPH 平板泊肃叶流动模拟

基于**光滑粒子流体动力学（Smoothed Particle Hydrodynamics, SPH）**的二维平板泊肃叶流（Poiseuille flow）数值模拟程序，采用 **MATLAB + C/MEX + OpenMP** 实现。

当前主线版本已经合入两次关键稳定性修复：

1. 2026-03-19：厚壁通道 + dual-criteria 时间推进，显著抑制长时速度脉动。
2. 2026-03-20：周期边界 ghost 邻居插入，修复接缝漏邻居导致的剖面不连续。

> 2026-02-21 历史备注：曾尝试将计算链路整体迁移到 GPU（含自定义 CUDA MEX），但实际性能与 CPU 占用未达预期，现已回滚到当前 CPU/MEX 稳定方案。

## 近期关键修复

| 日期 | Commit | 已定位问题 | 主要修复内容 | 效果摘要 |
|------|--------|------------|--------------|----------|
| 2026-03-19 | `3eb6de087f5c50f2b4ec54f0d3663e0a098ccba8` | 单层 CFL + 较强 transport correction 会放大近壁横向扰动；单层 shell 壁面支撑不足；主循环末尾硬 `bounding_from_wall` 会持续注入非物理扰动 | 主推进改为 dual-criteria 时间推进；壁面由单层 shell 改为 `BW = 4dp` 厚壁粒子区；移除主循环硬 wall clip；transport correction 改为外层步执行，并将 `transport_coeff` 收敛到 `0.1`；周期边界改为 immediate wrap | `u_y RMS` 约从 `16.56% Umax` 降到 `1.32% Umax`，中心带 `u_x std` 降到约 `0.72% Umax`，20 s 全长算例 `L2 = 2.43%` |
| 2026-03-20 | `3d620e801bb5af4552c18c22c7a96deaa876515f` | 周期边界搜索阶段只检查 wrapped cell 的 `3x3` 邻域，导致接缝两侧漏掉实际仍在 `2h` 内的跨周期近邻 | 在 `sph_neighbor_search_mex` 中引入 `CellEntryBuffer`，为靠近 `x` 周期边界的粒子插入平移 ghost entry；对 real/ghost entry 使用 `seen_neighbor` 去重；保留 minimum-image 距离修正 | `edge ux gap` 从 `0.0587` 降到 `0.0232`，`edge profile L2` 从 `0.0250` 降到 `0.00514`，最新 20 s 全长算例 `L2 = 0.49%` |

这两次修复之后，当前默认实现的核心特征是：

- dual-criteria 时间推进
- 厚壁粒子区 + 算子内 no-slip / no-penetration
- 周期 ghost 邻居插入 + minimum-image 距离修正

## 物理问题

模拟两平行板之间的重力驱动层流。该问题具有解析解，适合作为 SPH 方法的标准验证算例。

解析解：

$$
u(y) = \frac{g}{2\nu} y(H-y)
$$

其中：

- `g` 为等效体积力（压力梯度）
- `nu = mu / rho0` 为运动粘度
- `H = DH` 为通道高度

## 默认参数

默认参数由根目录下的 `config.ini` 提供：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DH` | `1.0 m` | 流体域高度 |
| `DL` | `3.0 m` | 流体域长度，X 方向周期性 |
| `dp` | `0.04 m` | 粒子间距 |
| `rho0` | `1.0 kg/m^3` | 参考密度 |
| `mu` | `0.1 Pa·s` | 动力粘度 |
| `U_bulk` | `0.667 m/s` | 截面平均速度 |
| `c_f` | `10.0` | 人工声速因子，`c = c_f * U_bulk` |
| `end_time` | `5.0 s` | 仿真终止时间 |
| `output_interval` | `0.2 s` | 输出间隔 |
| `sort_interval` | `100` | 粒子排序步数间隔 |
| `transport_coeff` | `0.1` | 当前代码内固定的保守 shifting 系数 |

## 数值方法与当前实现

### 时间推进

- 采用**dual-criteria 时间推进**
- 外层时间步 `Dt` 由对流/黏性尺度控制
- 内层时间步 `dt` 由声学 CFL 控制，在一个 `Dt` 内执行多次压力/密度松弛

### 邻居搜索

- 使用 **Cell-Linked List**
- 单元尺寸取 `2h`
- X 方向采用**周期 ghost entry 插入 + minimum-image 距离修正**
- 对 real/ghost entry 使用 `seen_neighbor` 去重，避免跨周期重复计数

### 边界处理

- X 方向：周期边界，位置采用 immediate wrap
- Y 方向：`BW = 4dp` 的厚壁粒子区
- no-slip / no-penetration 由物理算子内部的镜像壁速与虚拟壁压处理，不再依赖主循环末尾的硬裁剪

### 主要物理离散

- 密度重初始化：核函数求和
- 核梯度修正：Kernel Gradient Correction
- 压力模型：弱可压缩状态方程 `p = p0 * (rho / rho0 - 1)`
- 粘性项：基于核梯度修正的层流粘性力
- 传输修正：Transport Velocity Correction，仅在外层步执行

## 运行方式

### 环境要求

- MATLAB R2018b 或更高版本
- 支持 OpenMP 的 C 编译器
  - Windows：推荐 MSVC
  - Linux/macOS：推荐 GCC

### 推荐运行方式

```bash
matlab -batch "run('SPH_Poiseuille.m')"
```

程序会自动完成：

1. 检测并编译 MEX 模块（如有需要）
2. 读取 `config.ini`
3. 初始化流体粒子与厚壁粒子区
4. 执行 SPH 主循环
5. 输出结果图 `SPH_Poiseuille_result.png`
6. 输出中间截面演化图 `SPH_centerline_profile_evolution.png`

### 交互式运行

```matlab
run('SPH_Poiseuille.m')
```

## MEX 编译

默认情况下，主脚本会在首次运行时自动编译并把产物放到 `build/` 目录。

如需手动编译，可使用：

```matlab
% Windows (MSVC)
mex -R2018a -O COMPFLAGS="$COMPFLAGS /openmp" -output sph_neighbor_search_mex -outdir build mex/sph_neighbor_search_mex.c
mex -R2018a -O COMPFLAGS="$COMPFLAGS /openmp" -output sph_physics_shell_mex -outdir build mex/sph_physics_mex.c

% Linux/macOS (GCC)
mex -R2018a -O CFLAGS="$CFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" -output sph_neighbor_search_mex -outdir build mex/sph_neighbor_search_mex.c
mex -R2018a -O CFLAGS="$CFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" -output sph_physics_shell_mex -outdir build mex/sph_physics_mex.c
```

说明：

- C 源文件位于 `mex/`
- 编译产物位于 `build/`
- `mex/sph_physics_mex.c` 同时提供底层物理 mode 与兼容性门面 mode：`advance_shell_step`、`wall_shear_monitor`

## 输出与验证

![模拟结果](SPH_Poiseuille_result.png?v=2)

- 左图：速度剖面与解析解对比
- 右图：通道速度场与厚壁区域示意

按 2026-03-20 提交 `3d620e801bb5af4552c18c22c7a96deaa876515f` 中记录的 20 s 长时验证结果：

- 周期接缝两侧剖面已明显连续
- `edge ux gap = 0.0232`
- `edge profile L2 = 0.00514`
- 全长算例 `L2 = 0.49%`

## 参数配置

所有物理与仿真控制参数均通过 `config.ini` 管理。

### `[physical]`

- `DL`
- `DH`
- `dp`
- `rho0`
- `mu`
- `U_bulk`
- `c_f`

### `[simulation]`

- `end_time`
- `output_interval`
- `sort_interval`
- `restart_from_file`

### 可选环境变量覆盖

为方便测试或隔离输出目录，主脚本支持以下环境变量：

- `SPH_CONFIG_OVERRIDE`
- `SPH_RESTART_PATH_OVERRIDE`
- `SPH_RESULT_PNG_OVERRIDE`
- `SPH_PROFILE_PNG_OVERRIDE`

未设置时，仍默认读写项目根目录下的对应文件。

## 文件结构

| 文件/目录 | 说明 |
|-----------|------|
| `SPH_Poiseuille.m` | 主 MATLAB 脚本，负责配置、初始化、dual-criteria 主循环、监控与后处理 |
| `build_shell_wall_particles.m` | 厚壁粒子区几何生成函数 |
| `config.ini` | 物理参数与仿真控制参数 |
| `mex/sph_neighbor_search_mex.c` | 周期邻居搜索 MEX，包含 ghost entry 插入与去重逻辑 |
| `mex/sph_physics_mex.c` | 物理算子 MEX，包含密度修正、粘性、传输修正、两阶段积分与壁面剪应力监控 |
| `build/` | MEX 编译输出目录 |
| `restart.mat` | 断点续算文件 |
| `SPH_Poiseuille_result.png` | 结果对比图 |
| `SPH_centerline_profile_evolution.png` | 中间截面速度剖面演化图 |

## 性能与调参建议

| 配置 | `dp` | `end_time` | 预期用途 |
|------|------|------------|----------|
| 快速调试 | `0.05` | `1.0` | 算法验证、快速迭代 |
| 平衡配置 | `0.0333` | `1.5` | 日常计算 |
| 高精度 | `0.025` | `1.5` | 更高分辨率结果 |

调参入口主要是 `config.ini` 中的 `dp` 与 `end_time`。

## 参考文献

- Morris, J. P., Fox, P. J., & Zhu, Y. (1997). Modeling low Reynolds number incompressible flows using SPH. *Journal of Computational Physics*, 136(1), 214-226.
- Liu, G. R., & Liu, M. B. (2003). *Smoothed Particle Hydrodynamics: A Meshfree Particle Method*. World Scientific.
- Zhang, C., et al. (2020). SPHinXsys: An open-source multi-physics and multi-resolution library based on smoothed particle hydrodynamics. *Computer Physics Communications*, 267, 108066.

## 许可证

本项目仅供学习与交流使用。
