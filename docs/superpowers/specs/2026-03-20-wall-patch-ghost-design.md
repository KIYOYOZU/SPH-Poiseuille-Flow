# 厚壁 Patch Ghost Velocity 设计

## 目标

在保留当前 `BW=4dp` 厚壁粒子区几何的前提下，把壁面处理从“`wall_vel=0` + 镜像速度”迁移为“按 patch 离散配平重构壁面 ghost velocity”，对齐参考算例的核心物理机制。

## 约束

- 不把当前 2D MATLAB/MEX 几何改成单层 shell。
- 只改必要链路，避免扩散到周期搜索和主体时间推进框架。
- 壁面 ghost velocity 每个外层步更新一次，随后由粘性项和第二阶段积分统一消费。

## 方案

### 1. 保留厚壁几何，补充 patch 元数据

继续使用 `build_shell_wall_particles.m` 生成上下厚壁粒子区，但新增每个壁粒子的：

- `patch_id`
- `patch_ix`
- `patch_side`
- `patch_area`

patch 只沿 `x` 方向切分；上下壁分别编号。这样能在 2D 通道里复用参考算例“共享 patch ghost velocity”的思路，而不强行引入 3D `z` 向 patch。

### 2. 在 MATLAB 侧重构 patch ghost velocity

新增一个 MATLAB 辅助函数，按当前 fluid-wall 邻居对累积离散系数：

- `sum_K_patch = Σ K_ij`
- `sum_Ku_patch = Σ K_ij * u_i`

其中 `K_ij` 与当前 MEX 粘性壁面分支保持同一离散口径。然后按目标壁面剪应力解出每个 patch 的共享 ghost velocity，并回写到该 patch 下全部壁粒子的 `wall_vel(:,1)`。

### 3. 最小改动下游算子

- `mode_viscous_force`：壁面分支直接使用重构后的 `wall_vel_j`
- `mode_integration_2nd`：壁面密度散度分支不再使用 `2*wall_vel-vel_i` 镜像，而是直接使用重构后的 ghost velocity
- `mode_integration_1st`：先保留当前压力壁面处理，只维持接口兼容，不额外扩大改动范围

## 验证

- 新增一个 MATLAB 自测脚本，先验证 patch 映射和 ghost velocity 重构
- 再验证 MEX 壁面粘性分支和 `integration_2nd` 确实读取到了更新后的 ghost velocity
- 最后运行主脚本做一次整体回归
