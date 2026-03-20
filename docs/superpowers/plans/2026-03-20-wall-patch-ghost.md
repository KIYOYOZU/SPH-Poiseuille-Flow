# Wall Patch Ghost Velocity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the existing thick-wall geometry and replace the zero-wall-velocity wall treatment with patch-based ghost velocity reconstruction.

**Architecture:** MATLAB owns patch metadata and per-step ghost velocity reconstruction because the patch target force is configuration-level logic. The existing MEX kernels stay mostly unchanged and only consume the reconstructed ghost velocity in the wall branches that already depend on `wall_vel`.

**Tech Stack:** MATLAB, MEX C, git worktree

---

### Task 1: 建立可失败的自测入口

**Files:**
- Create: `wall_patch_ghost_selftest.m`

- [ ] **Step 1: 写 patch ghost velocity 自测**

覆盖：
- 壁粒子 patch 编号是否正确
- patch 汇总系数是否得到非零值
- ghost velocity 是否按 patch 回写

- [ ] **Step 2: 运行自测，确认先失败**

Run: `matlab -batch "run('wall_patch_ghost_selftest.m')"`
Expected: FAIL，提示缺少 patch 元数据或 ghost velocity 更新函数

- [ ] **Step 3: 最小实现 patch 元数据与 ghost velocity 更新**

修改：
- `build_shell_wall_particles.m`
- `SPH_Poiseuille.m`

- [ ] **Step 4: 重新运行自测，确认通过**

Run: `matlab -batch "run('wall_patch_ghost_selftest.m')"`
Expected: PASS

### Task 2: 让 MEX 壁面算子消费 ghost velocity

**Files:**
- Modify: `mex/sph_physics_mex.c`
- Update: `wall_patch_ghost_selftest.m`

- [ ] **Step 1: 扩展自测覆盖 MEX 壁面分支**

覆盖：
- `viscous_force` 壁面项读取 ghost velocity
- `integration_2nd` 壁面项读取 ghost velocity

- [ ] **Step 2: 运行自测，确认先失败**

Run: `matlab -batch "run('wall_patch_ghost_selftest.m')"`
Expected: FAIL，旧镜像壁速度逻辑与预期不一致

- [ ] **Step 3: 对 MEX 做最小改动**

只改：
- `mode_viscous_force`
- `mode_integration_2nd`

- [ ] **Step 4: 重新运行自测，确认通过**

Run: `matlab -batch "run('wall_patch_ghost_selftest.m')"`
Expected: PASS

### Task 3: 主脚本回归验证

**Files:**
- Modify: `SPH_Poiseuille.m`
- Update: `README.md`

- [ ] **Step 1: 把 ghost velocity 更新接入主循环**

放在外层步、粘性力与声学子步之前。

- [ ] **Step 2: 运行主脚本回归**

Run: `matlab -batch "run('SPH_Poiseuille.m')"`
Expected: 成功完成，输出结果文件更新

- [ ] **Step 3: 记录变更说明**

更新 `README.md` 的壁面处理描述与验证说明。
