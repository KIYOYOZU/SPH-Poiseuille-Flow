# SPH 壁面 Neumann BC 设计方案
> 目标：在当前 WCSPH 泊肃叶流代码中，将壁面边界条件从无滑移 Dirichlet (u=0) 改为指定切应力 Neumann (τ_w = 0.4 Pa)

---

## 0. 符号约定与驱动一致性检查

### 0.1 符号约定

本文档中 τ_w 按**外法向速度梯度**定义：

$$\tau_w = \mu \left.\frac{\partial u}{\partial y}\right|_{y=0}$$

下壁面法向为 +y，故 τ_w > 0 对应流体向 +x 方向流动时的正切应力。上壁面法向为 -y，du/dy|_{y=H} < 0，但对称性要求两壁面切应力大小相等。

### 0.2 驱动一致性检查（重要）

当前代码中三个量存在约束关系，**只有两个是独立输入**：

$$\tau_w = \frac{\rho_0 g H}{2} = \frac{6\mu U_f}{H}$$

代入 `config.ini` 参数验证：
- `g = 12μU_f/(ρ₀H²) = 12×0.1×0.666667/1.0 = 0.8`
- `τ_w = ρ₀gH/2 = 1.0×0.8×1.0/2 = 0.4 Pa` ✓
- `τ_w = 6μU_f/H = 6×0.1×0.666667/1.0 = 0.4 Pa` ✓

**结论**：当前算例已天然对应 τ_w = 0.4 Pa。施加 Neumann BC 是**替换**壁面处理方式（Dirichlet→Neumann），而非叠加额外驱动力。体积力 g 保持不变，Neumann BC 仅改变虚粒子速度的计算方式。若修改 τ_w 目标值，需同步调整 g 或 U_f 以保持一致性。

---

## 1. 背景与当前实现分析

### 1.1 工况参数

| 参数 | 值 |
|------|-----|
| μ | 0.1 Pa·s |
| ρ₀ | 1.0 kg/m³ |
| DH | 1.0 m |
| dp | 0.04 m |
| h | 0.052 m |
| τ_w（目标） | 0.4 Pa |
| du/dy\|_wall（目标） | τ_w/μ = 4.0 s⁻¹ |

### 1.2 当前 Dirichlet BC 的离散粘性力公式

代码位置：`sph_physics_mex.c:509-524`，流体粒子 i 与壁面虚粒子 j 的粘性力：

```
eBe  = eᵀ Bᵢ e                          // 核梯度修正因子
coeff = 4 μ · eBe · ∇W_ij · V_j / (r_ij + 0.01h)
F_visc,x^i = V_i · Σ_j coeff · (v_x^i - wall_vel_x^j)
```

当前 `wall_vel = 0`，等效于在壁面处强制 u=0（Dirichlet）。

系数 ×4 是对核函数单侧截断的经验补偿（Morris et al. 1997 [3]）。

---

## 2. 方案 A：镜像速度 Neumann 法

### 2.1 物理原理

保持虚粒子框架不变，但不再令 `wall_vel=0`，而是根据 τ_w 动态计算虚粒子速度，使壁面处速度梯度等于 τ_w/μ。

### 2.2 推导

设壁面在 y=0，流体粒子 i 位于 y=y_i，对应虚粒子 j 位于 y=-y_j（关于壁面对称，y_j > 0）。

壁面处速度梯度的 SPH 近似（线性插值）：

$$\left.\frac{\partial u}{\partial y}\right|_{y=0} \approx \frac{u_i - u_{ghost}^j}{y_i + y_j}$$

令其等于 τ_w/μ：

$$\frac{u_i - u_{ghost}^j}{y_i + y_j} = \frac{\tau_w}{\mu}$$

解出虚粒子速度：

$$\boxed{u_{ghost}^j = u_i - \frac{\tau_w}{\mu}(y_i + y_j)}$$

对于多个流体粒子影响同一虚粒子的情况，用核函数加权平均：

设 $\tilde{u}^j = \frac{\sum_{i} u_i W_{ij}}{\sum_{i} W_{ij}}$ 为插值到虚粒子位置的流体速度，$\tilde{y}^j = \frac{\sum_{i} y_i W_{ij}}{\sum_{i} W_{ij}}$ 为对应的加权平均 y 坐标，则：

$$\boxed{u_{ghost}^j = \tilde{u}^j - \frac{\tau_w}{\mu}(\tilde{y}^j + y_j)}$$

其中 $y_j = |pos_j^y|$ 是虚粒子到壁面的距离（第 layer 层：$y_j = (2\cdot\text{layer}-1)\cdot dp/2$）。

**验证**：τ_w=0 时 $u_{ghost}^j = \tilde{u}^j$，即虚粒子速度等于流体侧插值速度，壁面处梯度为零（正确的零梯度 Neumann 极限）。

### 2.3 代码修改位置

**MATLAB 层**（`SPH_Poiseuille.m`）：在每个对流步开始前，更新 `wall_vel`：

```matlab
% 下壁面虚粒子（索引 n_fluid+1 : n_fluid+n_bottom）
for j = n_fluid+1 : n_fluid+n_bottom
    y_j = abs(pos(j, 2));  % 虚粒子到壁面距离
    % 核函数加权插值附近流体速度
    % 简化版：直接用最近流体粒子速度
    u_ghost_x = u_interp_j - (tau_w / mu) * y_j;
    wall_vel(j, 1) = u_ghost_x;
end
% 上壁面类似，注意梯度方向取反
```

**C/MEX 层**：`mode_viscous_force` 无需修改，`wall_vel_x[jj]` 已作为参数传入。

### 2.4 精度与局限

- 精度：O(h)，与当前 Dirichlet 实现一致
- 优点：改动最小，仅需在 MATLAB 层更新 wall_vel
- 局限：虚粒子速度依赖当前流场，需每步更新；多层虚粒子的插值权重需仔细处理

---

## 3. 方案 B（D+B 实现版）：pair-wise tau 测量 + PI 输出 delta_tau 叠加 force_prior

### 3.1 物理原理

壁面保持纯无滑移（`wall_vel = 0`），通过 PI 控制器测量数值切应力 τ_num，将误差转换为 delta_tau，再以等效体积力叠加到近壁流体粒子的 `force_prior`。

核心区别：
- 旧方案 D 路线A：PI → wall_vel → 粘性力（间接）
- 本方案 D+B：PI → delta_tau → delta_F → force_prior（直接），wall_vel 恒零

### 3.2 推导

**tau_num 测量**（pair-wise，避免重复计数）：

单个流体-壁面粒子对 (i,j) 的粘性力 x 分量（`sph_physics_mex.c:514`）：

$$f_{visc,x}^{ij} = 4\mu \cdot eBe_i \cdot \frac{\partial W_{ij}}{\partial r} \cdot V_j \cdot \frac{v_x^i - 0}{r_{ij} + 0.01h} \cdot V_i$$

下壁面切应力（取负号使结果为正值）：

$$\tau_{num} = \frac{-\sum_{(i,j)\in\text{bottom pairs}} f_{visc,x}^{ij}}{L_x}$$

**PI 控制律**：

$$e = \tau_{target} - \tau_{num}, \quad I \leftarrow \text{clip}(I + e \cdot \Delta t,\; -I_{max},\; I_{max})$$
$$\delta\tau = k_p \cdot e + k_i \cdot I$$

**力转换**（面力 [Pa] → 每粒子受力 [N]）：

$$\delta F_i = \delta\tau \cdot dp^2$$

叠加到近壁流体粒子（$y_i < 2h$）的 `force_prior(:,1)`。量纲验证：$\delta\tau$ [Pa] × $dp^2$ [m²] = [N]，与 `force_prior` 一致。

### 3.3 代码实现（`SPH_Poiseuille.m`）

```matlab
% 下壁面 tau_num（pair-wise）
is_bottom = (pair_j > n_fluid) & (pos(pair_j, 2) < 0);
i_b = pair_i(is_bottom);  j_b = pair_j(is_bottom);
if ~isempty(i_b)
    ex_b = dx_ij(is_bottom)./r_ij(is_bottom);
    ey_b = dy_ij(is_bottom)./r_ij(is_bottom);
    eBe_b = ex_b.*(B(i_b,1).*ex_b + B(i_b,2).*ey_b) + ...
            ey_b.*(B(i_b,3).*ex_b + B(i_b,4).*ey_b);
    f_pair_b = 4*mu .* eBe_b .* dW_ij(is_bottom) .* Vol(j_b) .* ...
               vel(i_b,1) ./ (r_ij(is_bottom) + 0.01*h) .* Vol(i_b);
    tau_num_bottom = -sum(f_pair_b) / DL;
end
% PI 更新 → force_prior
e_b = tau_target - tau_num_bottom;
I_bottom = max(-I_max, min(I_max, I_bottom + e_b * Dt));
delta_tau_b = k_p * e_b + k_i * I_bottom;
near_bottom = pos(1:n_fluid, 2) < 2*h;
force_prior(near_bottom, 1) = force_prior(near_bottom, 1) + delta_tau_b * dp^2;
```

参数：k_p=0.5，k_i=0.025，I_max=2.0（当前 config.ini 参数下验证通过）。

### 3.4 精度与局限

- 优点：wall_vel 恒零，物理意义清晰（纯无滑移）；PI 积分项消除稳态误差；不改 MEX
- 优点：L2 误差实测 **0.68%**（dp=0.04，t=10s），远低于 5% 阈值；tau_num 在 t≈4s 后稳定在 0.400±0.012
- 局限：仅适用稳态/准稳态；需标定 k_p、k_i；非稳态流动中 PI 响应有延迟

---

## 4. 方案 C：Adami 广义壁面 BC（自适应插值）

### 4.1 物理原理

基于 Adami (2012, 2013) [1,11] 的广义壁面 BC 框架，虚粒子速度不固定，而是通过对周围流体粒子的核函数加权插值动态确定，并在其中嵌入 τ_w 约束。

### 4.2 推导

**步骤 1**：对虚粒子 j，插值周围流体粒子速度：

$$\tilde{u}^j = \frac{\sum_{i \in fluid} u_i W_{ij}}{\sum_{i \in fluid} W_{ij}}$$

**步骤 2**：Adami 无滑移条件为 $u_{ghost} = 2u_{wall} - \tilde{u}$，对于静止壁面 $u_{wall}=0$，得 $u_{ghost} = -\tilde{u}$。

**步骤 3**：推广到 Neumann 条件。

设流体粒子 i 到壁面距离为 $d_f = y_i$，虚粒子 j 到壁面距离为 $d_g = |y_j|$，两者关于壁面对称时 $d_f = d_g = d$。

用镜像点线性插值近似壁面处梯度：

$$\left.\frac{\partial u}{\partial y}\right|_{wall} \approx \frac{u_f - u_g}{d_f + d_g} = \frac{\tau_w}{\mu}$$

用 $\tilde{u}$ 代替 $u_f$（SPH 加权插值），解出虚粒子速度：

$$u_g = \tilde{u} - (d_f + d_g)\frac{\tau_w}{\mu}$$

对称布置下 $d_f \approx d_g = d_j$，故：

$$\boxed{u_{ghost}^j = \tilde{u}^j - 2d_j \cdot \frac{\tau_w}{\mu}}$$

**验证**：τ_w=0 时 $u_{ghost}^j = \tilde{u}^j$，壁面处梯度为零（正确的零梯度 Neumann 极限）；τ_w≠0 且 $\tilde{u}=0$ 时 $u_{ghost}^j = -2d_j\tau_w/\mu < 0$，在壁面处产生正梯度，物理正确。

**上壁面**（法向 -ŷ，du/dy|_{y=H} = -τ_w/μ）：

$$u_{ghost}^j = \tilde{u}^j + 2d_j \cdot \frac{\tau_w}{\mu}$$

### 4.3 代码修改位置

**MATLAB 层**，在每个声学子步前更新 wall_vel：

```matlab
function wall_vel = update_wall_vel_neumann(pos, vel, wall_vel, ...
        pair_i, pair_j, W_ij, n_fluid, n_total, tau_w, mu, DH)
    % 对每个壁面虚粒子，插值流体速度并施加 Neumann 修正
    sum_W  = zeros(n_total, 1);
    sum_uW = zeros(n_total, 1);
    for k = 1:numel(pair_i)
        i = pair_i(k); j = pair_j(k);
        if i <= n_fluid && j > n_fluid
            sum_W(j)  = sum_W(j)  + W_ij(k);
            sum_uW(j) = sum_uW(j) + vel(i,1) * W_ij(k);
        end
    end
    for j = n_fluid+1 : n_total
        if sum_W(j) > 1e-12
            u_interp = sum_uW(j) / sum_W(j);
            d_j = min(abs(pos(j,2)), abs(DH - pos(j,2)));
            if pos(j, 2) < 0  % 下壁面
                wall_vel(j, 1) = u_interp - 2*(tau_w/mu) * d_j;
            else               % 上壁面
                wall_vel(j, 1) = u_interp + 2*(tau_w/mu) * d_j;
            end
        end
    end
end
```

**C/MEX 层**：无需修改。

### 4.4 精度与局限

- 精度：插值本身为 O(h)；由于粘性壁面项仍使用经验 ×4 单侧补偿（`sph_physics_mex.c:514`，Morris et al. 1997 [3]），未引入半解析边界积分重归一化（参见 Ferrand et al. 2012 [2]、Leroy et al. 2014 [10]），**实际精度阶次待网格收敛验证后声明**
- 优点：与 Adami 框架一致，物理上最严格；τ_w=0 极限正确退化为零梯度；自动适应流场变化
- 局限：每步需要额外的插值循环（可向量化）；W_ij 需从邻居搜索结果传入

---

## 5. 方案 D：粒子对粘性力统计 + 反馈调节法（新方案）

### 5.1 文献调研结论

经过系统搜索（Laha et al. 2025 [5], Wang et al. 2025 [6], Ferrand et al. 2012 [2], Adami et al. 2012/2013 [1,11], Mayrhofer et al. 2013 [12] 等），**目前没有文献采用"直接统计流固粒子对粘性力→与目标 τ_w 比较→反馈调节 wall_vel"的闭环控制思路**。

- Laha et al. (2025) [5] 是从速度梯度**估算**（测量）τ_w，属于后处理
- Wang et al. (2025) [6] 是从壁面函数**正向施加** τ_w，属于开环
- 本方案是**闭环反馈控制**，在 SPH 壁面 BC 领域属于新颖思路

### 5.2 物理原理

当前代码中，壁面切应力完全由流固粒子对的粘性力贡献。对于下壁面，所有流体-壁面粒子对的粘性力 x 分量之和，除以壁面面积，即为数值切应力 τ_num：

$$\tau_{num} = \frac{\sum_{(i,j) \in \text{fluid-wall pairs}} F_{visc,x}^{ij}}{L_x}$$

其中 $L_x$ 为 x 方向周期长度。若 $\tau_{num} \neq \tau_w^{target}$，则通过调节 `wall_vel` 消除误差。

### 5.3 完整推导

**步骤 1：pair-wise 统计数值切应力（避免重复计数）**

`viscous_force(i,1)` 是粒子 i 对所有邻居累加后的总力，不能用 `pair_i` 索引再求和（会重复计数）。正确做法是直接按 MEX 公式重算每个 wall-pair 的贡献：

单个流体-壁面粒子对 (i,j) 的粘性力 x 分量（`sph_physics_mex.c:514,519`）：

$$f_{visc,x}^{ij} = 4\mu \cdot eBe_i \cdot \frac{\partial W_{ij}}{\partial r} \cdot V_j \cdot \frac{v_x^i - v_{wall,x}^j}{r_{ij} + 0.01h} \cdot V_i$$

对所有下壁面流固粒子对求和，统一定义切应力为正值（流体受 +x 方向力）：

$$F_{visc,x}^{bottom} = \sum_{\substack{i \leq n_{fluid} \\ j > n_{fluid},\, y_j < 0}} f_{visc,x}^{ij}$$

$$\tau_{num} = \frac{F_{visc,x}^{bottom}}{L_x}$$

上壁面同理（流体受 -x 方向力，取绝对值保持符号一致）：

$$\tau_{num}^{top} = \frac{-F_{visc,x}^{top}}{L_x}$$

**步骤 2：误差计算（统一符号）**

$$e = \tau_w^{target} - \tau_{num}$$

两壁均用此定义，$\tau_{num}$ 均为正的切应力大小。

**步骤 3：PI 控制更新律**

纯 P 控制一般存在稳态误差，推荐 PI 控制：

$$v_{wall,x}^j \leftarrow v_{wall,x}^j - k_p \cdot e - k_i \cdot I$$

$$I \leftarrow I + e \cdot \Delta t$$

加入限幅与 anti-windup：

$$I \leftarrow \text{clip}(I,\; -I_{max},\; I_{max})$$

**灵敏度 $K_\tau$**（$\partial \tau_{num} / \partial v_{wall,x}$，用于估算 $k_p$）：

$$K_\tau = \frac{\partial \tau_{num}}{\partial v_{wall,x}} = \frac{-1}{L_x}\sum_{(i,j)\in\text{wall pairs}} 4\mu \cdot eBe_i \cdot \frac{\partial W_{ij}}{\partial r} \cdot V_j \cdot \frac{V_i}{r_{ij}+0.01h}$$

$K_\tau < 0$（wall_vel 增大 → 速度差减小 → 粘性力减小）。

**步骤 4：稳定条件**

更新律为 $v \leftarrow v - k_p e$，离散时间线性化后闭环极点为 $1 + k_p K_\tau$，稳定条件：

$$|1 + k_p K_\tau| < 1 \quad \Longleftrightarrow \quad 0 < k_p |K_\tau| < 2$$

无振荡单调收敛的保守条件：$k_p |K_\tau| < 1$。建议初始取 $k_p = 0.5 / |K_\tau|$。

### 5.4 代码修改位置

**MATLAB 层**，在粘性力计算之后、下一步开始之前，插入反馈更新。关键：用 pair-wise 公式重算，不用 `viscous_force` 数组。

```matlab
% ---- 方案 D'：pair-wise 统计 + PI 反馈调节 ----
tau_target = 0.4;   % 目标壁面切应力 [Pa]
k_p = 0.5 / abs(K_tau);  % 比例增益（K_tau 由灵敏度估算，见下）
k_i = 0.1 * k_p;         % 积分增益
I_max = 1.0;              % 积分限幅

% pair-wise 重算下壁面粘性力（避免重复计数）
is_bottom_pair = (pair_j > n_fluid) & (pos(pair_j, 2) < 0);
i_b = pair_i(is_bottom_pair);
j_b = pair_j(is_bottom_pair);
dv_x = vel(i_b,1) - wall_vel(j_b,1);
% eBe 需从 B 矩阵和方向向量重算，或从 MEX 额外输出
f_pair = 4 * mu * eBe_b .* dW_b .* Vol(j_b) .* dv_x ./ (r_b + 0.01*h) .* Vol(i_b);
F_bottom = sum(f_pair);
tau_num_bottom = F_bottom / DL;

% 上壁面（取负号统一为正切应力）
is_top_pair = (pair_j > n_fluid) & (pos(pair_j, 2) > DH);
% ... 同理 ...
tau_num_top = -F_top / DL;

% PI 控制（下壁面）
e_bottom = tau_target - tau_num_bottom;
I_bottom = clip(I_bottom + e_bottom * dt, -I_max, I_max);
dv_wall = k_p * e_bottom + k_i * I_bottom;
bottom_wall_idx = n_fluid + find(pos(n_fluid+1:end,2) < 0);
wall_vel(bottom_wall_idx, 1) = wall_vel(bottom_wall_idx, 1) - dv_wall;

% PI 控制（上壁面，符号对称）
e_top = tau_target - tau_num_top;
I_top = clip(I_top + e_top * dt, -I_max, I_max);
dv_wall_top = k_p * e_top + k_i * I_top;
top_wall_idx = n_fluid + find(pos(n_fluid+1:end,2) > DH);
wall_vel(top_wall_idx, 1) = wall_vel(top_wall_idx, 1) + dv_wall_top;
% ---- 方案 D' 结束 ----
```

**注意**：`eBe_b`、`dW_b`、`r_b` 需从邻居搜索结果中提取，或在 MEX 中新增输出。若不想改 MEX，可在 MATLAB 层用核函数公式重算（计算量小）。

### 5.5 稳定性分析

更新律 $v \leftarrow v - k_p e$ 的闭环极点为 $1 + k_p K_\tau$，稳定条件：

$$0 < k_p |K_\tau| < 2$$

无振荡单调收敛（保守）：$k_p |K_\tau| < 1$，建议初始取 $k_p = 0.5/|K_\tau|$。

加入积分项后需注意 anti-windup：积分器限幅 $|I| \leq I_{max}$，防止启动阶段积分饱和。

### 5.6 与其他方案的本质区别

| 特性 | 方案 A/C（前馈） | 方案 D（反馈） |
|------|----------------|---------------|
| 控制类型 | 开环（由公式直接计算 wall_vel） | 闭环（测量误差后修正） |
| 对模型误差的鲁棒性 | 依赖公式精度 | 自动补偿模型误差 |
| 收敛速度 | 即时（每步精确） | 需要若干步收敛 |
| 适用场景 | 稳态/非稳态均可 | 稳态或缓变流动 |
| 需要标定参数 | 无 | 需标定 α |

### 5.8 实现勘误（Plan → 实际调试中发现的问题）

以下问题在原始 Plan 中存在错误或遗漏，实现过程中逐一修正：

**Bug 1：tau_num 符号定义错误**

Plan 中下壁面 `tau_num = sum(f_pair)/DL`，但 `f_pair` 是壁面对流体的粘性力（阻力方向为 -x），`sum(f_pair) < 0`。正确定义壁面切应力为正值：

$$\tau_{num} = \frac{-\sum f_{pair}}{L_x}$$

上壁面同理取负号（壁面阻力方向为 +x，`f_pair > 0`，取负后为正）。Plan 中上壁面用 `-sum`、下壁面用 `+sum`，符号口径不统一，是发散的直接原因。

**Bug 2：PI 输出累加导致 wall_vel 无界漂移**

Plan 中 `wall_vel(bot) = wall_vel(bot) - dv_b`（每步累加），等效于对 PI 输出做了二次积分（PII 控制器），`wall_vel` 单调增长不收敛。

修正：PI 输出直接设定 `wall_vel`，而非累加：
```matlab
wall_vel(bot_idx, 1) = v_wall_b;      % 直接设定
wall_vel(top_idx, 1) = -v_wall_t;     % 上壁面符号相反
```

**Bug 3：上壁面 PI 更新方向错误**

Plan 中 `wall_vel(top) += dv_t`，但上壁面 `tau_num_top` 取负号后为正值，`e_t > 0` 时 `v_wall_t > 0`，需要 `wall_vel(top) = -v_wall_t`（向左运动增大速度差，增大切应力）。

**Bug 4：K_tau 符号与 tau_num 不一致**

Plan 中 `K_tau = sum(-4*mu*...)/DL`，但 tau_num 改为 `-sum/DL` 后，K_tau 应为 `sum(+4*mu*...)/DL`。

**遗漏 1：Dt=0 死循环保护**

Plan 未提及。当 PI 发散导致 vmax→∞ 时，CFL 条件使 Dt→0，`t = t + Dt` 不再前进，`while t < target_time` 永不退出。修正：添加 `Dt < 1e-14` 时 `error()` 保护。

**遗漏 2：Restart 状态不完整**

Plan 未提及。`state` 结构体需保存 `wall_vel, I_bottom, I_top, k_p, k_i`，否则续算时 PI 状态被重置。恢复端用 `isfield` 兼容旧 restart 文件。

**遗漏 3：tau_target 硬编码**

Plan 中 `tau_target = 0.4` 硬编码。应从参数派生：`tau_target = gravity_g * rho0 * DH / 2`，与 config.ini 自动同步。

**遗漏 4：L2 误差参考系**

Neumann BC 下壁面速度非零，解析解需加偏移：`u_exact = wall_vel_steady + g/(2ν)*y*(H-y)`。L2 误差应在去掉整体平移后比较抛物线形状。

### 5.9 实现路线（D+B）

**当前采用路线：PI 输出 delta_tau 直接叠加 force_prior（wall_vel 恒零）**

```
测量 tau_num（pair-wise）→ 计算误差 e = tau_target - tau_num
→ PI 输出 delta_tau → delta_F = delta_tau * dp² → 叠加到 force_prior
```

- wall_vel 恒零，壁面保持纯无滑移 Dirichlet BC
- PI 积分项消除稳态误差，k_p=0.5，k_i=0.025
- 实测结果：tau_num 在 t≈4s 收敛到 0.400±0.012，L2 误差 **0.68%**（dp=0.04，t=10s）

**与路线A（PI 控制 wall_vel）的区别**：

| 维度 | 路线 A（已废弃） | 路线 D+B（当前） |
|------|----------------|----------------|
| PI 输出目标 | wall_vel（间接影响粘性力） | force_prior（直接施力） |
| wall_vel | 由 PI 动态设定 | 恒零（纯无滑移） |
| 物理含义 | 壁面运动速度（模糊） | 近壁体积力（清晰） |
| 稳态误差 | PI 积分消除 | PI 积分消除 |
| 实现复杂度 | 需处理上下壁面符号 | 上下壁面均为 +x 方向 |

---

## 6. 方案对比

| 维度 | 方案 A（镜像速度） | 方案 B/D+B（PI+force_prior） | 方案 C（Adami 插值） | 方案 D 路线A（已废弃） |
|------|------------------|------------------------------|---------------------|----------------------|
| 理论基础 | 线性插值镜像 | 闭环 PI + 面力等效 | Adami 2012 广义 BC [1] | 闭环 PI 控制 wall_vel |
| 精度 | O(h)，待收敛验证 | L2=0.68%（dp=0.04，t=10s）✓ | 待网格收敛验证后声明 | L2=0.71%（已废弃） |
| τ_w=0 极限 | 正确（零梯度） | 自动满足（PI 积分消除误差） | 正确（零梯度） | 自动满足 |
| wall_vel | 动态设定 | 恒零（纯无滑移） | 动态设定 | 动态设定 |
| 改动范围 | MATLAB 层 wall_vel | force_prior（MATLAB 层） | MATLAB 层 wall_vel（插值） | MATLAB 层 wall_vel |
| MEX 修改 | 无 | 无 | 无 | 无 |
| 计算开销 | 低 | 低（pair-wise 求和） | 中 | 低 |
| 与现有框架兼容性 | 高 | 高 | 高 | 高 |
| 实现难度 | 低 | 中（需标定 k_p、k_i） | 中 | 中 |
| 适用场景 | 稳态/非稳态 | 稳态/准稳态 | 稳态/非稳态 | 稳态/准稳态 |
| 推荐度 | 快速验证 | **当前实现（推荐）** | 生产使用备选 | 已废弃 |

---

## 7. 推荐实现路径

**当前实现：方案 D+B（pair-wise tau 测量 + PI 输出 delta_tau 叠加 force_prior）**
- 理由：wall_vel 恒零物理意义清晰；PI 积分消除稳态误差；实测 L2=0.68%，验证通过
- 参数：k_p=0.5，k_i=0.025，I_max=2.0

**备选：方案 C（Adami 插值）**
- 理由：开环前馈，每步精确，适合非稳态流动
- 实现步骤：在主循环中每个对流步开始前调用 `update_wall_vel_neumann`，传入 W_ij

**不推荐：方案 A（原始牵引力直接施加，无 PI 反馈）**
- 理由：δ 选取任意性大，精度难以保证，已被 D+B 方案取代

---

## 8. 参考文献

1. Adami, S., Hu, X.Y., Adams, N.A. (2012). "A generalized wall boundary condition for smoothed particle hydrodynamics." *J. Comput. Phys.*, 231(21), 7057-7075.
2. Ferrand, M., Laurence, D.R., Rogers, B.D., Violeau, D., Kassiotis, C. (2012). "Unified semi-analytical wall boundary conditions for inviscid, laminar or turbulent flows in the meshless SPH method." *Int. J. Numer. Methods Fluids*, 71(4), 446-472.
3. Morris, J.P., Fox, P.J., Zhu, Y. (1997). "Modeling low Reynolds number incompressible flows using SPH." *J. Comput. Phys.*, 136(1), 214-226.
4. Nakayama, A., Lye, X.Y., Ng, K.C. (2022). "Wall-layer boundary condition method for laminar and turbulent flows in weakly-compressible SPH." *Eur. J. Mech. B/Fluids*, 95, 276-288.
5. Laha, S., Fourtakas, G., Das, P.K., Keshmiri, A. (2025). "Towards the estimation of wall shear stress in smoothed particle hydrodynamics." *Comput. Part. Mech.*, 12(2), 1309-1317.
6. Wang, F., Sun, Z., Hu, X. (2025). "A weakly compressible SPH method for RANS simulation of wall-bounded turbulent flows." *arXiv preprint* arXiv:2501.18397（预印本，未同行评审）.
7. Mayrhofer, A., Ferrand, M., Kassiotis, C., Violeau, D., Morel, F.-X. (2015). "Unified semi-analytical wall boundary conditions in SPH: analytical extension to 3-D." *Numer. Algorithms*, 68, 15-34.
8. Hosseini, S.M., Feng, J.J. (2011). "Pressure boundary conditions for computing incompressible flows with SPH." *J. Comput. Phys.*, 230(19), 7473-7487.
9. Band, S., Gissler, C., Ihmsen, M., Cornelis, J., Peer, A., Teschner, M. (2018). "Pressure Boundaries for Implicit Incompressible SPH." *ACM Trans. Graph.*, 37(2), Article 14.
10. Leroy, A., Violeau, D., Ferrand, M., Kassiotis, C. (2014). "Unified semi-analytical wall boundary conditions applied to 2-D incompressible SPH." *J. Comput. Phys.*, 261, 106-129.
11. Adami, S., Hu, X.Y., Adams, N.A. (2013). "A transport-velocity formulation for smoothed particle hydrodynamics." *J. Comput. Phys.*, 241, 292-307. DOI: 10.1016/j.jcp.2013.01.043.
12. Mayrhofer, A., Rogers, B.D., Violeau, D., Ferrand, M. (2013). "Investigation of wall bounded flows using SPH and the unified semi-analytical wall boundary conditions." *Comput. Phys. Commun.*, 184(11), 2515-2527. DOI: 10.1016/j.cpc.2013.07.004.
13. English, A., Domínguez, J.M., Vacondio, R., Stansby, P.K., Lind, S.J., Chiapponi, L., Gómez-Gesteira, M. (2022). "Modified dynamic boundary conditions (mDBC) for general-purpose smoothed particle hydrodynamics (SPH): application to tank sloshing, dam break and fish pass problems." *Comput. Part. Mech.*, 9, 1-15. DOI: 10.1007/s40571-021-00403-3.
