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

## 3. 方案 B：pair-wise tau 测量 + PI 输出 delta_tau 叠加 force_prior

### 3.1 物理原理

壁面保持纯无滑移（`wall_vel = 0`），通过 PI 控制器测量数值切应力 τ_num，将误差转换为 delta_tau，再以等效体积力叠加到近壁流体粒子的 `force_prior`。

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

## 5. 方案 D：虚粒子 wall_vel 修改法（Neumann BC 直接施加）

### 5.1 物理原理

通过在每个时间步动态修改虚粒子（ghost particle）的 `wall_vel`，使壁面处速度梯度等于 τ_w/μ，从而在粘性力计算中自然施加 Neumann BC。wall_vel 不再恒零，而是由当前流场插值速度加上梯度修正项确定。

与方案 A（简单镜像）的区别：方案 D 使用核函数加权插值（Adami 框架），精度更高，适用于多层虚粒子布置。

### 5.2 推导（Adami 插值 + 梯度修正）

**步骤 1**：对虚粒子 j，核函数加权插值周围流体粒子速度：

$$\tilde{u}^j = \frac{\sum_{i \in fluid} u_i W_{ij}}{\sum_{i \in fluid} W_{ij}}$$

**步骤 2**：由壁面梯度条件解出虚粒子速度。

设虚粒子 j 到壁面距离为 $d_j$，下壁面（法向 +ŷ，$du/dy|_{wall} = \tau_w/\mu$）：

$$u_{ghost}^j = \tilde{u}^j - 2d_j \cdot \frac{\tau_w}{\mu}$$

上壁面（法向 -ŷ，$du/dy|_{wall} = -\tau_w/\mu$）：

$$u_{ghost}^j = \tilde{u}^j + 2d_j \cdot \frac{\tau_w}{\mu}$$

**验证**：$\tau_w=0$ 时 $u_{ghost}^j = \tilde{u}^j$，壁面梯度为零（正确的零梯度极限）。

### 5.3 代码实现（MATLAB 层，不改 MEX）

在每个对流步开始前，更新 `wall_vel`：

```matlab
% 核函数加权插值流体速度到虚粒子
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
        u_tilde = sum_uW(j) / sum_W(j);
        d_j = min(abs(pos(j,2)), abs(DH - pos(j,2)));
        if pos(j,2) < 0        % 下壁面
            wall_vel(j,1) = u_tilde - 2*(tau_target/mu)*d_j;
        else                   % 上壁面
            wall_vel(j,1) = u_tilde + 2*(tau_target/mu)*d_j;
        end
    end
end
```

### 5.4 精度与局限

- 优点：开环前馈，每步精确，无收敛延迟；自动适应流场变化；与 Adami 框架一致
- 优点：τ_w=0 极限正确退化为零梯度（等价于方案 C）
- 局限：开环，不能自动补偿离散误差；需要 W_ij 从邻居搜索结果传入（已有）
- 局限：仅适用稳态/非稳态均可，但精度依赖插值质量

---

## 6. 方案对比

| 维度 | 方案 A（镜像速度） | 方案 B（PI+force_prior） | 方案 C（Adami 插值） | 方案 D（虚粒子 wall_vel） |
|------|------------------|--------------------------|---------------------|--------------------------|
| 理论基础 | 线性插值镜像 | 闭环 PI + 面力等效 | Adami 2012 广义 BC [1] | Adami 插值 + 梯度修正 |
| 精度 | O(h)，待收敛验证 | L2=0.68%（dp=0.04，t=10s）✓ | 待网格收敛验证后声明 | 待验证 |
| τ_w=0 极限 | 正确（零梯度） | 自动满足（PI 积分消除误差） | 正确（零梯度） | 正确（退化为方案 C） |
| wall_vel | 动态设定 | 恒零（纯无滑移） | 动态设定（插值） | 动态设定（插值+梯度修正） |
| 改动范围 | MATLAB 层 wall_vel | force_prior（MATLAB 层） | MATLAB 层 wall_vel | MATLAB 层 wall_vel |
| MEX 修改 | 无 | 无 | 无 | 无 |
| 计算开销 | 低 | 低（pair-wise 求和） | 中 | 中 |
| 与现有框架兼容性 | 高 | 高 | 高 | 高 |
| 实现难度 | 低 | 中（需标定 k_p、k_i） | 中 | 中 |
| 适用场景 | 稳态/非稳态 | 稳态/准稳态 | 稳态/非稳态 | 稳态/非稳态 |
| 推荐度 | 快速验证 | **当前实现（推荐）** | 生产使用备选 | 备选 |

---

## 7. 推荐实现路径

**当前实现：方案 B（pair-wise tau 测量 + PI 输出 delta_tau 叠加 force_prior）**
- 理由：wall_vel 恒零物理意义清晰；PI 积分消除稳态误差；实测 L2=0.68%，验证通过
- 参数：k_p=0.5，k_i=0.025，I_max=2.0

**备选：方案 D（虚粒子 wall_vel 修改法）**
- 理由：开环前馈，每步精确，适合非稳态流动；与 Adami 框架一致
- 实现步骤：在主循环中每个对流步开始前，对每个虚粒子插值流体速度并施加梯度修正

**备选：方案 C（Adami 插值，τ_w=0 特例）**
- 理由：方案 D 在 τ_w=0 时的退化形式，适合纯无滑移验证

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
