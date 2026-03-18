%% SPH_Poiseuille.m
% 基于 SPH（光滑粒子流体动力学）方法的二维平板泊肃叶流动模拟
% 采用弱可压缩 SPH + 双层时间步进（对流步 + 声学步）
% 计算加速：C/MEX + OpenMP（邻居搜索与物理计算分离）

clear; clc; close all;

fprintf('SPH Poiseuille Flow Simulation\n');

% 路径初始化
project_dir = fileparts(mfilename('fullpath'));
if isempty(project_dir)
    project_dir = pwd;
end

build_dir = fullfile(project_dir, 'build');     % MEX 编译输出目录
mex_dir = fullfile(project_dir, 'mex');         % MEX C 源码目录
config_path = fullfile(project_dir, 'config.ini');
restart_path = fullfile(project_dir, 'restart.mat');
result_png = fullfile(project_dir, 'SPH_Poiseuille_result.png');
profile_evolution_png = fullfile(project_dir, 'SPH_centerline_profile_evolution.png');

if ~exist(build_dir, 'dir')
    mkdir(build_dir);
end
addpath(build_dir);

%% S1: MEX 自动编译
neighbor_mex_name = 'sph_neighbor_search_mex';
physics_mex_name = 'sph_physics_mex';
ensure_mex_compiled(fullfile(mex_dir, 'sph_neighbor_search_mex.c'), neighbor_mex_name, build_dir);
ensure_mex_compiled(fullfile(mex_dir, 'sph_physics_mex.c'), physics_mex_name, build_dir);

%% S2: INI 读取 + 参数计算
cfg = parse_ini(config_path);

% 物理参数
DL = get_ini_numeric(cfg, 'physical', 'DL');       % 流体域长度（X方向，周期性）
DH = get_ini_numeric(cfg, 'physical', 'DH');       % 流体域高度（Y方向，壁面）
dp = get_ini_numeric(cfg, 'physical', 'dp');       % 粒子间距
rho0 = get_ini_numeric(cfg, 'physical', 'rho0');   % 参考密度
mu = get_ini_numeric(cfg, 'physical', 'mu');       % 动力粘度
U_bulk = get_ini_numeric(cfg, 'physical', 'U_bulk');  % 截面平均速度
c_f = get_ini_numeric(cfg, 'physical', 'c_f');     % 人工声速因子

% 仿真控制参数
t_end = get_ini_numeric(cfg, 'simulation', 'end_time');
output_interval = get_ini_numeric(cfg, 'simulation', 'output_interval');
sort_interval = round(get_ini_numeric(cfg, 'simulation', 'sort_interval'));
restart_from_file = round(get_ini_numeric(cfg, 'simulation', 'restart_from_file'));
bc_mode = get_ini_string(cfg, 'simulation', 'bc_mode');  % 'B' 或 'D'
if ~ismember(bc_mode, {'B', 'D'})
    error('bc_mode 必须为 B 或 D，当前值: %s', bc_mode);
end
fprintf('BC 模式: 方案%s\n', bc_mode);

% 几何参数自动对齐：确保 DL/dp 和 DH/dp 为整数（周期边界 + 均匀粒子排列）
DL_raw = DL;  DH_raw = DH;
DL = round(DL / dp) * dp;
DH = round(DH / dp) * dp;
if abs(DL - DL_raw) > 1e-12 || abs(DH - DH_raw) > 1e-12
    fprintf('几何对齐: DL %.4f→%.4f, DH %.4f→%.4f (dp=%.4f)\n', DL_raw, DL, DH_raw, DH, dp);
end

% 派生参数
gravity_g = 12.0 * mu * U_bulk / (rho0 * DH^2);  % 等效体积力（驱动泊肃叶流）
U_max = 1.5 * U_bulk;                             % 中心线最大速度（2D平板泊肃叶流解析关系）
h = 1.3 * dp;                                   % 光滑长度
BW = 4.0 * dp; %#ok<NASGU>                      % 壁面层厚度
periodic_buffer = BW;                            % 周期性缓冲层宽度
p0 = rho0 * c_f^2;                              % 参考压力（弱可压缩状态方程）
inv_sigma0 = dp^2;                               % 核函数归一化因子的倒数
nu = mu / rho0;                                  % 运动粘度

fprintf('参数: DL=%.3f, DH=%.3f, dp=%.4f, h=%.4f\n', DL, DH, dp, h);
fprintf('参数: rho0=%.3f, mu=%.3f, U_bulk=%.6f, c_f=%.3f\n', rho0, mu, U_bulk, c_f);
fprintf('派生: g=%.6f, Umax=%.6f, p0=%.6f\n', gravity_g, U_max, p0);
fprintf('仿真: end_time=%.3f, output_interval=%.3f, sort_interval=%d\n', t_end, output_interval, sort_interval);
fprintf('周期缓冲层: buffer=%.4f (%.2f*h)\n', periodic_buffer, periodic_buffer / h);

if sort_interval <= 0
    error('sort_interval 必须为正整数。');
end

%% S3: 粒子初始化
% 流体粒子：均匀网格分布在 [0,DL] x [0,DH] 域内
x_fluid = (dp/2 : dp : DL - dp/2)';
y_fluid = (dp/2 : dp : DH - dp/2)';
[X_fluid, Y_fluid] = meshgrid(x_fluid, y_fluid);
pos_fluid = [X_fluid(:), Y_fluid(:)];
n_fluid = size(pos_fluid, 1);

% 壁面虚粒子：上下各 4 层，用于施加无滑移边界条件
x_wall = (dp/2 : dp : DL - dp/2)';
x_bottom = [];
y_bottom = [];
x_top = [];
y_top = [];
for layer = 1:4
    yb = -((2 * layer - 1) * dp / 2);
    yt = DH + ((2 * layer - 1) * dp / 2);
    x_bottom = [x_bottom; x_wall]; %#ok<AGROW>
    y_bottom = [y_bottom; yb * ones(size(x_wall))]; %#ok<AGROW>
    x_top = [x_top; x_wall]; %#ok<AGROW>
    y_top = [y_top; yt * ones(size(x_wall))]; %#ok<AGROW>
end
pos_wall = [x_bottom, y_bottom; x_top, y_top];
n_wall = size(pos_wall, 1);
n_total = n_fluid + n_wall;

pos = [pos_fluid; pos_wall];
vel = zeros(n_total, 2);
wall_vel = zeros(n_total, 2);
rho = rho0 * ones(n_total, 1);
p = zeros(n_total, 1);
drho_dt = zeros(n_total, 1);
force = zeros(n_total, 2);
force_prior = zeros(n_total, 2);

mass_fluid = rho0 * dp^2;
mass_wall = rho0 * dp^2;
mass = [mass_fluid * ones(n_fluid, 1); mass_wall * ones(n_wall, 1)];
Vol = mass ./ rho;
B = zeros(n_total, 4);
B(:, 1) = 1.0;
B(:, 4) = 1.0;

fprintf('粒子数: fluid=%d, wall=%d, total=%d\n', n_fluid, n_wall, n_total);

%% S4: Restart 检查
config_signature = create_config_signature(DL, DH, dp, rho0, mu, U_bulk, c_f, t_end, output_interval, sort_interval);
t = 0.0;
step = 0;

if restart_from_file && exist(restart_path, 'file')
    restart_data = load(restart_path, 'state', 'config_signature');
    can_resume = isfield(restart_data, 'state') && isfield(restart_data, 'config_signature') ...
        && strcmp(restart_data.config_signature, config_signature);
    if can_resume
        state = restart_data.state;
        valid_state = isfield(state, 'pos') && isequal(size(state.pos), [n_total, 2]) ...
            && isfield(state, 'vel') && isequal(size(state.vel), [n_total, 2]) ...
            && isfield(state, 'rho') && isequal(size(state.rho), [n_total, 1]) ...
            && isfield(state, 'p') && isequal(size(state.p), [n_total, 1]) ...
            && isfield(state, 'drho_dt') && isequal(size(state.drho_dt), [n_total, 1]) ...
            && isfield(state, 'force') && isequal(size(state.force), [n_total, 2]) ...
            && isfield(state, 'force_prior') && isequal(size(state.force_prior), [n_total, 2]) ...
            && isfield(state, 't') && isfield(state, 'step');
        if valid_state
            pos = state.pos;
            vel = state.vel;
            rho = state.rho;
            p = state.p;
            drho_dt = state.drho_dt;
            force = state.force;
            force_prior = state.force_prior;
            t = state.t;
            step = state.step;
            % 恢复 PI 控制器状态（若旧 restart 文件无此字段则保持默认值）
            if isfield(state, 'I_bottom'),  I_bottom = state.I_bottom;  end
            if isfield(state, 'I_top'),     I_top    = state.I_top;     end
            if isfield(state, 'k_p'),       k_p      = state.k_p;       end
            if isfield(state, 'k_i'),       k_i      = state.k_i;       end
            fprintf('Restart: 从 t=%.6f, step=%d 继续。\n', t, step);
        else
            fprintf('Restart 文件存在但状态结构不兼容，重新开始。\n');
        end
    else
        fprintf('Restart 文件签名不匹配，重新开始。\n');
    end
end

%% S5: 初始邻居搜索 + 密度求和 + 核梯度修正
% 建立粒子对列表，计算核函数值和梯度，用于后续物理量插值
[pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij] = feval(neighbor_mex_name, pos, n_fluid, n_total, h, DL);
% 密度重初始化 + 核梯度修正矩阵 B（提高边界附近精度）
[rho, Vol, B] = feval(physics_mex_name, 'density_correction', ...
    pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij, ...
    mass, n_fluid, n_total, rho0, h, inv_sigma0);
p(1:n_fluid) = p0 * (rho(1:n_fluid) ./ rho0 - 1.0);
p(n_fluid+1:end) = 0.0;

% 壁面切应力目标值（两种方案共用）
tau_target = gravity_g * rho0 * DH / 2;  % 泊肃叶流壁面切应力解析值 [Pa]
tau_num_bottom = 0.0;   % 监控变量初始化
tau_num_top    = 0.0;

if strcmp(bc_mode, 'B')
    % 方案 B：PI 控制器参数
    I_bottom = 0.0;     % 积分项（下壁面）
    I_top    = 0.0;     % 积分项（上壁面）
    I_max    = 2.0;     % 积分限幅
    k_p = 0.5;          % PI 比例增益
    k_i = 0.025;        % PI 积分增益
else
    % 方案 D：无需 PI 参数，wall_vel 每步由 Adami 插值动态设定
    I_bottom = 0.0; I_top = 0.0; k_p = 0.0; k_i = 0.0; I_max = 0.0;  % 占位，restart 兼容
end

% 槽道中间截面（x=DL/2）速度剖面演化记录：每个输出时刻保留一条 u(y)
n_bins = max(20, round(DH / dp));
mid_x = 0.5 * DL;
mid_half_width = max(dp, h);
profile_times = zeros(0, 1);
mid_profile_u = NaN(n_bins, 0);
[~, u_mid_init] = compute_mid_channel_profile( ...
    pos(1:n_fluid, :), vel(1:n_fluid, 1), DL, DH, mid_x, mid_half_width, n_bins);
profile_times(end + 1, 1) = t;
mid_profile_u(:, end + 1) = u_mid_init;

%% S6: 主循环（双层时间步进：外层对流步 + 内层声学步）
% 外层：计算粘性力、传输修正、对流时间步长 Dt
% 内层：在 Dt 内以声学时间步 dt 推进压力-速度耦合
while t < t_end - 1e-12
    target_time = min(t + output_interval, t_end);

    while t < target_time - 1e-12
        step = step + 1;

        % 对流时间步长（基于粒子速度、加速度和粘性约束）
        Dt = advection_time_step(vel(1:n_fluid, :), force(1:n_fluid, :), force_prior(1:n_fluid, :), ...
            mass(1:n_fluid), h, mu, rho0, U_bulk);
        Dt = min(Dt, target_time - t);
        Dt = min(Dt, t_end - t);
        if Dt < 1e-14
            error('时间步长退化为零（Dt=%.2e），仿真在 t=%.6f step=%d 发散，请检查 PI 控制器参数。', Dt, t, step);
        end

        % 密度重初始化 + 核梯度修正
        [rho, Vol, B] = feval(physics_mex_name, 'density_correction', ...
            pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij, ...
            mass, n_fluid, n_total, rho0, h, inv_sigma0);

        % 粘性力计算（含壁面无滑移镜像速度）
        if strcmp(bc_mode, 'D')
            viscous_force = feval(physics_mex_name, 'viscous_force', ...
                pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
                vel, Vol, B, mu, h, n_fluid, n_total, mass, wall_vel, ...
                pos, tau_target/mu, DH);
        else
            viscous_force = feval(physics_mex_name, 'viscous_force', ...
                pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
                vel, Vol, B, mu, h, n_fluid, n_total, mass, wall_vel);
        end
        force_prior = viscous_force;
        force_prior(1:n_fluid, 1) = force_prior(1:n_fluid, 1) + mass(1:n_fluid) * gravity_g;  % 叠加体积力

        if strcmp(bc_mode, 'B')
        % === 方案 B：pair-wise tau 测量 + PI 输出 delta_tau 叠加 force_prior ===
        is_bottom = (pair_j > n_fluid) & (pos(pair_j, 2) < 0);
        i_b = pair_i(is_bottom);
        j_b = pair_j(is_bottom);
        if ~isempty(i_b)
            dx_b = dx_ij(is_bottom);  dy_b = dy_ij(is_bottom);
            r_b  = r_ij(is_bottom);   dW_b  = dW_ij(is_bottom);
            ex_b = dx_b ./ r_b;  ey_b = dy_b ./ r_b;
            eBe_b = ex_b.*(B(i_b,1).*ex_b + B(i_b,2).*ey_b) + ...
                    ey_b.*(B(i_b,3).*ex_b + B(i_b,4).*ey_b);
            dv_x_b = vel(i_b,1) - wall_vel(j_b,1);
            f_pair_b = 4*mu .* eBe_b .* dW_b .* Vol(j_b) .* dv_x_b ...
                       ./ (r_b + 0.01*h) .* Vol(i_b);
            tau_num_bottom = -sum(f_pair_b) / DL;
        else
            tau_num_bottom = 0.0;
        end
        is_top = (pair_j > n_fluid) & (pos(pair_j, 2) > DH);
        i_t = pair_i(is_top);
        j_t = pair_j(is_top);
        if ~isempty(i_t)
            dx_t = dx_ij(is_top);  dy_t = dy_ij(is_top);
            r_t  = r_ij(is_top);   dW_t  = dW_ij(is_top);
            ex_t = dx_t ./ r_t;  ey_t = dy_t ./ r_t;
            eBe_t = ex_t.*(B(i_t,1).*ex_t + B(i_t,2).*ey_t) + ...
                    ey_t.*(B(i_t,3).*ex_t + B(i_t,4).*ey_t);
            dv_x_t = vel(i_t,1) - wall_vel(j_t,1);
            f_pair_t = 4*mu .* eBe_t .* dW_t .* Vol(j_t) .* dv_x_t ...
                       ./ (r_t + 0.01*h) .* Vol(i_t);
            tau_num_top = -sum(f_pair_t) / DL;
        else
            tau_num_top = 0.0;
        end
        e_b = tau_target - tau_num_bottom;
        I_bottom = max(-I_max, min(I_max, I_bottom + e_b * Dt));
        delta_tau_b = k_p * e_b + k_i * I_bottom;
        near_bottom = pos(1:n_fluid, 2) < 2*h;
        force_prior(near_bottom, 1) = force_prior(near_bottom, 1) + delta_tau_b * dp^2;
        e_t = tau_target - tau_num_top;
        I_top = max(-I_max, min(I_max, I_top + e_t * Dt));
        delta_tau_t = k_p * e_t + k_i * I_top;
        near_top = pos(1:n_fluid, 2) > DH - 2*h;
        force_prior(near_top, 1) = force_prior(near_top, 1) + delta_tau_t * dp^2;
        % === 方案 B 结束 ===

        else
        % === 方案 D：wall_vel 置零，MEX 内部用显式镜像+梯度修正 ===
        % u_ghost = -v_i ± G*d_j，u_wall=0 精确锚定，梯度通过虚粒子施加
        wall_vel(n_fluid+1:end, :) = 0;
        % 监控：pair-wise 测量下壁面和上壁面 tau_num（仅用于输出）
        is_bottom = (pair_j > n_fluid) & (pos(pair_j, 2) < 0);
        i_b = pair_i(is_bottom); j_b = pair_j(is_bottom);
        if ~isempty(i_b)
            r_b = r_ij(is_bottom); dW_b = dW_ij(is_bottom);
            dx_b = dx_ij(is_bottom); dy_b = dy_ij(is_bottom);
            ex_b = dx_b./r_b; ey_b = dy_b./r_b;
            eBe_b = ex_b.*(B(i_b,1).*ex_b+B(i_b,2).*ey_b) + ...
                    ey_b.*(B(i_b,3).*ex_b+B(i_b,4).*ey_b);
            f_pair_b = 4*mu.*eBe_b.*dW_b.*Vol(j_b).*(vel(i_b,1)-wall_vel(j_b,1)) ...
                       ./(r_b+0.01*h).*Vol(i_b);
            tau_num_bottom = -sum(f_pair_b) / DL;
        else
            tau_num_bottom = 0.0;
        end
        is_top = (pair_j > n_fluid) & (pos(pair_j, 2) > DH);
        i_t = pair_i(is_top); j_t = pair_j(is_top);
        if ~isempty(i_t)
            r_t = r_ij(is_top); dW_t = dW_ij(is_top);
            dx_t = dx_ij(is_top); dy_t = dy_ij(is_top);
            ex_t = dx_t./r_t; ey_t = dy_t./r_t;
            eBe_t = ex_t.*(B(i_t,1).*ex_t+B(i_t,2).*ey_t) + ...
                    ey_t.*(B(i_t,3).*ex_t+B(i_t,4).*ey_t);
            f_pair_t = 4*mu.*eBe_t.*dW_t.*Vol(j_t).*(vel(i_t,1)-wall_vel(j_t,1)) ...
                       ./(r_t+0.01*h).*Vol(i_t);
            tau_num_top = sum(f_pair_t) / DL;
        else
            tau_num_top = 0.0;
        end
        e_b = tau_target - tau_num_bottom;  % 仅用于日志
        % === 方案 D 结束 ===
        end

        % 传输速度修正（抑制张力不稳定性）
        pos = feval(physics_mex_name, 'transport_correction', ...
            pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
            Vol, B, pos, h, n_fluid, n_total);

        pos = bounding_from_wall(pos, n_fluid, DH, dp);

        % 内层声学子步循环
        relaxation_time = 0.0;
        while relaxation_time < Dt - 1e-12
            % 声学时间步长（CFL 条件：基于声速 + 粒子速度）
            dt = acoustic_time_step(vel(1:n_fluid, :), c_f, h, Dt - relaxation_time);

            % 第一阶段积分：密度演化 + 压力更新 + 位置半步推进 + 压力梯度力
            [rho, p, pos, force, drho_dt] = feval(physics_mex_name, 'integration_1st', ...
                pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
                Vol, B, rho, mass, pos, vel, drho_dt, force_prior, dt, ...
                n_fluid, n_total, rho0, p0, c_f, wall_vel);

            % 速度更新：合力（粘性力 + 压力梯度力）/ 质量 * dt
            acc_fluid = (force_prior(1:n_fluid, :) + force(1:n_fluid, :)) ./ mass(1:n_fluid);
            vel(1:n_fluid, :) = vel(1:n_fluid, :) + acc_fluid * dt;

            % 第二阶段积分：位置修正 + 密度散度修正
            [pos, drho_dt, ~] = feval(physics_mex_name, 'integration_2nd', ...
                pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
                Vol, rho, pos, vel, dt, n_fluid, n_total, wall_vel);
            rho(1:n_fluid) = rho(1:n_fluid) + drho_dt(1:n_fluid) * (dt / 2.0);  % 密度半步修正

            % X 方向周期性边界处理
            [pos, wrapped] = periodic_bounding(pos, n_fluid, DL, periodic_buffer);
            if wrapped
                [pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij] = feval(neighbor_mex_name, pos, n_fluid, n_total, h, DL);
            end

            % BoundingFromWall: 壁面防穿透
            pos = bounding_from_wall(pos, n_fluid, DH, dp);

            relaxation_time = relaxation_time + dt;
        end

        % 对流步结束：周期性边界 + 壁面防穿透
        [pos, ~] = periodic_bounding(pos, n_fluid, DL, periodic_buffer);
        pos = bounding_from_wall(pos, n_fluid, DH, dp);
        % 方案B：wall_vel 恒零，重置虚粒子速度；方案D：wall_vel 已动态设定，保持不变
        if strcmp(bc_mode, 'B')
            vel(n_fluid+1:end, :) = zeros(n_wall, 2);
        else
            vel(n_fluid+1:end, :) = wall_vel(n_fluid+1:end, :);
        end

        % 定期按空间网格排序粒子（提升缓存命中率）
        if mod(step, sort_interval) == 0 && step ~= 1
            [pos, vel, rho, mass, wall_vel, drho_dt, force_prior, force, p, Vol, B] = sort_particles_by_cell( ...
                pos, vel, rho, mass, wall_vel, drho_dt, force_prior, force, p, Vol, B, n_fluid, DL, h);
        end
        [pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij] = feval(neighbor_mex_name, pos, n_fluid, n_total, h, DL);

        t = t + Dt;

        if mod(step, 20) == 0
            vmax = max(vecnorm(vel(1:n_fluid, :), 2, 2));
            fprintf('step=%d, t=%.6f/%.6f, Dt=%.4e, pairs=%d, vmax=%.4f\n', ...
                step, t, t_end, Dt, numel(pair_i), vmax);
            if strcmp(bc_mode, 'B')
                fprintf('  [B] tau_bot=%.4f, tau_top=%.4f, tau_target=%.4f, e_b=%.4f\n', ...
                    tau_num_bottom, tau_num_top, tau_target, e_b);
            else
                fprintf('  [D] tau_bot=%.4f, tau_top=%.4f, tau_target=%.4f, e_b=%.4f\n', ...
                    tau_num_bottom, tau_num_top, tau_target, e_b);
            end
        end
    end

    fprintf('输出点: t=%.6f, step=%d\n', t, step);
    state = struct( ...
        'pos', pos, ...
        'vel', vel, ...
        'rho', rho, ...
        'p', p, ...
        'drho_dt', drho_dt, ...
        'force', force, ...
        'force_prior', force_prior, ...
        't', t, ...
        'step', step, ...
        'I_bottom', I_bottom, ...
        'I_top', I_top, ...
        'k_p', k_p, ...
        'k_i', k_i);
    save_restart(restart_path, config_signature, state);

    [~, u_mid_now] = compute_mid_channel_profile( ...
        pos(1:n_fluid, :), vel(1:n_fluid, 1), DL, DH, mid_x, mid_half_width, n_bins);
    profile_times(end + 1, 1) = t;
    mid_profile_u(:, end + 1) = u_mid_now;
end

%% S7: 后处理与验证
% 提取流体粒子速度剖面，与泊肃叶流解析解对比
fluid_pos = pos(1:n_fluid, :);
fluid_pos(:, 1) = mod(fluid_pos(:, 1), DL);
fluid_vel = vel(1:n_fluid, :);

[y_mid, u_mean] = compute_binned_profile_mean(fluid_pos(:, 2), fluid_vel(:, 1), 0.0, DH, n_bins);
% 无滑移 BC 下解析解：标准抛物线
u_exact = gravity_g / (2.0 * nu) .* y_mid .* (DH - y_mid);

valid = ~isnan(u_mean);
if ~any(valid)
    error('后处理失败：速度剖面分箱为空。');
end

L2_error = sqrt(sum((u_mean(valid) - u_exact(valid)).^2) / max(sum(u_exact(valid).^2), eps));
fprintf('L2 相对误差 = %.4f%%\n', 100.0 * L2_error);

if L2_error < 0.05
    fprintf('验证通过: L2_error < 5%%\n');
else
    fprintf('验证未达标: L2_error >= 5%%\n');
end

fig = figure('Color', 'w', 'Position', [100, 100, 1400, 520], 'Renderer', 'painters');

% --- 左图：速度剖面对比 ---
ax1 = subplot(1, 2, 1);
y_norm = y_mid / DH;
u_norm_exact = u_exact / U_max;
u_norm_sph   = u_mean  / U_max;

plot(ax1, u_norm_exact, y_norm, '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.8); hold(ax1, 'on');
plot(ax1, u_norm_sph, y_norm, 'o', 'Color', [0.85 0.2 0.2], ...
    'MarkerSize', 5, 'LineWidth', 1.2, 'MarkerFaceColor', [0.85 0.2 0.2]);
hold(ax1, 'off');

set(ax1, 'FontName', 'Times New Roman', 'FontSize', 13, 'LineWidth', 1.0, ...
    'TickDir', 'in', 'TickLength', [0.015 0.015], 'Box', 'on');
xlabel(ax1, '$u_x / U_{max}$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel(ax1, '$y / H$', 'Interpreter', 'latex', 'FontSize', 14);
legend(ax1, {'Analytical', 'SPH'}, 'Location', 'southeast', ...
    'FontSize', 11, 'FontName', 'Times New Roman', 'Box', 'off');
xlim(ax1, [-0.05, 1.15]);
ylim(ax1, [0, 1]);
text(ax1, 0.05, 0.95, sprintf('$L_2 = %.2f\\%%$', 100*L2_error), ...
    'Units', 'normalized', 'Interpreter', 'latex', 'FontSize', 12, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
    'FontName', 'Times New Roman');
title(ax1, '(a) Velocity profile', 'FontName', 'Times New Roman', ...
    'FontSize', 13, 'FontWeight', 'normal');

% --- 右图：光滑速度场 ---
ax2 = subplot(1, 2, 2);

% 构建规则网格，分辨率与粒子间距匹配
nx_grid = round(DL / dp) * 2;
ny_grid = round(DH / dp) * 2;
xg = linspace(0, DL, nx_grid);
yg = linspace(0, DH, ny_grid);
[Xg, Yg] = meshgrid(xg, yg);

% 将周期性流体粒子位置展开，补 ghost 粒子处理周期边界
fp_x = mod(fluid_pos(:,1), DL);
fp_y = fluid_pos(:,2);
fv_x = fluid_vel(:,1);
% 左侧 ghost：把右侧粒子（x > DL - 2h）复制到 x - DL
right_mask = fp_x > DL - 2*h;
% 右侧 ghost：把左侧粒子（x < 2h）复制到 x + DL
left_mask  = fp_x < 2*h;
fp_x_ext = [fp_x; fp_x(right_mask) - DL; fp_x(left_mask) + DL];
fp_y_ext = [fp_y; fp_y(right_mask);       fp_y(left_mask)];
fv_x_ext = [fv_x; fv_x(right_mask);      fv_x(left_mask)];
F_interp = scatteredInterpolant(fp_x_ext, fp_y_ext, fv_x_ext, 'natural', 'nearest');
Ug = F_interp(Xg, Yg);

% 壁面区域厚度
wall_thick = 4 * dp;
y_lo = -wall_thick;
y_hi = DH + wall_thick;

% 绘制下壁面灰色区域
fill(ax2, [0 DL DL 0], [y_lo y_lo 0 0], [0.75 0.75 0.75], ...
    'EdgeColor', 'none'); hold(ax2, 'on');
% 绘制上壁面灰色区域
fill(ax2, [0 DL DL 0], [DH DH y_hi y_hi], [0.75 0.75 0.75], ...
    'EdgeColor', 'none');
% 壁面边界线
plot(ax2, [0 DL], [0 0], 'k-', 'LineWidth', 1.2);
plot(ax2, [0 DL], [DH DH], 'k-', 'LineWidth', 1.2);

% 流体速度场
imagesc(ax2, xg, yg, Ug);
set(ax2, 'YDir', 'normal');

% 重绘壁面（imagesc 会覆盖 fill）
fill(ax2, [0 DL DL 0], [y_lo y_lo 0 0], [0.75 0.75 0.75], 'EdgeColor', 'none');
fill(ax2, [0 DL DL 0], [DH DH y_hi y_hi], [0.75 0.75 0.75], 'EdgeColor', 'none');
plot(ax2, [0 DL], [0 0], 'k-', 'LineWidth', 1.2);
plot(ax2, [0 DL], [DH DH], 'k-', 'LineWidth', 1.2);

% 壁面标注
text(ax2, DL/2, y_lo/2, 'Wall', 'HorizontalAlignment', 'center', ...
    'FontName', 'Times New Roman', 'FontSize', 11, 'Color', [0.3 0.3 0.3]);
text(ax2, DL/2, DH + wall_thick/2, 'Wall', 'HorizontalAlignment', 'center', ...
    'FontName', 'Times New Roman', 'FontSize', 11, 'Color', [0.3 0.3 0.3]);
hold(ax2, 'off');

axis(ax2, 'equal');
xlim(ax2, [0, DL]);
ylim(ax2, [y_lo, y_hi]);
set(ax2, 'FontName', 'Times New Roman', 'FontSize', 13, 'LineWidth', 1.0, ...
    'TickDir', 'in', 'TickLength', [0.015 0.015], 'Box', 'on');
xlabel(ax2, '$x$ (m)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel(ax2, '$y$ (m)', 'Interpreter', 'latex', 'FontSize', 14);
colormap(ax2, turbo);
caxis(ax2, [0, U_max * 1.1]);
cb = colorbar(ax2);
cb.Label.String = '$u_x$ (m/s)';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 13;
set(cb, 'FontName', 'Times New Roman', 'FontSize', 11, ...
    'TickDirection', 'in', 'LineWidth', 1.0);
title(ax2, '(b) Velocity field', 'FontName', 'Times New Roman', ...
    'FontSize', 13, 'FontWeight', 'normal');

saveas(fig, result_png);
fprintf('结果图已保存: %s\n', result_png);

% --- 中间截面速度剖面演化图（每个时间一条剖面） ---
fig_evo = figure('Color', 'w', 'Position', [140, 140, 760, 560], 'Renderer', 'painters');
ax_evo = axes(fig_evo);
hold(ax_evo, 'on');

tvals = profile_times(:)';
n_profiles = numel(tvals);
line_cmap = parula(max(n_profiles, 2));

for k = 1:n_profiles
    u_k = mid_profile_u(:, k) / U_max;
    valid_k = ~isnan(u_k);
    if any(valid_k)
        plot(ax_evo, u_k(valid_k), y_mid(valid_k) / DH, '-', ...
            'Color', line_cmap(k, :), 'LineWidth', 1.0);
    end
end

plot(ax_evo, u_norm_exact, y_norm, '--', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.6);
hold(ax_evo, 'off');

set(ax_evo, 'FontName', 'Times New Roman', 'FontSize', 13, 'LineWidth', 1.0, ...
    'TickDir', 'in', 'TickLength', [0.015 0.015], 'Box', 'on');
xlabel(ax_evo, '$u_x / U_{max}$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel(ax_evo, '$y / H$', 'Interpreter', 'latex', 'FontSize', 14);
xlim(ax_evo, [-0.05, 1.15]);
ylim(ax_evo, [0, 1]);
title(ax_evo, '(c) Mid-channel profile evolution', 'FontName', 'Times New Roman', ...
    'FontSize', 13, 'FontWeight', 'normal');

colormap(ax_evo, line_cmap);
t_min = min(tvals);
t_max = max(tvals);
if t_max <= t_min
    t_max = t_min + 1.0;
end
caxis(ax_evo, [t_min, t_max]);
cb_evo = colorbar(ax_evo);
cb_evo.Label.String = '$t$ (s)';
cb_evo.Label.Interpreter = 'latex';
cb_evo.Label.FontSize = 12;
set(cb_evo, 'FontName', 'Times New Roman', 'FontSize', 11, ...
    'TickDirection', 'in', 'LineWidth', 1.0);

saveas(fig_evo, profile_evolution_png);
fprintf('中间截面剖面演化图已保存: %s\n', profile_evolution_png);

%% Local Functions
function ensure_mex_compiled(src_path, out_name, build_dir)
    if ~exist(src_path, 'file')
        error('MEX 源码不存在: %s', src_path);
    end

    out_bin = fullfile(build_dir, [out_name, '.', mexext]);
    need_compile = ~exist(out_bin, 'file');
    if ~need_compile
        src_info = dir(src_path);
        bin_info = dir(out_bin);
        need_compile = src_info.datenum > bin_info.datenum;
    end

    if ~need_compile
        return;
    end

    fprintf('编译 MEX: %s\n', out_name);
    try
        if ispc
            mex('-R2018a', '-O', 'COMPFLAGS="$COMPFLAGS /openmp"', ...
                '-output', out_name, '-outdir', build_dir, src_path);
        else
            mex('-R2018a', '-O', ...
                'CFLAGS="$CFLAGS -fopenmp"', ...
                'LDFLAGS="$LDFLAGS -fopenmp"', ...
                '-output', out_name, '-outdir', build_dir, src_path);
        end
    catch ME
        error('编译 %s 失败: %s', out_name, ME.message);
    end

    if ~exist(out_bin, 'file')
        error('编译后未生成: %s', out_bin);
    end
end

function cfg = parse_ini(filename)
    if ~exist(filename, 'file')
        error('配置文件不存在: %s', filename);
    end

    text = fileread(filename);
    lines = regexp(text, '\r\n|\n|\r', 'split');
    cfg = struct();
    section = '';

    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if isempty(line)
            continue;
        end
        if startsWith(line, ';') || startsWith(line, '#')
            continue;
        end
        if startsWith(line, '[') && endsWith(line, ']')
            section = strtrim(line(2:end-1));
            if ~isvarname(section)
                error('INI 段名非法: %s', section);
            end
            if ~isfield(cfg, section)
                cfg.(section) = struct();
            end
            continue;
        end

        eq_pos = strfind(line, '=');
        if isempty(eq_pos)
            continue;
        end
        key = strtrim(line(1:eq_pos(1)-1));
        val_raw = strtrim(line(eq_pos(1)+1:end));

        val_raw = regexprep(val_raw, '[;#].*$', '');
        val_raw = strtrim(val_raw);
        if isempty(section)
            error('INI 键值不在任何段内: %s', line);
        end
        if ~isvarname(key)
            error('INI 键名非法: %s', key);
        end

        num_val = str2double(val_raw);
        if ~isnan(num_val)
            cfg.(section).(key) = num_val;
        else
            cfg.(section).(key) = val_raw;
        end
    end
end

function value = get_ini_string(cfg, section, key)
    if ~isfield(cfg, section)
        error('缺少段: [%s]', section);
    end
    if ~isfield(cfg.(section), key)
        error('缺少键: [%s].%s', section, key);
    end
    value = cfg.(section).(key);
    if ~ischar(value)
        value = num2str(value);
    end
    value = strtrim(value);
end

function value = get_ini_numeric(cfg, section, key)
    if ~isfield(cfg, section)
        error('缺少段: [%s]', section);
    end
    if ~isfield(cfg.(section), key)
        error('缺少键: [%s].%s', section, key);
    end
    value = cfg.(section).(key);
    if ~(isnumeric(value) && isscalar(value) && isfinite(value))
        error('键 [%s].%s 不是有效数值。', section, key);
    end
end

function sig = create_config_signature(DL, DH, dp, rho0, mu, U_bulk, c_f, t_end, output_interval, sort_interval)
    sig = sprintf('DL=%.12g|DH=%.12g|dp=%.12g|rho0=%.12g|mu=%.12g|Ub=%.12g|cf=%.12g|t=%.12g|oi=%.12g|si=%d', ...
        DL, DH, dp, rho0, mu, U_bulk, c_f, t_end, output_interval, sort_interval);
end

function Dt = advection_time_step(vel, force, force_prior, mass, h, mu, rho0, U_bulk)
% 对流时间步长：综合考虑速度、加速度和粘性扩散约束
% Dt = 0.25 * h / max(v_max, sqrt(4h*a_max), viscous_speed)
    viscous_speed = mu / (rho0 * h);
    speed_ref = max(viscous_speed, U_bulk);

    v_mag = vecnorm(vel, 2, 2);
    v_max = max(v_mag);

    acc_vec = (force + force_prior) ./ mass;
    acc_mag = vecnorm(acc_vec, 2, 2);
    acc_max = max(4.0 * h * acc_mag);

    speed_max = max(sqrt(max(v_max^2, acc_max)), speed_ref);
    speed_max = max(speed_max, 1e-12);
    Dt = 0.25 * h / speed_max;
end

function dt = acoustic_time_step(vel, c_f, h, remain)
% 声学时间步长：CFL 条件 dt = 0.6 * h / (c + |v|_max)，不超过剩余时间
    signal = c_f + vecnorm(vel, 2, 2);
    max_signal = max(signal);
    max_signal = max(max_signal, 1e-12);
    dt = min(0.6 * h / max_signal, remain);
    dt = max(dt, 1e-12);
end

function [pos, vel, rho, mass, wall_vel, drho_dt, force_prior, force, p, Vol, B] = sort_particles_by_cell( ...
    pos, vel, rho, mass, wall_vel, drho_dt, force_prior, force, p, Vol, B, n_fluid, DL, h)
% 按空间网格对流体粒子排序，壁面粒子保持不变
% 目的：提升邻居搜索的缓存局部性

    n_total = size(pos, 1);
    if n_fluid <= 0 || n_fluid >= n_total
        return;
    end

    idx_f = sort_subset_indices(pos(1:n_fluid, :), DL, h);
    idx = [idx_f; (n_fluid+1:n_total)'];

    pos = pos(idx, :);
    vel = vel(idx, :);
    rho = rho(idx, :);
    mass = mass(idx, :);
    wall_vel = wall_vel(idx, :);
    drho_dt = drho_dt(idx, :);
    force_prior = force_prior(idx, :);
    force = force(idx, :);
    p = p(idx, :);
    Vol = Vol(idx, :);
    B = B(idx, :);
end

function idx = sort_subset_indices(pos_sub, DL, h)
% 按 2D 网格单元排序粒子索引（先 Y 后 X），用于空间局部性优化
    n = size(pos_sub, 1);
    if n == 0
        idx = zeros(0, 1);
        return;
    end

    cell_size = 2.0 * h;
    cx = floor(mod(pos_sub(:, 1), DL) / cell_size);
    y0 = min(pos_sub(:, 2));
    cy = floor((pos_sub(:, 2) - y0) / cell_size);
    [~, idx] = sortrows([cy, cx], [1, 2]);
end

function [pos, wrapped] = periodic_bounding(pos, n_fluid, DL, buffer_width)
% X 方向周期性边界：超出 [−buffer, DL+buffer] 的粒子平移回域内
    x = pos(1:n_fluid, 1);
    wrapped = false;
    left_mask = (x < -buffer_width);
    while any(left_mask)
        x(left_mask) = x(left_mask) + DL;
        wrapped = true;
        left_mask = (x < -buffer_width);
    end
    right_mask = (x >= DL + buffer_width);
    while any(right_mask)
        x(right_mask) = x(right_mask) - DL;
        wrapped = true;
        right_mask = (x >= DL + buffer_width);
    end
    pos(1:n_fluid, 1) = x;
end

function [y_mid, u_mean] = compute_binned_profile_mean(y_values, u_values, y_min, y_max, n_bins)
% 将粒子按 Y 坐标分箱，计算每个箱内的平均 X 速度（用于速度剖面）
    edges = linspace(y_min, y_max, n_bins + 1);
    y_mid = 0.5 * (edges(1:end-1) + edges(2:end));
    y_mid = y_mid(:);
    bin_id = discretize(y_values, edges);

    sum_u = accumarray(bin_id(~isnan(bin_id)), u_values(~isnan(bin_id)), [n_bins, 1], @sum, 0);
    cnt_u = accumarray(bin_id(~isnan(bin_id)), 1, [n_bins, 1], @sum, 0);
    u_mean = sum_u ./ max(cnt_u, 1);
    u_mean(cnt_u == 0) = NaN;
end

function [y_mid, u_mean] = compute_mid_channel_profile(pos, u_x, DL, DH, mid_x, half_width, n_bins)
% 提取 x=mid_x 附近竖向切片上的 u(y) 速度剖面
    x_wrap = mod(pos(:, 1), DL);
    dx_mid = abs(x_wrap - mid_x);
    dx_mid = min(dx_mid, DL - dx_mid);
    is_mid = (dx_mid <= half_width);

    if ~any(is_mid)
        [y_mid, u_mean] = compute_binned_profile_mean([], [], 0.0, DH, n_bins);
        return;
    end

    [y_mid, u_mean] = compute_binned_profile_mean(pos(is_mid, 2), u_x(is_mid), 0.0, DH, n_bins);
end

function save_restart(restart_path, config_signature, state)
% 保存重启文件（包含完整粒子状态和配置签名，用于断点续算）
    save(restart_path, 'state', 'config_signature', '-v7.3');
end

function pos = bounding_from_wall(pos, n_fluid, DH, dp)
% 壁面防穿透：粒子到壁面距离 < 0.25*dp 时，弹回到 0.5*dp 处
    dist_min = 0.25 * dp;
    half_dp = 0.5 * dp;
    y = pos(1:n_fluid, 2);

    % 下壁面 (y=0, 法向 +y)
    too_close_bottom = (y < dist_min);
    if any(too_close_bottom)
        pos(too_close_bottom, 2) = half_dp;
    end

    % 上壁面 (y=DH, 法向 -y)
    dist_top = DH - y;
    too_close_top = (dist_top < dist_min);
    if any(too_close_top)
        pos(too_close_top, 2) = DH - half_dp;
    end
end
