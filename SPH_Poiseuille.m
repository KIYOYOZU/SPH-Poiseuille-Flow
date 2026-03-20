%% SPH_Poiseuille.m
% 基于 SPH（光滑粒子流体动力学）方法的二维平板泊肃叶流动模拟
% 采用弱可压缩 SPH + dual-criteria 时间步进
% 计算加速：C/MEX + OpenMP（邻居搜索与物理计算分离）
% 当前默认链路：厚壁粒子区 + 周期 ghost 邻居搜索 + dual-criteria 声学子步

clear; clc; close all;

fprintf('SPH Poiseuille Flow Simulation\n');

% 路径初始化
project_dir = fileparts(mfilename('fullpath'));
if isempty(project_dir)
    project_dir = pwd;
end

build_dir = fullfile(project_dir, 'build');     % MEX 编译输出目录
mex_dir = fullfile(project_dir, 'mex');         % MEX C 源码目录
config_path = get_env_override('SPH_CONFIG_OVERRIDE', fullfile(project_dir, 'config.ini'));
restart_path = get_env_override('SPH_RESTART_PATH_OVERRIDE', fullfile(project_dir, 'restart.mat'));
results_dir = fullfile(project_dir, 'results');
result_png = get_env_override('SPH_RESULT_PNG_OVERRIDE', fullfile(results_dir, 'SPH_Poiseuille_result.png'));
profile_evolution_png = get_env_override('SPH_PROFILE_PNG_OVERRIDE', ...
    fullfile(results_dir, 'SPH_centerline_profile_evolution.png'));

if ~exist(build_dir, 'dir')
    mkdir(build_dir);
end
ensure_parent_dir(restart_path);
ensure_parent_dir(result_png);
ensure_parent_dir(profile_evolution_png);
addpath(build_dir);

%% S1: MEX 自动编译
neighbor_mex_name = 'sph_neighbor_search_mex';
physics_mex_name = 'sph_physics_shell_mex';
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
fprintf('边界模式: 厚壁粒子 + 算子内 no-slip / no-penetration\n');

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
wall_thickness = 4.0 * dp;                       % 厚壁宽度，对齐 SPHinXsys 的 BW = 4dp
periodic_buffer = 0.0;                           % 周期边界立即包裹，不保留额外缓冲带
transport_coeff = 0.1;                           % dual-criteria 下保守的 shifting 系数
p0 = rho0 * c_f^2;                              % 参考压力（弱可压缩状态方程）
inv_sigma0 = dp^2;                               % 核函数归一化因子的倒数
nu = mu / rho0;                                  % 运动粘度

fprintf('参数: DL=%.3f, DH=%.3f, dp=%.4f, h=%.4f\n', DL, DH, dp, h);
fprintf('参数: rho0=%.3f, mu=%.3f, U_bulk=%.6f, c_f=%.3f\n', rho0, mu, U_bulk, c_f);
fprintf('派生: g=%.6f, Umax=%.6f, p0=%.6f, transport_coeff=%.3f\n', gravity_g, U_max, p0, transport_coeff);
fprintf('仿真: end_time=%.3f, output_interval=%.3f, sort_interval=%d\n', t_end, output_interval, sort_interval);
fprintf('周期边界: immediate wrap + minimum-image neighbor search\n');
fprintf('壁面: 厚壁粒子区, BW=%.4f (%.0f*dp)\n', wall_thickness, wall_thickness / dp);

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

% 厚壁粒子区：上下各 BW 厚度的规则壁面粒子层
% 关键修复背景（3eb6de0）：旧单层 shell 壁面支撑不足，
% 长时算例里会与硬 wall clip 一起放大近壁非物理扰动。
[pos_wall, wall_normal, wall_measure, wall_thickness_arr] = build_shell_wall_particles(DL, DH, dp, wall_thickness);
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
wall_particle_volume = wall_measure .* wall_thickness_arr;
mass_wall = rho0 * wall_particle_volume;
mass = [mass_fluid * ones(n_fluid, 1); mass_wall];
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

cfg = struct( ...
    'DL', DL, ...
    'DH', DH, ...
    'dp', dp, ...
    'rho0', rho0, ...
    'mu', mu, ...
    'U_bulk', U_bulk, ...
    'c_f', c_f, ...
    't_end', t_end, ...
    'output_interval', output_interval, ...
    'sort_interval', sort_interval, ...
    'restart_from_file', restart_from_file, ...
    'gravity_g', gravity_g, ...
    'U_max', U_max, ...
    'h', h, ...
    'wall_thickness', wall_thickness, ...
    'periodic_buffer', periodic_buffer, ...
    'p0', p0, ...
    'inv_sigma0', inv_sigma0, ...
    'nu', nu, ...
    'transport_coeff', transport_coeff, ...
    'config_signature', config_signature);

geom = struct( ...
    'n_fluid', n_fluid, ...
    'n_wall', n_wall, ...
    'n_total', n_total, ...
    'mass', mass, ...
    'wall_vel', wall_vel, ...
    'wall_normal', wall_normal, ...
    'wall_measure', wall_measure, ...
    'wall_thickness_arr', wall_thickness_arr);

state = struct( ...
    'pos', pos, ...
    'vel', vel, ...
    'rho', rho, ...
    'p', p, ...
    'drho_dt', drho_dt, ...
    'force', force, ...
    'force_prior', force_prior, ...
    'Vol', Vol, ...
    'B', B, ...
    't', t, ...
    'step', step);

neighbor = struct( ...
    'pair_i', pair_i, ...
    'pair_j', pair_j, ...
    'dx_ij', dx_ij, ...
    'dy_ij', dy_ij, ...
    'r_ij', r_ij, ...
    'W_ij', W_ij, ...
    'dW_ij', dW_ij);

monitor = struct( ...
    'tau_target', gravity_g * rho0 * DH / 2, ...
    'tau_num_bottom', 0.0, ...
    'tau_num_top', 0.0, ...
    'n_bins', max(20, round(DH / dp)), ...
    'mid_x', 0.5 * DL, ...
    'mid_half_width', max(dp, h), ...
    'profile_times', zeros(0, 1), ...
    'mid_profile_u', NaN(max(20, round(DH / dp)), 0));

[~, u_mid_init] = compute_mid_channel_profile( ...
    state.pos(1:geom.n_fluid, :), state.vel(1:geom.n_fluid, 1), ...
    cfg.DL, cfg.DH, monitor.mid_x, monitor.mid_half_width, monitor.n_bins);
monitor.profile_times(end + 1, 1) = state.t;
monitor.mid_profile_u(:, end + 1) = u_mid_init;

%% S6: 主循环（dual-criteria 时间步进）
% 每个外层对流/黏性步 Dt 只做一次密度求和、核修正、粘性力和传输修正，
% 然后在 Dt 内部执行多次声学子步推进压力/密度松弛。
% 关键修复背景（3eb6de0）：旧版“单层 CFL + 每步较强 transport correction”
% 会持续放大近壁横向脉动，所以当前把 shifting 限定在外层步。
while state.t < cfg.t_end - 1e-12
    target_time = min(state.t + cfg.output_interval, cfg.t_end);

    while state.t < target_time - 1e-12
        state.step = state.step + 1;

        remain_outer = min(target_time - state.t, cfg.t_end - state.t);
        Dt_outer = advection_viscous_time_step( ...
            state.vel(1:geom.n_fluid, :), state.force(1:geom.n_fluid, :), ...
            state.force_prior(1:geom.n_fluid, :), geom.mass(1:geom.n_fluid), ...
            cfg.U_bulk, cfg.nu, cfg.h, remain_outer);
        if Dt_outer < 1e-14
            error('外层时间步退化为零（Dt=%.2e），仿真在 t=%.6f step=%d 发散。', ...
                Dt_outer, state.t, state.step);
        end

        [state.rho, state.Vol, state.B] = density_correction_mex( ...
            neighbor, geom.mass, geom.n_fluid, geom.n_total, cfg, physics_mex_name);
        state.force_prior = compute_force_prior_mex(state, geom, neighbor, cfg, physics_mex_name);
        state.pos = transport_correction_mex(state.pos, state.Vol, state.B, neighbor, geom, cfg, physics_mex_name);

        relaxation_time = 0.0;
        dt_inner = 0.0;
        while relaxation_time < Dt_outer - 1e-12
            dt_inner = cfl_time_step(state.vel(1:geom.n_fluid, :), cfg.c_f, cfg.h, Dt_outer - relaxation_time);
            dt_inner = min(dt_inner, target_time - state.t);
            dt_inner = min(dt_inner, cfg.t_end - state.t);
            if dt_inner < 1e-14
                error('声学时间步退化为零（dt=%.2e），仿真在 t=%.6f step=%d 发散。', ...
                    dt_inner, state.t, state.step);
            end

            [state.rho, state.p, state.pos, state.force, state.drho_dt] = integration_1st_mex( ...
                state, geom, neighbor, cfg, dt_inner, physics_mex_name);
            state.vel = update_velocity(state.vel, geom.mass, state.force_prior, state.force, geom.n_fluid, geom.n_total, dt_inner);
            [state.pos, state.drho_dt] = integration_2nd_mex( ...
                state, geom, neighbor, dt_inner, physics_mex_name);
            state.rho(1:geom.n_fluid) = state.rho(1:geom.n_fluid) + state.drho_dt(1:geom.n_fluid) * (0.5 * dt_inner);
            state.p(1:geom.n_fluid) = cfg.p0 * (state.rho(1:geom.n_fluid) ./ cfg.rho0 - 1.0);
            state.p(geom.n_fluid+1:end) = 0.0;

            relaxation_time = relaxation_time + dt_inner;
            state.t = state.t + dt_inner;
        end

        [state.pos, ~] = periodic_bounding(state.pos, geom.n_fluid, cfg.DL, cfg.periodic_buffer);
        state.vel(geom.n_fluid+1:end, :) = 0.0;

        if mod(state.step, cfg.sort_interval) == 0 && state.step ~= 1
            [state.pos, state.vel, state.rho, geom.mass, geom.wall_vel, state.drho_dt, ...
                state.force_prior, state.force, state.p, state.Vol, state.B] = sort_particles_by_cell( ...
                state.pos, state.vel, state.rho, geom.mass, geom.wall_vel, state.drho_dt, ...
                state.force_prior, state.force, state.p, state.Vol, state.B, ...
                geom.n_fluid, cfg.DL, cfg.h);
        end

        neighbor = build_neighbor_cache(state.pos, geom.n_fluid, geom.n_total, cfg.h, cfg.DL, neighbor_mex_name);
        [monitor.tau_num_bottom, monitor.tau_num_top] = wall_shear_monitor_mex( ...
            neighbor, state.pos, state.vel, geom.wall_vel, state.Vol, state.B, ...
            geom.n_fluid, cfg.DL, cfg.DH, cfg.mu, cfg.h, physics_mex_name);

        if mod(state.step, 20) == 0
            vmax = max(vecnorm(state.vel(1:geom.n_fluid, :), 2, 2));
            fprintf('step=%d, t=%.6f/%.6f, Dt=%.4e, dt=%.4e, pairs=%d, vmax=%.4f\n', ...
                state.step, state.t, cfg.t_end, Dt_outer, dt_inner, numel(neighbor.pair_i), vmax);
            fprintf('  [thick-wall-noslip] tau_bot=%.4f, tau_top=%.4f, tau_target=%.4f\n', ...
                monitor.tau_num_bottom, monitor.tau_num_top, monitor.tau_target);
        end
    end

    fprintf('输出点: t=%.6f, step=%d\n', state.t, state.step);
    save_restart(restart_path, cfg.config_signature, make_restart_state(state));

    [~, u_mid_now] = compute_mid_channel_profile( ...
        state.pos(1:geom.n_fluid, :), state.vel(1:geom.n_fluid, 1), ...
        cfg.DL, cfg.DH, monitor.mid_x, monitor.mid_half_width, monitor.n_bins);
    monitor.profile_times(end + 1, 1) = state.t;
    monitor.mid_profile_u(:, end + 1) = u_mid_now;
end

%% S7: 后处理与验证
% 提取流体粒子速度剖面，与泊肃叶流解析解对比
fluid_pos = state.pos(1:geom.n_fluid, :);
fluid_pos(:, 1) = mod(fluid_pos(:, 1), cfg.DL);
fluid_vel = state.vel(1:geom.n_fluid, :);

[y_mid, u_mean] = compute_binned_profile_mean(fluid_pos(:, 2), fluid_vel(:, 1), 0.0, cfg.DH, monitor.n_bins);
% 无滑移 BC 下解析解：标准抛物线
u_exact = cfg.gravity_g / (2.0 * cfg.nu) .* y_mid .* (cfg.DH - y_mid);

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
y_norm = y_mid / cfg.DH;
u_norm_exact = u_exact / cfg.U_max;
u_norm_sph   = u_mean  / cfg.U_max;

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
nx_grid = round(cfg.DL / cfg.dp) * 2;
ny_grid = round(cfg.DH / cfg.dp) * 2;
xg = linspace(0, cfg.DL, nx_grid);
yg = linspace(0, cfg.DH, ny_grid);
[Xg, Yg] = meshgrid(xg, yg);

% 将周期性流体粒子位置展开，补 ghost 粒子处理周期边界
fp_x = mod(fluid_pos(:,1), cfg.DL);
fp_y = fluid_pos(:,2);
fv_x = fluid_vel(:,1);
% 左侧 ghost：把右侧粒子（x > DL - 2h）复制到 x - DL
right_mask = fp_x > cfg.DL - 2*cfg.h;
% 右侧 ghost：把左侧粒子（x < 2h）复制到 x + DL
left_mask  = fp_x < 2*cfg.h;
fp_x_ext = [fp_x; fp_x(right_mask) - cfg.DL; fp_x(left_mask) + cfg.DL];
fp_y_ext = [fp_y; fp_y(right_mask);       fp_y(left_mask)];
fv_x_ext = [fv_x; fv_x(right_mask);      fv_x(left_mask)];
F_interp = scatteredInterpolant(fp_x_ext, fp_y_ext, fv_x_ext, 'natural', 'nearest');
Ug = F_interp(Xg, Yg);

% 厚壁粒子区显示范围
wall_thick = cfg.wall_thickness;
y_lo = -wall_thick;
y_hi = cfg.DH + wall_thick;

% 绘制下壁面灰色区域
fill(ax2, [0 cfg.DL cfg.DL 0], [y_lo y_lo 0 0], [0.75 0.75 0.75], ...
    'EdgeColor', 'none'); hold(ax2, 'on');
% 绘制上壁面灰色区域
fill(ax2, [0 cfg.DL cfg.DL 0], [cfg.DH cfg.DH y_hi y_hi], [0.75 0.75 0.75], ...
    'EdgeColor', 'none');
% 壁面边界线
plot(ax2, [0 cfg.DL], [0 0], 'k-', 'LineWidth', 1.2);
plot(ax2, [0 cfg.DL], [cfg.DH cfg.DH], 'k-', 'LineWidth', 1.2);

% 流体速度场
imagesc(ax2, xg, yg, Ug);
set(ax2, 'YDir', 'normal');

% 重绘壁面（imagesc 会覆盖 fill）
fill(ax2, [0 cfg.DL cfg.DL 0], [y_lo y_lo 0 0], [0.75 0.75 0.75], 'EdgeColor', 'none');
fill(ax2, [0 cfg.DL cfg.DL 0], [cfg.DH cfg.DH y_hi y_hi], [0.75 0.75 0.75], 'EdgeColor', 'none');
plot(ax2, [0 cfg.DL], [0 0], 'k-', 'LineWidth', 1.2);
plot(ax2, [0 cfg.DL], [cfg.DH cfg.DH], 'k-', 'LineWidth', 1.2);

% 壁面标注
text(ax2, cfg.DL/2, y_lo/2, 'Wall', 'HorizontalAlignment', 'center', ...
    'FontName', 'Times New Roman', 'FontSize', 11, 'Color', [0.3 0.3 0.3]);
text(ax2, cfg.DL/2, cfg.DH + wall_thick/2, 'Wall', 'HorizontalAlignment', 'center', ...
    'FontName', 'Times New Roman', 'FontSize', 11, 'Color', [0.3 0.3 0.3]);
hold(ax2, 'off');

axis(ax2, 'equal');
xlim(ax2, [0, cfg.DL]);
ylim(ax2, [y_lo, y_hi]);
set(ax2, 'FontName', 'Times New Roman', 'FontSize', 13, 'LineWidth', 1.0, ...
    'TickDir', 'in', 'TickLength', [0.015 0.015], 'Box', 'on');
xlabel(ax2, '$x$ (m)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel(ax2, '$y$ (m)', 'Interpreter', 'latex', 'FontSize', 14);
colormap(ax2, turbo);
caxis(ax2, [0, cfg.U_max * 1.1]);
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

tvals = monitor.profile_times(:)';
n_profiles = numel(tvals);
line_cmap = parula(max(n_profiles, 2));

for k = 1:n_profiles
    u_k = monitor.mid_profile_u(:, k) / cfg.U_max;
    valid_k = ~isnan(u_k);
    if any(valid_k)
        plot(ax_evo, u_k(valid_k), y_mid(valid_k) / cfg.DH, '-', ...
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

function value = get_env_override(env_name, default_value)
    value = getenv(env_name);
    if isempty(value)
        value = default_value;
    end
end

function ensure_parent_dir(file_path)
    parent_dir = fileparts(file_path);
    if ~isempty(parent_dir) && ~exist(parent_dir, 'dir')
        mkdir(parent_dir);
    end
end

function neighbor = build_neighbor_cache(pos, n_fluid, n_total, h, DL, neighbor_mex_name)
% 统一封装周期邻居搜索 MEX，返回供各物理算子复用的 pair cache。
    [pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij] = feval( ...
        neighbor_mex_name, pos, n_fluid, n_total, h, DL);
    neighbor = struct( ...
        'pair_i', pair_i, ...
        'pair_j', pair_j, ...
        'dx_ij', dx_ij, ...
        'dy_ij', dy_ij, ...
        'r_ij', r_ij, ...
        'W_ij', W_ij, ...
        'dW_ij', dW_ij);
end

function [rho, Vol, B] = density_correction_mex(neighbor, mass, n_fluid, n_total, cfg, physics_mex_name)
% 调用密度重初始化与核梯度修正 mode，保持主循环只负责状态编排。
    [rho, Vol, B] = feval( ...
        physics_mex_name, 'density_correction', ...
        neighbor.pair_i, neighbor.pair_j, neighbor.dx_ij, neighbor.dy_ij, neighbor.r_ij, neighbor.W_ij, neighbor.dW_ij, ...
        mass, n_fluid, n_total, cfg.rho0, cfg.h, cfg.inv_sigma0);
end

function force_prior = compute_force_prior_mex(state, geom, neighbor, cfg, physics_mex_name)
% 计算黏性先验力并叠加体积力，供后续压力步做速度更新。
    force_prior = feval( ...
        physics_mex_name, 'viscous_force', ...
        neighbor.pair_i, neighbor.pair_j, neighbor.dx_ij, neighbor.dy_ij, neighbor.r_ij, neighbor.dW_ij, ...
        state.vel, state.Vol, state.B, cfg.mu, cfg.h, geom.n_fluid, geom.n_total, geom.mass, geom.wall_vel);
    force_prior(1:geom.n_fluid, 1) = force_prior(1:geom.n_fluid, 1) + geom.mass(1:geom.n_fluid) * cfg.gravity_g;
    force_prior(geom.n_fluid+1:end, :) = 0.0;
end

function pos = transport_correction_mex(pos, Vol, B, neighbor, geom, cfg, physics_mex_name)
% 外层步执行一次粒子 shifting，抑制聚集但不直接修改速度。
    pos = feval( ...
        physics_mex_name, 'transport_correction', ...
        neighbor.pair_i, neighbor.pair_j, neighbor.dx_ij, neighbor.dy_ij, neighbor.r_ij, neighbor.dW_ij, ...
        Vol, B, pos, cfg.h, geom.n_fluid, geom.n_total, cfg.transport_coeff);
end

function [rho, p, pos, force, drho_dt] = integration_1st_mex(state, geom, neighbor, cfg, dt_step, physics_mex_name)
% 第一阶段声学子步：半步推进 rho/pos，并计算压力梯度力。
    [rho, p, pos, force, drho_dt] = feval( ...
        physics_mex_name, 'integration_1st', ...
        neighbor.pair_i, neighbor.pair_j, neighbor.dx_ij, neighbor.dy_ij, neighbor.r_ij, neighbor.dW_ij, ...
        state.Vol, state.B, state.rho, geom.mass, state.pos, state.vel, state.drho_dt, state.force_prior, ...
        dt_step, geom.n_fluid, geom.n_total, cfg.rho0, cfg.p0, cfg.c_f, geom.wall_vel);
end

function vel = update_velocity(vel, mass, force_prior, force, n_fluid, n_total, dt_step)
% 用黏性先验力与压力力更新流体速度，壁粒子速度始终保持零。
    acc_fluid = (force_prior(1:n_fluid, :) + force(1:n_fluid, :)) ./ mass(1:n_fluid);
    vel(1:n_fluid, :) = vel(1:n_fluid, :) + acc_fluid * dt_step;
    vel(n_fluid+1:n_total, :) = 0.0;
end

function [pos, drho_dt] = integration_2nd_mex(state, geom, neighbor, dt_step, physics_mex_name)
% 第二阶段声学子步：完成位置推进，并返回密度散度修正项。
    [pos, drho_dt, ~] = feval( ...
        physics_mex_name, 'integration_2nd', ...
        neighbor.pair_i, neighbor.pair_j, neighbor.dx_ij, neighbor.dy_ij, neighbor.r_ij, neighbor.dW_ij, ...
        state.Vol, state.rho, state.pos, state.vel, dt_step, geom.n_fluid, geom.n_total, geom.wall_vel);
end

function Dt = advection_viscous_time_step(vel, force, force_prior, mass, U_ref, nu, h, remain)
% Dual-criteria 外层时间步：对流/黏性尺度控制一次配置更新的间隔
    accel_norm = vecnorm(force + force_prior, 2, 2);
    accel_scale = 4.0 * h * accel_norm ./ max(mass, eps);
    speed_sq = vecnorm(vel, 2, 2) .^ 2;
    speed_max = sqrt(max([speed_sq; accel_scale; 0.0]));
    speed_ref = max(U_ref, nu / max(h, eps));
    Dt = min(0.25 * h / max(max(speed_max, speed_ref), 1e-12), remain);
    Dt = max(Dt, 1e-12);
end

function sink = void_buffer(~)
    sink = [];
end

function state = advance_one_step_mex(state, geom, neighbor, cfg, dt_step, physics_mex_name)
% 兼容旧单步推进门面；当前 dual-criteria 主循环默认不走这条路径。
    [state.rho, state.p, state.pos, state.vel, state.drho_dt, state.force, state.force_prior, state.Vol, state.B] = feval( ...
        physics_mex_name, 'advance_shell_step', ...
        neighbor.pair_i, neighbor.pair_j, neighbor.dx_ij, neighbor.dy_ij, neighbor.r_ij, neighbor.W_ij, neighbor.dW_ij, ...
        geom.mass, state.pos, state.vel, geom.wall_vel, state.rho, state.drho_dt, ...
        dt_step, geom.n_fluid, geom.n_total, cfg.rho0, cfg.p0, cfg.c_f, cfg.mu, cfg.h, cfg.inv_sigma0, cfg.gravity_g);
end

function [tau_bottom, tau_top] = wall_shear_monitor_mex(neighbor, pos, vel, wall_vel, Vol, B, n_fluid, DL, DH, mu, h, physics_mex_name)
% 读取 MEX 侧壁面剪应力监控值，不改动任何粒子状态。
    [tau_bottom, tau_top] = feval( ...
        physics_mex_name, 'wall_shear_monitor', ...
        neighbor.pair_i, neighbor.pair_j, neighbor.dx_ij, neighbor.dy_ij, neighbor.r_ij, neighbor.dW_ij, ...
        pos, vel, wall_vel, Vol, B, n_fluid, DL, DH, mu, h);
end

function restart_state = make_restart_state(state)
    restart_state = struct( ...
        'pos', state.pos, ...
        'vel', state.vel, ...
        'rho', state.rho, ...
        'p', state.p, ...
        'drho_dt', state.drho_dt, ...
        'force', state.force, ...
        'force_prior', state.force_prior, ...
        't', state.t, ...
        'step', state.step);
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
    sig = sprintf('DL=%.12g|DH=%.12g|dp=%.12g|rho0=%.12g|mu=%.12g|Ub=%.12g|cf=%.12g|t=%.12g|oi=%.12g|si=%d|wall=thick-wall-noslip-dual-dt', ...
        DL, DH, dp, rho0, mu, U_bulk, c_f, t_end, output_interval, sort_interval);
end

function dt = cfl_time_step(vel, c_f, h, remain)
% 统一 CFL 时间步长：dt = 0.3 * h / (c + |v|_max)，不超过剩余时间
    signal = c_f + vecnorm(vel, 2, 2);
    max_signal = max(signal);
    max_signal = max(max_signal, 1e-12);
    dt = min(0.3 * h / max_signal, remain);
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
% X 方向周期性边界：粒子越界后立即包裹回 [0, DL)
    x = pos(1:n_fluid, 1);
    x_wrapped = mod(x, DL);
    wrapped = any(abs(x_wrapped - x) > 1e-12);
    pos(1:n_fluid, 1) = x_wrapped;
    void_buffer(buffer_width);
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
% 旧的硬壁回弹辅助函数；当前厚壁/no-penetration 主路径不再调用。
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
