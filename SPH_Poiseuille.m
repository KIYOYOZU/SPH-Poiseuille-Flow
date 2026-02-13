%% SPH_Poiseuille.m
% SPH 平板泊肃叶流动模拟 (MEX C+OpenMP 加速版)
% 解析解: u(y) = (1/2mu) * |dp/dx| * y * (H - y)

clear; clc; close all;

%% MEX 自动编译
mex_dir = fileparts(mfilename('fullpath'));
mex_src_dir = fullfile(mex_dir, 'mex');
build_dir = fullfile(mex_dir, 'build');
step_mex_name = 'sph_step_mex';

% 创建 build 目录
if ~exist(build_dir, 'dir')
    mkdir(build_dir);
    fprintf('创建 build 目录: %s\n', build_dir);
end

% 添加 build 目录到 MATLAB 路径
addpath(build_dir);

step_src = fullfile(mex_src_dir, 'sph_step_mex.c');
step_bin = fullfile(build_dir, [step_mex_name '.' mexext]);
if ~exist(step_src, 'file')
    error('未找到 MEX 源码: %s', step_src);
end

need_compile = ~exist(step_bin, 'file');
if ~need_compile
    d_src = dir(step_src);
    d_bin = dir(step_bin);
    need_compile = d_src.datenum > d_bin.datenum;
end

if need_compile
    fprintf('编译 %s -> %s ...\n', 'sph_step_mex.c', step_mex_name);
    try
        mex('-R2018a', '-O', 'COMPFLAGS="$COMPFLAGS /openmp"', ...
            '-output', step_mex_name, '-outdir', build_dir, step_src);
        fprintf('编译成功! 输出到: %s\n', build_dir);
    catch e
        error('MEX 编译失败，无法继续: %s', e.message);
    end
end

if ~exist(step_bin, 'file')
    error('MEX 二进制不存在，无法继续: %s', step_bin);
end

d_src = dir(step_src);
d_bin = dir(step_bin);
if d_bin.datenum < d_src.datenum
    error('MEX 二进制过期且未能更新: %s', step_bin);
end

fprintf('MEX 加速: 融合单步=true\n');

%% 参数设置
fprintf('SPH 平板泊肃叶流动模拟\n');

H = 1.0;            % 板间距
L = 2.0;            % 计算域长度 (x方向周期性)
dx = 0.02;          % 粒子间距
h = 1.3 * dx;       % 光滑长度

rho0 = 1.0;         % 参考密度
mu = 0.1;           % 动力粘度
nu = mu / rho0;

% 壁面有效厚度（SPHinXsys 风格）
wall_thickness = 10 * dx;
env_wall_thickness = str2double(strtrim(getenv('SPH_WALL_THICKNESS')));
if isfinite(env_wall_thickness) && env_wall_thickness > 0
    wall_thickness = env_wall_thickness * dx;
end

dpdx = -0.8;        % 压力梯度 -> u_max = |dpdx|*H^2/(8*mu) = 1.0
gx = -dpdx / rho0;

u_max_theory = (1/(2*mu)) * abs(dpdx) * (H/2)^2;
cs = 10 * max(u_max_theory, 1.0);

dt_cfl = 0.25 * h / cs;
dt_visc = 0.125 * h^2 / nu;
dt = 0.5 * min(dt_cfl, dt_visc);
t_end = 5.0;

env_t_end = str2double(strtrim(getenv('SPH_T_END')));
if isfinite(env_t_end) && env_t_end > 0
    t_end = env_t_end;
end

epsilon_xsph = 0.5;

% VTP 流场输出参数
enable_vtp_output = true;
vtp_n_frames = 10;           % 输出帧数
vtp_output_dir = 'vtp_output';
env_vtp_frames = str2double(strtrim(getenv('SPH_VTP_FRAMES')));
if isfinite(env_vtp_frames) && env_vtp_frames > 0
    vtp_n_frames = round(env_vtp_frames);
end
if strcmpi(strtrim(getenv('SPH_VTP_OUTPUT')), '0')
    enable_vtp_output = false;
end
fprintf('VTP输出: %s (帧数=%d, 目录=%s)\n', ...
    mat2str(enable_vtp_output), vtp_n_frames, vtp_output_dir);

enable_early_stop = true;
if strcmpi(strtrim(getenv('SPH_EARLY_STOP')), '0')
    enable_early_stop = false;
end

early_check_interval = 50;
early_stable_hold = 8;
early_min_time = 1.0;
early_du_tol = 2e-4;
early_profile_tol = 5e-4;

fprintf('u_max_theory=%.4f, cs=%.2f, dt=%.6f, t_end=%.3f\n', ...
    u_max_theory, cs, dt, t_end);
fprintf('稳态提前停止=%s (du<%.1e, profile<%.1e, 连续%d次, t>=%.2fs)\n', ...
    mat2str(enable_early_stop), early_du_tol, early_profile_tol, early_stable_hold, early_min_time);

% 壁面边界条件类型: 'dirichlet' | 'neumann_laminar' | 'neumann_wallmodel' | 'contact_only'
wall_bc_type = 'neumann_laminar';
env_wall_bc = strtrim(getenv('SPH_WALL_BC'));
if ~isempty(env_wall_bc)
    valid_bc_types = {'dirichlet', 'neumann_laminar', 'neumann_wallmodel', 'contact_only'};
    if ismember(env_wall_bc, valid_bc_types)
        wall_bc_type = env_wall_bc;
    else
        warning('无效的 SPH_WALL_BC="%s"，使用默认 neumann_laminar', env_wall_bc);
    end
end

fprintf('壁面边界条件: %s\n', wall_bc_type);

%% 粒子初始化
fprintf('\n初始化粒子...\n');

% 流体粒子
x_fluid = (dx/2 : dx : L-dx/2)';
y_fluid = (dx/2 : dx : H-dx/2)';
[X_fluid, Y_fluid] = meshgrid(x_fluid, y_fluid);
x_f = X_fluid(:);
y_f = Y_fluid(:);
n_fluid = length(x_f);

% 边界粒子 (单层壁面粒子，SPHinXsys 风格)
n_layers = 1;
x_wall = (dx/2 : dx : L-dx/2)';
n_wall_per_layer = length(x_wall);

% 下壁面 (y = -dx/2)，仅 1 层
x_bottom = x_wall;
y_bottom = ones(n_wall_per_layer, 1) * (-dx/2);

% 上壁面 (y = H + dx/2)，仅 1 层
x_top = x_wall;
y_top = ones(n_wall_per_layer, 1) * (H + dx/2);

% 合并边界粒子
x_b = [x_bottom; x_top];
y_b = [y_bottom; y_top];
n_boundary = length(x_b);
n_bottom = n_wall_per_layer;  % 单层
n_top = n_wall_per_layer;     % 单层

% 合并所有粒子
x = [x_f; x_b];
y = [y_f; y_b];
n_total = n_fluid + n_boundary;

vx = zeros(n_total, 1);
vy = zeros(n_total, 1);
rho = rho0 * ones(n_total, 1);
mass = rho0 * dx^2;

% 壁面粒子有效质量（SPHinXsys 风格）
mass_wall = rho0 * dx * wall_thickness;
fprintf('壁面参数: n_layers=%d, wall_thickness=%.3f (%.1f*dx), mass_wall/mass=%.1f\n', ...
    n_layers, wall_thickness, wall_thickness/dx, mass_wall/mass);

fprintf('流体粒子数: %d\n', n_fluid);
fprintf('边界粒子数: %d\n', n_boundary);
fprintf('总粒子数: %d\n', n_total);

%% 壁面速度预计算映射表
fprintf('预计算壁面镜像映射表...\n');

% 流体粒子的唯一y层（初始化时规则排列）
y_fluid_init = y(1:n_fluid);
y_layers = unique(round(y_fluid_init, 10));  % 唯一y层
n_layers_fluid = length(y_layers);

% 每个流体粒子属于哪一层
[~, ~, fluid_layer_id] = unique(round(y_fluid_init, 10));

% 每层粒子数（用于快速求层平均速度）
layer_particle_count = accumarray(fluid_layer_id, 1, [n_layers_fluid, 1]);

% 下壁面镜像y及插值权重
idx_bottom = (n_fluid+1):(n_fluid+n_bottom);
y_mirror_bottom = max(0, min(H, -y(idx_bottom)));
[wall_bottom_k1, wall_bottom_k2, wall_bottom_w1, wall_bottom_w2] = ...
    precompute_interp_weights(y_mirror_bottom, y_layers);

% 上壁面镜像y及插值权重
idx_top = (n_fluid+n_bottom+1):(n_fluid+n_bottom+n_top);
y_mirror_top = max(0, min(H, 2*H - y(idx_top)));
[wall_top_k1, wall_top_k2, wall_top_w1, wall_top_w2] = ...
    precompute_interp_weights(y_mirror_top, y_layers);

fprintf('壁面映射表预计算完成: %d 个流体层, %d 个壁面粒子\n', ...
    n_layers_fluid, n_boundary);

% Neumann BC 预计算：虚粒子到壁面距离 & 近壁流体层信息
d_bottom = abs(y(idx_bottom));          % 下壁面虚粒子到 y=0 的距离
d_top    = abs(y(idx_top) - H);         % 上壁面虚粒子到 y=H 的距离

% 动态 Neumann 梯度状态（直接力求和方案）
is_dynamic_neumann = ismember(wall_bc_type, {'neumann_laminar', 'neumann_wallmodel'});
bl_dudy_bottom_ema = 0;
bl_dudy_top_ema = 0;
bl_ema_alpha = 0.3;
bl_grad = zeros(2, 1);
bl_diag = zeros(8, 1);  % [n_bottom, L_bottom, tau_bottom, 0, n_top, L_top, tau_top, 0]
bl_diag_last = zeros(8, 1);
if is_dynamic_neumann
    fprintf('动态Neumann启用: 直接力求和方案, EMA alpha=%.2f\n', bl_ema_alpha);
end

%% 核函数参数
alpha_kernel = 10 / (7 * pi * h^2);  % 2D Cubic Spline 归一化系数
r_cut = 2 * h;

%% SPH主循环 (Cell-Linked List + Velocity Verlet)
fprintf('\n开始SPH计算...\n');
progress_bar_width = 50;

% 历史记录（预分配）
max_history = 5000;
time_history = zeros(max_history, 1);
u_max_history = zeros(max_history, 1);
history_idx = 0;

% Cell-Linked List 网格参数（y方向域范围含边界虚粒子）
y_min_domain = min(y) - dx;
y_max_domain = max(y) + dx;
n_cell_x = max(1, floor(L / r_cut));
n_cell_y = max(1, floor((y_max_domain - y_min_domain) / r_cut));
cell_size_x = L / n_cell_x;
cell_size_y = (y_max_domain - y_min_domain) / n_cell_y;
n_cells = n_cell_x * n_cell_y;

% 核函数自身贡献 W(r=0)
W_self = alpha_kernel * 2/3;

% 粘性稳定性时间步长（常数，提到循环外）
dt_visc = 0.125 * h^2 / nu;

% XSPH 修正初始化（第一步无历史数据，设为零）
vx_xsph = zeros(n_total, 1);
vy_xsph = zeros(n_total, 1);

% 初始力计算（Velocity Verlet 需要初始加速度）
fprintf('计算初始加速度...\n');

% 壁面速度边界条件（初始速度为零，此处主要为完整性）
if is_dynamic_neumann
    vx = apply_neumann_boundary_layer(vx, n_fluid, n_bottom, n_top, ...
        d_bottom, d_top, bl_dudy_bottom_ema, bl_dudy_top_ema);
else
    vx = update_wall_velocity_unified(vx, n_fluid, n_bottom, n_top, ...
        fluid_layer_id, layer_particle_count, n_layers_fluid, ...
        wall_bottom_k1, wall_bottom_k2, wall_bottom_w1, wall_bottom_w2, ...
        wall_top_k1, wall_top_k2, wall_top_w1, wall_top_w2, ...
        wall_bc_type);
end

[ax, ay, rho, p, vx_xsph, vy_xsph, n_pairs, bl_grad, bl_diag] = ...
    feval(step_mex_name, x, y, vx, vy, n_total, n_fluid, L, r_cut, ...
    y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, n_cells, ...
    h, alpha_kernel, W_self, mass, cs, rho0, mu, gx, 0, n_bottom, H, dx, mass_wall);

% 用初始 SPH 密度修正质量（消除核函数离散求和的系统偏差）
rho0_sph = mean(rho(1:n_fluid));
fprintf('SPH 初始密度: mean=%.4f, min=%.4f, max=%.4f (理论 rho0=%.1f)\n', ...
    rho0_sph, min(rho(1:n_fluid)), max(rho(1:n_fluid)), rho0);
mass_correction = rho0 / rho0_sph;
mass = mass * mass_correction;
fprintf('质量修正因子: %.4f, 修正后 mass=%.6e\n', mass_correction, mass);

% 重新计算初始力（使用修正后的 mass）
[ax, ay, rho, p, vx_xsph, vy_xsph, n_pairs, bl_grad, bl_diag] = ...
    feval(step_mex_name, x, y, vx, vy, n_total, n_fluid, L, r_cut, ...
    y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, n_cells, ...
    h, alpha_kernel, W_self, mass, cs, rho0, mu, gx, 0, n_bottom, H, dx, mass_wall);
fprintf('修正后 SPH 密度: mean=%.4f\n', mean(rho(1:n_fluid)));

fprintf('初始邻居对数: %d (平均每粒子 %.1f 个邻居)\n', n_pairs, 2*n_pairs/n_total);

% 稳态提前停止状态
profile_n_bins = 40;
profile_y_bins = linspace(0, H, profile_n_bins+1);
prev_check_u_max = NaN;
prev_check_profile = zeros(profile_n_bins, 1);
prev_check_valid = false;
stable_hit_count = 0;
early_stop_triggered = false;
early_stop_msg = '';

%% 主循环（Velocity Verlet + 自适应时间步长）
tic;
t_current = 0;
step = 0;
last_display_time = -1;

% VTP 帧调度初始化
if enable_vtp_output
    if ~exist(vtp_output_dir, 'dir'), mkdir(vtp_output_dir); end
    vtp_frame_interval = t_end / vtp_n_frames;
    vtp_next_frame_time = 0;
    vtp_frame_count = 0;
    vtp_last_save_time = -inf;
end

while t_current < t_end
    step = step + 1;

    % --- 自适应时间步长 ---
    v_sq = vx(1:n_fluid).^2 + vy(1:n_fluid).^2;
    v_max = sqrt(max(v_sq));
    dt_cfl = 0.25 * h / (cs + v_max);
    dt = 0.5 * min(dt_cfl, dt_visc);
    dt = min(dt, t_end - t_current);

    % --- Verlet 半步速度 ---
    vx(1:n_fluid) = vx(1:n_fluid) + 0.5 * ax(1:n_fluid) * dt;
    vy(1:n_fluid) = vy(1:n_fluid) + 0.5 * ay(1:n_fluid) * dt;

    % --- 全步位置 + XSPH 修正 ---
    x(1:n_fluid) = x(1:n_fluid) + (vx(1:n_fluid) + epsilon_xsph*vx_xsph(1:n_fluid)) * dt;
    y(1:n_fluid) = y(1:n_fluid) + (vy(1:n_fluid) + epsilon_xsph*vy_xsph(1:n_fluid)) * dt;

    % --- 周期性边界 ---
    x(1:n_fluid) = mod(x(1:n_fluid), L);

    % --- 壁面速度边界条件 ---
    if is_dynamic_neumann
        vx = apply_neumann_boundary_layer(vx, n_fluid, n_bottom, n_top, ...
            d_bottom, d_top, bl_dudy_bottom_ema, bl_dudy_top_ema);
    else
        vx = update_wall_velocity_unified(vx, n_fluid, n_bottom, n_top, ...
            fluid_layer_id, layer_particle_count, n_layers_fluid, ...
            wall_bottom_k1, wall_bottom_k2, wall_bottom_w1, wall_bottom_w2, ...
            wall_top_k1, wall_top_k2, wall_top_w1, wall_top_w2, ...
            wall_bc_type);
    end

    % --- SPH 单步计算（仅 MEX 路径） ---
    do_shepard = (mod(step, 30) == 0);
    [ax, ay, rho, p, vx_xsph, vy_xsph, n_pairs, bl_grad, bl_diag] = ...
        feval(step_mex_name, x, y, vx, vy, n_total, n_fluid, L, r_cut, ...
        y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, n_cells, ...
        h, alpha_kernel, W_self, mass, cs, rho0, mu, gx, double(do_shepard), n_bottom, H, dx, mass_wall);

    if is_dynamic_neumann
        dudy_bottom_raw = bl_grad(1);
        dudy_top_raw = bl_grad(2);

        if dudy_bottom_raw > 0 && bl_diag(1) >= 2
            bl_dudy_bottom_ema = (1 - bl_ema_alpha) * bl_dudy_bottom_ema + ...
                                 bl_ema_alpha * dudy_bottom_raw;
        end
        if dudy_top_raw < 0 && bl_diag(5) >= 2
            bl_dudy_top_ema = (1 - bl_ema_alpha) * bl_dudy_top_ema + ...
                              bl_ema_alpha * dudy_top_raw;
        end
        bl_diag_last = bl_diag;
    end

    % --- Verlet 完成速度 ---
    vx(1:n_fluid) = vx(1:n_fluid) + 0.5 * ax(1:n_fluid) * dt;
    vy(1:n_fluid) = vy(1:n_fluid) + 0.5 * ay(1:n_fluid) * dt;

    t_current = t_current + dt;

    % --- VTP 帧输出 ---
    if enable_vtp_output && t_current >= vtp_next_frame_time
        vtp_frame_count = vtp_frame_count + 1;
        particle_type = [zeros(n_fluid, 1); ones(n_total - n_fluid, 1)];
        write_vtp(fullfile(vtp_output_dir, ...
            sprintf('sph_frame_%04d.vtp', vtp_frame_count)), ...
            x, y, vx, vy, rho, p, particle_type, ...
            t_current, vtp_frame_count, n_total);
        vtp_last_save_time = t_current;
        vtp_next_frame_time = vtp_next_frame_time + vtp_frame_interval;
    end

    u_max_current = max(vx(1:n_fluid));

    % --- 稳态提前停止检查 ---
    if enable_early_stop && t_current >= early_min_time && mod(step, early_check_interval) == 0
        current_profile = compute_binned_profile_mean(y(1:n_fluid), vx(1:n_fluid), profile_y_bins);

        if prev_check_valid
            du_rel = abs(u_max_current - prev_check_u_max) / max(abs(u_max_current), 1e-12);
            profile_rel = norm(current_profile - prev_check_profile, 2) / ...
                max(norm(current_profile, 2), 1e-12);

            if du_rel < early_du_tol && profile_rel < early_profile_tol
                stable_hit_count = stable_hit_count + 1;
            else
                stable_hit_count = 0;
            end

            if stable_hit_count >= early_stable_hold
                early_stop_triggered = true;
                early_stop_msg = sprintf('触发稳态提前停止: t=%.3fs, du=%.2e, profile=%.2e, 连续%d次', ...
                    t_current, du_rel, profile_rel, stable_hit_count);
                fprintf('\n%s\n', early_stop_msg);
                break;
            end
        end

        prev_check_u_max = u_max_current;
        prev_check_profile = current_profile;
        prev_check_valid = true;
    end

    if is_dynamic_neumann && mod(step, 100) == 0
        fprintf(['\n[直接力求和] step=%d | bottom: n_wall=%d, L=%.3f, tau=%.4e, ' ...
                 'dudy_raw=%.4f, dudy_ema=%.4f | top: n_wall=%d, L=%.3f, tau=%.4e, ' ...
                 'dudy_raw=%.4f, dudy_ema=%.4f\n'], ...
            step, round(bl_diag_last(1)), bl_diag_last(2), bl_diag_last(3), ...
            bl_grad(1), bl_dudy_bottom_ema, ...
            round(bl_diag_last(5)), bl_diag_last(6), bl_diag_last(7), ...
            bl_grad(2), bl_dudy_top_ema);
    end

    % --- 进度显示 ---
    if t_current - last_display_time >= t_end/100 || t_current >= t_end
        last_display_time = t_current;

        % 记录历史
        history_idx = history_idx + 1;
        if history_idx > max_history
            max_history = max_history * 2;
            time_history(end+1:max_history) = 0;
            u_max_history(end+1:max_history) = 0;
        end
        time_history(history_idx) = t_current;
        u_max_history(history_idx) = u_max_current;

        % 进度条
        progress = t_current / t_end;
        progress_chars = floor(progress * progress_bar_width);
        fprintf('\r进度: [');
        for pb = 1:progress_chars
            fprintf('=');
        end
        for pb = progress_chars+1:progress_bar_width
            fprintf(' ');
        end
        fprintf('] %5.1f%% | 步数: %d | t=%.3fs | dt=%.2e | u_max=%.4f m/s', ...
                progress*100, step, t_current, dt, u_max_current);
    end
end

% VTP 最终帧保底输出
if enable_vtp_output && vtp_last_save_time < t_current - dt/2
    vtp_frame_count = vtp_frame_count + 1;
    particle_type = [zeros(n_fluid, 1); ones(n_total - n_fluid, 1)];
    write_vtp(fullfile(vtp_output_dir, ...
        sprintf('sph_frame_%04d.vtp', vtp_frame_count)), ...
        x, y, vx, vy, rho, p, particle_type, ...
        t_current, vtp_frame_count, n_total);
    fprintf('VTP 最终帧已保存 (frame %d, t=%.4f)\n', vtp_frame_count, t_current);
end

elapsed_time = toc;
fprintf('\n\n计算完成! 耗时: %.2f 秒\n', elapsed_time);

if early_stop_triggered
    fprintf('%s\n', early_stop_msg);
end

if history_idx == 0 || time_history(history_idx) < t_current
    history_idx = history_idx + 1;
    if history_idx > max_history
        max_history = max_history * 2;
        time_history(end+1:max_history) = 0;
        u_max_history(end+1:max_history) = 0;
    end
    time_history(history_idx) = t_current;
    u_max_history(history_idx) = max(vx(1:n_fluid));
end

% 截断历史数组
time_history = time_history(1:history_idx);
u_max_history = u_max_history(1:history_idx);

%% 后处理与验证
fprintf('\n后处理与验证\n');

x_final = x(1:n_fluid);
y_final = y(1:n_fluid);
vx_final = vx(1:n_fluid);

% 解析解
y_analytical = linspace(0, H, 100);
u_analytical = (1/(2*mu)) * abs(dpdx) * y_analytical .* (H - y_analytical);

% SPH 平均速度剖面
n_bins = 40;
y_bins = linspace(0, H, n_bins+1);
y_centers = (y_bins(1:end-1) + y_bins(2:end)) / 2;
u_sph = zeros(n_bins, 1);
u_std = zeros(n_bins, 1);

for i = 1:n_bins
    mask = (y_final >= y_bins(i)) & (y_final < y_bins(i+1));
    if sum(mask) > 0
        u_sph(i) = mean(vx_final(mask));
        u_std(i) = std(vx_final(mask));
    end
end

% 解析解在 bin 中心
u_analytical_bins = (1/(2*mu)) * abs(dpdx) * y_centers .* (H - y_centers);

% L2 误差
valid_bins = u_sph > 0;
L2_error = sqrt(mean((u_sph(valid_bins) - u_analytical_bins(valid_bins)').^2)) / max(u_analytical_bins);
fprintf('L2相对误差: %.2f%%\n', L2_error * 100);

%% 绘图
figure('Position', [100, 100, 1400, 450]);

% 图1: 速度剖面对比
subplot(1, 3, 1);
plot(u_analytical, y_analytical, 'b-', 'LineWidth', 2, 'DisplayName', '解析解');
hold on;
% 只绘制有粒子的 bin（过滤掉 u_sph = 0 的点）
valid_bins = u_sph > 0;
errorbar(u_sph(valid_bins), y_centers(valid_bins), [], [], u_std(valid_bins), u_std(valid_bins), 'ro', 'MarkerSize', 6, ...
    'MarkerFaceColor', 'r', 'DisplayName', 'SPH结果');
xlabel('速度 u [m/s]', 'FontSize', 12);
ylabel('y [m]', 'FontSize', 12);
title(sprintf('速度剖面对比 (L2误差: %.2f%%, 有效bin: %d/%d)', L2_error*100, sum(valid_bins), n_bins), 'FontSize', 14);
legend('Location', 'best');
grid on;
xlim([0, max(u_analytical)*1.1]);
ylim([0, H]);

% 图2: 粒子分布与速度场
subplot(1, 3, 2);
scatter(x_final, y_final, 15, vx_final, 'filled');
colorbar;
colormap(jet);
xlabel('x [m]', 'FontSize', 12);
ylabel('y [m]', 'FontSize', 12);
title('粒子分布与速度场', 'FontSize', 14);
axis equal;
xlim([0, L]);
ylim([-0.1, H+0.1]);

% 图3: 收敛历史
subplot(1, 3, 3);
plot(time_history, u_max_history, 'b-', 'LineWidth', 1.5);
hold on;
yline(u_max_theory, 'r--', 'LineWidth', 2, 'DisplayName', '理论值');
xlabel('时间 [s]', 'FontSize', 12);
ylabel('最大速度 [m/s]', 'FontSize', 12);
title('收敛历史', 'FontSize', 14);
legend('SPH', '理论值', 'Location', 'best');
grid on;

% 保存图片
saveas(gcf, 'SPH_Poiseuille_result.png');
fprintf('结果图片已保存: SPH_Poiseuille_result.png\n');

%% 输出验证结果
fprintf('\n验证总结\n');
fprintf('理论最大速度: %.4f m/s\n', u_max_theory);
fprintf('SPH最大速度:  %.4f m/s\n', max(vx_final));
fprintf('SPH最小速度:  %.4f m/s\n', min(vx_final));
fprintf('相对偏差:     %.2f%%\n', abs(max(vx_final) - u_max_theory) / u_max_theory * 100);
fprintf('L2相对误差:   %.2f%%\n', L2_error * 100);

% 诊断：检查速度异常低的粒子
low_vel_threshold = 0.01;  % 速度阈值
low_vel_mask = vx_final < low_vel_threshold;
n_low_vel = sum(low_vel_mask);
if n_low_vel > 0
    fprintf('\n--- 速度异常诊断 ---\n');
    fprintf('速度 < %.3f m/s 的粒子数: %d (占比 %.1f%%)\n', ...
        low_vel_threshold, n_low_vel, n_low_vel/n_fluid*100);
    fprintf('这些粒子的 y 坐标范围: [%.4f, %.4f]\n', ...
        min(y_final(low_vel_mask)), max(y_final(low_vel_mask)));
    fprintf('这些粒子的平均速度: %.6f m/s\n', mean(vx_final(low_vel_mask)));
end

% 速度分布统计
fprintf('\n--- 速度分布统计 ---\n');
fprintf('速度范围: [%.4f, %.4f] m/s\n', min(vx_final), max(vx_final));
fprintf('速度均值: %.4f m/s\n', mean(vx_final));
fprintf('速度标准差: %.4f m/s\n', std(vx_final));
vel_percentiles = prctile(vx_final, [1, 5, 10, 25, 50, 75, 90, 95, 99]);
fprintf('速度百分位数:\n');
fprintf('  1%%: %.4f, 5%%: %.4f, 10%%: %.4f, 25%%: %.4f\n', ...
    vel_percentiles(1), vel_percentiles(2), vel_percentiles(3), vel_percentiles(4));
fprintf('  50%%: %.4f, 75%%: %.4f, 90%%: %.4f, 95%%: %.4f, 99%%: %.4f\n', ...
    vel_percentiles(5), vel_percentiles(6), vel_percentiles(7), vel_percentiles(8), vel_percentiles(9));

if L2_error < 0.05
    fprintf('验证通过! L2误差 < 5%%\n');
else
    fprintf('验证未通过, 需要调整参数\n');
end

%% 壁面剪应力验证
fprintf('\n--- 壁面剪应力验证 ---\n');
% 计算各层平均速度
vx_final_fluid = vx(1:n_fluid);
vx_layer_sum = accumarray(fluid_layer_id, vx_final_fluid, [n_layers_fluid, 1]);
vx_layer_avg = vx_layer_sum ./ layer_particle_count;

% 解析壁面剪应力: tau_w = mu * |du/dy|_wall = |dpdx| * H/2
tau_w_analytical = abs(dpdx) * H / 2;

% SPH 壁面 du/dy（根据边界条件类型）
switch wall_bc_type
    case 'dirichlet'
        % 单侧差分: du/dy ≈ u(layer1) / y(layer1)
        dudy_bottom = vx_layer_avg(1) / y_layers(1);
        dudy_top = -vx_layer_avg(end) / (H - y_layers(end));
    case {'neumann_laminar', 'neumann_wallmodel'}
        dudy_bottom = bl_dudy_bottom_ema;
        dudy_top = bl_dudy_top_ema;
    case 'contact_only'
        % 仅接触力: 用单侧差分估计实际产生的 du/dy
        dudy_bottom = vx_layer_avg(1) / y_layers(1);
        dudy_top = -vx_layer_avg(end) / (H - y_layers(end));
end

tau_w_bottom = mu * abs(dudy_bottom);
tau_w_top = mu * abs(dudy_top);
err_bottom = abs(tau_w_bottom - tau_w_analytical) / tau_w_analytical;
err_top = abs(tau_w_top - tau_w_analytical) / tau_w_analytical;

fprintf('解析 tau_w = %.4f Pa\n', tau_w_analytical);
fprintf('下壁面: tau_w = %.4f Pa, 相对误差 = %.2f%%\n', ...
    tau_w_bottom, err_bottom * 100);
fprintf('上壁面: tau_w = %.4f Pa, 相对误差 = %.2f%%\n', ...
    tau_w_top, err_top * 100);
if max(err_bottom, err_top) < 0.05
    fprintf('壁面剪应力验证通过! 误差 < 5%%\n');
else
    fprintf('壁面剪应力验证未通过, 最大误差 %.2f%%\n', ...
        max(err_bottom, err_top) * 100);
end

fprintf('\n程序运行完毕!\n');

%% Local Functions




function [k1, k2, w1, w2] = precompute_interp_weights(y_query, y_layers)
% 预计算线性插值的索引和权重: w1*v(k1) + w2*v(k2)

    n_q = length(y_query);
    n_layers = length(y_layers);
    k1 = ones(n_q, 1);
    k2 = ones(n_q, 1);
    w1 = ones(n_q, 1);
    w2 = zeros(n_q, 1);

    for i = 1:n_q
        yq = y_query(i);
        if yq <= y_layers(1)
            k1(i) = 1; k2(i) = 1; w1(i) = 1; w2(i) = 0;
        elseif yq >= y_layers(end)
            k1(i) = n_layers; k2(i) = n_layers; w1(i) = 1; w2(i) = 0;
        else
            idx = find(y_layers <= yq, 1, 'last');
            if idx >= n_layers
                idx = n_layers - 1;
            end
            k1(i) = idx;
            k2(i) = idx + 1;
            frac = (yq - y_layers(idx)) / (y_layers(idx+1) - y_layers(idx));
            w1(i) = 1 - frac;
            w2(i) = frac;
        end
    end
end


function profile_mean = compute_binned_profile_mean(y_values, u_values, y_edges)
% 计算分箱平均速度剖面（用于稳态判据）

    n_bins = length(y_edges) - 1;
    bin_width = y_edges(2) - y_edges(1);
    bin_idx = floor((y_values - y_edges(1)) / bin_width) + 1;
    bin_idx(bin_idx < 1) = 1;
    bin_idx(bin_idx > n_bins) = n_bins;

    u_sum = accumarray(bin_idx, u_values, [n_bins, 1], @sum, 0);
    u_count = accumarray(bin_idx, 1, [n_bins, 1], @sum, 0);
    profile_mean = u_sum ./ max(u_count, 1);
end


function vx = apply_neumann_boundary_layer(vx, n_fluid, n_bottom, n_top, ...
    d_bottom, d_top, dudy_bottom, dudy_top)
% 动态 Neumann BC: 使用边界层全局拟合梯度更新虚粒子速度

    idx_bottom = (n_fluid+1):(n_fluid+n_bottom);
    idx_top = (n_fluid+n_bottom+1):(n_fluid+n_bottom+n_top);

    vx(idx_bottom) = -d_bottom .* dudy_bottom;
    vx(idx_top)    =  d_top    .* dudy_top;
end

function vx = update_wall_velocity_unified(vx, n_fluid, n_bottom, n_top, ...
    fluid_layer_id, layer_particle_count, n_layers_fluid, ...
    wall_bottom_k1, wall_bottom_k2, wall_bottom_w1, wall_bottom_w2, ...
    wall_top_k1, wall_top_k2, wall_top_w1, wall_top_w2, ...
    wall_bc_type)
% 统一壁面速度更新函数
% 支持模式: dirichlet / contact_only

    % 计算各层平均速度（所有模式共用）
    vx_fluid = vx(1:n_fluid);
    vx_layer_sum = accumarray(fluid_layer_id, vx_fluid, [n_layers_fluid, 1]);
    vx_layer = vx_layer_sum ./ layer_particle_count;

    % 壁面粒子索引
    idx_bottom = (n_fluid+1):(n_fluid+n_bottom);
    idx_top = (n_fluid+n_bottom+1):(n_fluid+n_bottom+n_top);

    % 镜像点速度插值（所有模式共用）
    vx_mirror_bottom = wall_bottom_w1 .* vx_layer(wall_bottom_k1) + ...
                       wall_bottom_w2 .* vx_layer(wall_bottom_k2);
    vx_mirror_top = wall_top_w1 .* vx_layer(wall_top_k1) + ...
                    wall_top_w2 .* vx_layer(wall_top_k2);

    switch wall_bc_type
        case 'dirichlet'
            % Morris (1997) 镜像速度法: u_ghost = -u_mirror
            vx(idx_bottom) = -vx_mirror_bottom;
            vx(idx_top)    = -vx_mirror_top;

        case 'contact_only'
            % 仅接触力: 虚粒子速度恒为0, 不施加切应力条件
            vx(idx_bottom) = 0;
            vx(idx_top)    = 0;

        otherwise
            error('未知的壁面边界条件类型: %s', wall_bc_type);
    end
end

function write_vtp(filepath, x, y, vx, vy, rho, p, ptype, time, step, np)
% 输出 VTK PolyData XML 格式 (.vtp) 文件
% 字段: Velocity(3D), Density, Pressure, ParticleType(Int32)
fid = fopen(filepath, 'w');
if fid == -1, error('无法创建VTP文件: %s', filepath); end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, '<?xml version="1.0"?>\n');
fprintf(fid, '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n');
fprintf(fid, '<PolyData>\n');
fprintf(fid, '<FieldData>\n');
fprintf(fid, '  <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">%.6e</DataArray>\n', time);
fprintf(fid, '  <DataArray type="Int32" Name="TimeStep" NumberOfTuples="1" format="ascii">%d</DataArray>\n', step);
fprintf(fid, '</FieldData>\n');
fprintf(fid, '<Piece NumberOfPoints="%d" NumberOfVerts="%d" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n', np, np);

% PointData
fprintf(fid, '<PointData>\n');
fprintf(fid, '  <DataArray type="Float64" Name="Velocity" NumberOfComponents="3" format="ascii">\n');
for i = 1:np
    fprintf(fid, '    %.6e %.6e 0.0\n', vx(i), vy(i));
end
fprintf(fid, '  </DataArray>\n');
fprintf(fid, '  <DataArray type="Float64" Name="Density" format="ascii">\n');
fprintf(fid, '    %.6e\n', rho);
fprintf(fid, '  </DataArray>\n');
fprintf(fid, '  <DataArray type="Float64" Name="Pressure" format="ascii">\n');
fprintf(fid, '    %.6e\n', p);
fprintf(fid, '  </DataArray>\n');
fprintf(fid, '  <DataArray type="Int32" Name="ParticleType" format="ascii">\n');
fprintf(fid, '    %d\n', int32(ptype));
fprintf(fid, '  </DataArray>\n');
fprintf(fid, '</PointData>\n');

% Points
fprintf(fid, '<Points>\n');
fprintf(fid, '  <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n');
for i = 1:np
    fprintf(fid, '    %.6e %.6e 0.0\n', x(i), y(i));
end
fprintf(fid, '  </DataArray>\n');
fprintf(fid, '</Points>\n');

% Verts
fprintf(fid, '<Verts>\n');
fprintf(fid, '  <DataArray type="Int32" Name="connectivity" format="ascii">\n');
fprintf(fid, '    %d\n', (0:np-1)');
fprintf(fid, '  </DataArray>\n');
fprintf(fid, '  <DataArray type="Int32" Name="offsets" format="ascii">\n');
fprintf(fid, '    %d\n', (1:np)');
fprintf(fid, '  </DataArray>\n');
fprintf(fid, '</Verts>\n');

fprintf(fid, '</Piece>\n');
fprintf(fid, '</PolyData>\n');
fprintf(fid, '</VTKFile>\n');

fprintf('  VTP frame %d saved: %s (t=%.4f)\n', step, filepath, time);
end
