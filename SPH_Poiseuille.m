%% SPH_Poiseuille.m
% SPH 平板泊肃叶流动模拟 (MEX C+OpenMP 加速版)
% 解析解: u(y) = (1/2mu) * |dp/dx| * y * (H - y)

clear; clc; close all;

%% MEX 自动编译
mex_dir = fileparts(mfilename('fullpath'));
mex_src_dir = fullfile(mex_dir, 'mex');
build_dir = fullfile(mex_dir, 'build');
use_mex_step = false;
step_mex_name = 'sph_step_mex_v1';

% 创建 build 目录
if ~exist(build_dir, 'dir')
    mkdir(build_dir);
    fprintf('创建 build 目录: %s\n', build_dir);
end

% 添加 build 目录到 MATLAB 路径
addpath(build_dir);

mex_sources = {
    'sph_step_mex.c', step_mex_name, 'use_mex_step'
};
for m = 1:size(mex_sources, 1)
    src = fullfile(mex_src_dir, mex_sources{m, 1});
    mex_name = mex_sources{m, 2};
    bin = fullfile(build_dir, [mex_name '.' mexext]);
    if ~exist(src, 'file'), continue; end
    need_compile = ~exist(bin, 'file');
    if ~need_compile
        d_src = dir(src); d_bin = dir(bin);
        need_compile = d_src.datenum > d_bin.datenum;
    end
    if need_compile
        fprintf('编译 %s -> %s ...\n', mex_sources{m, 1}, mex_name);
        try
            mex('-R2018a', '-O', 'COMPFLAGS="$COMPFLAGS /openmp"', ...
                '-output', mex_name, '-outdir', build_dir, src);
            fprintf('编译成功! 输出到: %s\n', build_dir);
        catch e
            fprintf('编译失败: %s\n', e.message);
        end
    end
    if exist(bin, 'file'), eval([mex_sources{m, 3} ' = true;']); end
end

if strcmpi(strtrim(getenv('SPH_FORCE_MATLAB')), '1')
    use_mex_step = false;
    fprintf('检测到 SPH_FORCE_MATLAB=1，强制使用 MATLAB 回退实现。\n');
end

fprintf('MEX 加速: 融合单步=%s\n', mat2str(use_mex_step));

%% 参数设置
fprintf('SPH 平板泊肃叶流动模拟\n');

H = 1.0;            % 板间距
L = 2.0;            % 计算域长度 (x方向周期性)
dx = 0.02;          % 粒子间距
h = 1.3 * dx;       % 光滑长度

rho0 = 1.0;         % 参考密度
mu = 1.0;           % 动力粘度
nu = mu / rho0;

dpdx = -8.0;        % 压力梯度 -> u_max = |dpdx|*H^2/(8*mu) = 1.0
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

%% 粒子初始化
fprintf('\n初始化粒子...\n');

% 流体粒子
x_fluid = (dx/2 : dx : L-dx/2)';
y_fluid = (dx/2 : dx : H-dx/2)';
[X_fluid, Y_fluid] = meshgrid(x_fluid, y_fluid);
x_f = X_fluid(:);
y_f = Y_fluid(:);
n_fluid = length(x_f);

% 边界粒子 (上下壁面各3层虚粒子)
n_layers = 3;
x_wall = (dx/2 : dx : L-dx/2)';
n_wall_per_layer = length(x_wall);

% 下壁面 (y < 0)
x_bottom = repmat(x_wall, n_layers, 1);
y_bottom = [];
for layer = 1:n_layers
    y_bottom = [y_bottom; ones(n_wall_per_layer, 1) * (-dx/2 - (layer-1)*dx)];
end

% 上壁面 (y > H)
x_top = repmat(x_wall, n_layers, 1);
y_top = [];
for layer = 1:n_layers
    y_top = [y_top; ones(n_wall_per_layer, 1) * (H + dx/2 + (layer-1)*dx)];
end

% 合并边界粒子
x_b = [x_bottom; x_top];
y_b = [y_bottom; y_top];
n_boundary = length(x_b);
n_bottom = n_layers * n_wall_per_layer;
n_top = n_layers * n_wall_per_layer;

% 合并所有粒子
x = [x_f; x_b];
y = [y_f; y_b];
n_total = n_fluid + n_boundary;

vx = zeros(n_total, 1);
vy = zeros(n_total, 1);
rho = rho0 * ones(n_total, 1);
mass = rho0 * dx^2;

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

% 半模板：[自身, 右, 左上, 上, 右上]（5个方向，覆盖所有不重复邻居对）
% 注意：不需要[-1,0]，因为(cx,cy)→(cx+1,cy)已覆盖所有水平相邻对
stencil = [0,0; 1,0; -1,1; 0,1; 1,1];

% 核函数自身贡献 W(r=0)
W_self = alpha_kernel * 2/3;

% 粘性稳定性时间步长（常数，提到循环外）
dt_visc = 0.125 * h^2 / nu;

% 预分配邻居列表数组
max_pairs = n_total * 30;
pair_i_buf = zeros(max_pairs, 1);
pair_j_buf = zeros(max_pairs, 1);
dx_pair_buf = zeros(max_pairs, 1);
dy_pair_buf = zeros(max_pairs, 1);
r_pair_buf = zeros(max_pairs, 1);

% XSPH 修正初始化（第一步无历史数据，设为零）
vx_xsph = zeros(n_total, 1);
vy_xsph = zeros(n_total, 1);

% 初始力计算（Velocity Verlet 需要初始加速度）
fprintf('计算初始加速度...\n');

% 镜像速度边界条件（初始速度为零，此处主要为完整性）
vx = update_wall_velocity_fast(vx, n_fluid, n_bottom, n_top, ...
    fluid_layer_id, layer_particle_count, n_layers_fluid, ...
    wall_bottom_k1, wall_bottom_k2, wall_bottom_w1, wall_bottom_w2, ...
    wall_top_k1, wall_top_k2, wall_top_w1, wall_top_w2);

if use_mex_step
    [ax, ay, rho, p, vx_xsph, vy_xsph, n_pairs] = ...
        feval(step_mex_name, x, y, vx, vy, n_total, n_fluid, L, r_cut, ...
        y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, n_cells, ...
        h, alpha_kernel, W_self, mass, cs, rho0, mu, gx, 0);
else
    [pair_i, pair_j, dx_pair, dy_pair, r_pair, n_pairs] = ...
        cell_linked_list_search(x, y, n_total, L, r_cut, ...
        y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, ...
        n_cells, stencil, pair_i_buf, pair_j_buf, dx_pair_buf, dy_pair_buf, r_pair_buf);

    [ax, ay, rho, p, ~, ~, ~, vx_xsph, vy_xsph] = ...
        sph_compute_forces(pair_i, pair_j, dx_pair, dy_pair, r_pair, n_pairs, ...
        x, y, vx, vy, n_total, n_fluid, h, alpha_kernel, W_self, mass, ...
        cs, rho0, mu, gx, epsilon_xsph, false);
end

% 用初始 SPH 密度修正质量（消除核函数离散求和的系统偏差）
rho0_sph = mean(rho(1:n_fluid));
fprintf('SPH 初始密度: mean=%.4f, min=%.4f, max=%.4f (理论 rho0=%.1f)\n', ...
    rho0_sph, min(rho(1:n_fluid)), max(rho(1:n_fluid)), rho0);
mass_correction = rho0 / rho0_sph;
mass = mass * mass_correction;
fprintf('质量修正因子: %.4f, 修正后 mass=%.6e\n', mass_correction, mass);

% 重新计算初始力（使用修正后的 mass）
if use_mex_step
    [ax, ay, rho, p, vx_xsph, vy_xsph, n_pairs] = ...
        feval(step_mex_name, x, y, vx, vy, n_total, n_fluid, L, r_cut, ...
        y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, n_cells, ...
        h, alpha_kernel, W_self, mass, cs, rho0, mu, gx, 0);
else
    [ax, ay, rho, p, ~, ~, ~, vx_xsph, vy_xsph] = ...
        sph_compute_forces(pair_i, pair_j, dx_pair, dy_pair, r_pair, n_pairs, ...
        x, y, vx, vy, n_total, n_fluid, h, alpha_kernel, W_self, mass, ...
        cs, rho0, mu, gx, epsilon_xsph, false);
end
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

    % --- 壁面镜像速度 ---
    vx = update_wall_velocity_fast(vx, n_fluid, n_bottom, n_top, ...
        fluid_layer_id, layer_particle_count, n_layers_fluid, ...
        wall_bottom_k1, wall_bottom_k2, wall_bottom_w1, wall_bottom_w2, ...
        wall_top_k1, wall_top_k2, wall_top_w1, wall_top_w2);

    % --- SPH 单步计算（融合 MEX 或 MATLAB 回退） ---
    do_shepard = (mod(step, 30) == 0);
    if use_mex_step
        [ax, ay, rho, p, vx_xsph, vy_xsph, n_pairs] = ...
            feval(step_mex_name, x, y, vx, vy, n_total, n_fluid, L, r_cut, ...
            y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, n_cells, ...
            h, alpha_kernel, W_self, mass, cs, rho0, mu, gx, double(do_shepard));
    else
        [pair_i, pair_j, dx_pair, dy_pair, r_pair, n_pairs] = ...
            cell_linked_list_search(x, y, n_total, L, r_cut, ...
            y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, ...
            n_cells, stencil, pair_i_buf, pair_j_buf, dx_pair_buf, dy_pair_buf, r_pair_buf);

        [ax, ay, rho, p, ~, ~, ~, vx_xsph, vy_xsph] = ...
            sph_compute_forces(pair_i, pair_j, dx_pair, dy_pair, r_pair, n_pairs, ...
            x, y, vx, vy, n_total, n_fluid, h, alpha_kernel, W_self, mass, ...
            cs, rho0, mu, gx, epsilon_xsph, do_shepard);
    end

    % --- Verlet 完成速度 ---
    vx(1:n_fluid) = vx(1:n_fluid) + 0.5 * ax(1:n_fluid) * dt;
    vy(1:n_fluid) = vy(1:n_fluid) + 0.5 * ay(1:n_fluid) * dt;

    t_current = t_current + dt;

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
errorbar(u_sph, y_centers, [], [], u_std, u_std, 'ro', 'MarkerSize', 6, ...
    'MarkerFaceColor', 'r', 'DisplayName', 'SPH结果');
xlabel('速度 u [m/s]', 'FontSize', 12);
ylabel('y [m]', 'FontSize', 12);
title(sprintf('速度剖面对比 (L2误差: %.2f%%)', L2_error*100), 'FontSize', 14);
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
fprintf('相对偏差:     %.2f%%\n', abs(max(vx_final) - u_max_theory) / u_max_theory * 100);
fprintf('L2相对误差:   %.2f%%\n', L2_error * 100);

if L2_error < 0.05
    fprintf('验证通过! L2误差 < 5%%\n');
else
    fprintf('验证未通过, 需要调整参数\n');
end

fprintf('\n程序运行完毕!\n');

%% Local Functions

function [pair_i, pair_j, dx_pair, dy_pair, r_pair, n_pairs] = ...
        cell_linked_list_search(x, y, n_total, L, r_cut, ...
        y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, ...
        n_cells, stencil, pair_i_buf, pair_j_buf, dx_pair_buf, dy_pair_buf, r_pair_buf)
% Cell-Linked List 邻居搜索（半模板遍历）

    % 粒子→单元映射 + 排序构建索引
    cell_ix = min(max(floor(mod(x, L) / cell_size_x) + 1, 1), n_cell_x);
    cell_iy = min(max(floor((y - y_min_domain) / cell_size_y) + 1, 1), n_cell_y);
    cell_id = (cell_iy - 1) * n_cell_x + cell_ix;

    [cell_id_sorted, sort_idx] = sort(cell_id);
    particle_sorted = sort_idx;
    cell_start = zeros(n_cells, 1);
    cell_end = zeros(n_cells, 1);
    changes = [1; find(diff(cell_id_sorted)) + 1; n_total + 1];
    for kk = 1:length(changes)-1
        cid = cell_id_sorted(changes(kk));
        cell_start(cid) = changes(kk);
        cell_end(cid) = changes(kk+1) - 1;
    end

    % 遍历单元对
    n_pairs = 0;
    r_cut_sq = r_cut^2;

    for cy = 1:n_cell_y
        for cx = 1:n_cell_x
            cid1 = (cy-1)*n_cell_x + cx;
            if cell_start(cid1) == 0, continue; end

            for s = 1:size(stencil, 1)
                ncx = cx + stencil(s, 1);
                ncy = cy + stencil(s, 2);

                % y方向越界跳过（非周期性）
                if ncy < 1 || ncy > n_cell_y, continue; end

                % x方向周期性 wrap
                x_shift = 0;
                if ncx < 1
                    ncx = ncx + n_cell_x;
                    x_shift = L;
                elseif ncx > n_cell_x
                    ncx = ncx - n_cell_x;
                    x_shift = -L;
                end

                cid2 = (ncy-1)*n_cell_x + ncx;
                if cell_start(cid2) == 0, continue; end

                % 遍历两个单元中的粒子对
                for ii = cell_start(cid1):cell_end(cid1)
                    pi_idx = particle_sorted(ii);
                    if cid1 == cid2
                        jj_start = ii + 1;
                    else
                        jj_start = cell_start(cid2);
                    end
                    for jj = jj_start:cell_end(cid2)
                        pj_idx = particle_sorted(jj);
                        dxx = x(pi_idx) - x(pj_idx) + x_shift;
                        dyy = y(pi_idx) - y(pj_idx);
                        rr_sq = dxx^2 + dyy^2;
                        if rr_sq < r_cut_sq && rr_sq > 0
                            n_pairs = n_pairs + 1;
                            pair_i_buf(n_pairs) = pi_idx;
                            pair_j_buf(n_pairs) = pj_idx;
                            dx_pair_buf(n_pairs) = dxx;
                            dy_pair_buf(n_pairs) = dyy;
                            r_pair_buf(n_pairs) = sqrt(rr_sq);
                        end
                    end
                end
            end
        end
    end

    pair_i = pair_i_buf(1:n_pairs);
    pair_j = pair_j_buf(1:n_pairs);
    dx_pair = dx_pair_buf(1:n_pairs);
    dy_pair = dy_pair_buf(1:n_pairs);
    r_pair = r_pair_buf(1:n_pairs);
end

function [ax, ay, rho, p_out, W_pair, rho_i, rho_j, vx_xsph_out, vy_xsph_out] = ...
        sph_compute_forces(pair_i, pair_j, dx_pair, dy_pair, r_pair, n_pairs, ...
        x, y, vx, vy, n_total, n_fluid, h, alpha_kernel, W_self, mass, ...
        cs, rho0, mu, gx, epsilon_xsph, do_shepard)
% SPH 力计算（核函数 + 密度 + 压力 + 加速度 + XSPH）
% Morris (1997) 粘性项 + Shepard 密度修正

    % 核函数（向量化，只对 n_pairs 个邻居对）
    q_pair = r_pair / h;
    W_pair = zeros(n_pairs, 1);
    dWdr_pair = zeros(n_pairs, 1);

    mask1 = q_pair < 1;
    W_pair(mask1) = alpha_kernel * (2/3 - q_pair(mask1).^2 + 0.5*q_pair(mask1).^3);
    dWdr_pair(mask1) = alpha_kernel * (-2*q_pair(mask1) + 1.5*q_pair(mask1).^2) / h;

    mask2 = (q_pair >= 1) & (q_pair < 2);
    W_pair(mask2) = alpha_kernel * (1/6) * (2 - q_pair(mask2)).^3;
    dWdr_pair(mask2) = alpha_kernel * (-0.5) * (2 - q_pair(mask2)).^2 / h;

    % 密度求和（对称累加 + 自身贡献）
    rho = mass * (accumarray(pair_i, W_pair, [n_total,1]) + ...
                  accumarray(pair_j, W_pair, [n_total,1]) + W_self);

    % Shepard 密度修正（使用未修正密度计算归一化因子，避免循环依赖）
    if do_shepard
        rho_raw = rho;  % 保存未修正密度
        % Shepard 归一化因子: S_i = Σ_j (m/ρ_j) * W_ij + (m/ρ_i)*W_self
        vol_j_W_ij = mass * W_pair ./ rho_raw(pair_j);
        vol_i_W_ji = mass * W_pair ./ rho_raw(pair_i);
        shepard_factor = accumarray(pair_i, vol_j_W_ij, [n_total,1]) + ...
                         accumarray(pair_j, vol_i_W_ji, [n_total,1]) + ...
                         mass * W_self ./ rho_raw;
        shepard_factor(shepard_factor < 0.5) = 0.5;
        shepard_factor(shepard_factor > 1.5) = 1.5;
        rho = rho_raw ./ shepard_factor;
    end

    % 压力（弱可压缩状态方程）
    p_out = cs^2 * (rho - rho0);

    % 方向向量
    r_safe = max(r_pair, 1e-12);
    ex = dx_pair ./ r_safe;
    ey = dy_pair ./ r_safe;

    % 提取粒子对属性
    p_i = p_out(pair_i);  p_j = p_out(pair_j);
    rho_i = rho(pair_i);  rho_j = rho(pair_j);

    % 压力项：f_ij = -m * (pi/ρi² + pj/ρj²) * dW/dr * e_ij
    press_coef = -mass * (p_i./rho_i.^2 + p_j./rho_j.^2) .* dWdr_pair;

    % 粘性项（Morris 1997，增加 0.01h² 正则化）
    dvx_pair = vx(pair_i) - vx(pair_j);
    dvy_pair = vy(pair_i) - vy(pair_j);
    visc_common = mass * 2*mu ./ (rho_i.*rho_j) .* ...
                  (r_pair .* dWdr_pair) ./ (r_pair.^2 + 0.01*h^2);

    % 合力（对粒子 i 的贡献）
    fx = press_coef .* ex + visc_common .* dvx_pair;
    fy = press_coef .* ey + visc_common .* dvy_pair;

    % 对称累加（牛顿第三定律）
    ax = accumarray(pair_i, fx, [n_total,1]) - accumarray(pair_j, fx, [n_total,1]);
    ay = accumarray(pair_i, fy, [n_total,1]) - accumarray(pair_j, fy, [n_total,1]);

    % 体积力（仅流体粒子）
    ax(1:n_fluid) = ax(1:n_fluid) + gx;

    % XSPH 速度修正
    dvx_xsph = mass * (vx(pair_j)-vx(pair_i)) .* W_pair .* 2./(rho_i+rho_j);
    dvy_xsph = mass * (vy(pair_j)-vy(pair_i)) .* W_pair .* 2./(rho_i+rho_j);
    vx_xsph_out = accumarray(pair_i, dvx_xsph, [n_total,1]) + ...
                   accumarray(pair_j, -dvx_xsph, [n_total,1]);
    vy_xsph_out = accumarray(pair_i, dvy_xsph, [n_total,1]) + ...
                   accumarray(pair_j, -dvy_xsph, [n_total,1]);
end

function vx = update_wall_velocity(vx, vy, y, n_fluid, n_bottom, n_top, H)
% 镜像速度边界条件: 壁面虚粒子速度 = -u_fluid(y_mirror)

    y_fluid = y(1:n_fluid);
    vx_fluid = vx(1:n_fluid);

    % 下壁面
    idx_bottom = (n_fluid+1):(n_fluid+n_bottom);
    y_mirror_bottom = -y(idx_bottom);
    y_mirror_bottom = max(0, min(H, y_mirror_bottom));

    % 上壁面
    idx_top = (n_fluid+n_bottom+1):(n_fluid+n_bottom+n_top);
    y_mirror_top = 2*H - y(idx_top);
    y_mirror_top = max(0, min(H, y_mirror_top));

    % 用 interp1 插值获取镜像点速度
    [y_sorted, sort_idx] = sort(y_fluid);
    vx_sorted = vx_fluid(sort_idx);

    % 用唯一y值做插值（同一y层取平均）
    [y_unique, ~, ic] = unique(round(y_sorted, 10));
    vx_unique = accumarray(ic, vx_sorted, [], @mean);

    % 线性插值获取镜像点速度，然后取反
    vx_mirror_bottom = interp1(y_unique, vx_unique, y_mirror_bottom, 'linear', 'extrap');
    vx(idx_bottom) = -vx_mirror_bottom;

    vx_mirror_top = interp1(y_unique, vx_unique, y_mirror_top, 'linear', 'extrap');
    vx(idx_top) = -vx_mirror_top;
end

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

function vx = update_wall_velocity_fast(vx, n_fluid, n_bottom, n_top, ...
    fluid_layer_id, layer_particle_count, n_layers_fluid, ...
    wall_bottom_k1, wall_bottom_k2, wall_bottom_w1, wall_bottom_w2, ...
    wall_top_k1, wall_top_k2, wall_top_w1, wall_top_w2)
% 快速壁面速度更新（预计算映射表，无 sort/unique/interp1）

    vx_fluid = vx(1:n_fluid);
    vx_layer_sum = accumarray(fluid_layer_id, vx_fluid, [n_layers_fluid, 1]);
    vx_layer = vx_layer_sum ./ layer_particle_count;

    % 下壁面
    idx_bottom = (n_fluid+1):(n_fluid+n_bottom);
    vx_mirror = wall_bottom_w1 .* vx_layer(wall_bottom_k1) + ...
                wall_bottom_w2 .* vx_layer(wall_bottom_k2);
    vx(idx_bottom) = -vx_mirror;

    % 上壁面
    idx_top = (n_fluid+n_bottom+1):(n_fluid+n_bottom+n_top);
    vx_mirror = wall_top_w1 .* vx_layer(wall_top_k1) + ...
                wall_top_w2 .* vx_layer(wall_top_k2);
    vx(idx_top) = -vx_mirror;
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
