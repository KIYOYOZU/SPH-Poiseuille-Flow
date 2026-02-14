%% SPH_Poiseuille.m
% SPHinXsys 风格 2D Poiseuille 流动（MATLAB + 双 MEX）

clear; clc; close all;

fprintf('SPH Poiseuille Flow (SPHinXsys-aligned)\n');

project_dir = fileparts(mfilename('fullpath'));
if isempty(project_dir)
    project_dir = pwd;
end

build_dir = fullfile(project_dir, 'build');
mex_dir = fullfile(project_dir, 'mex');
config_path = fullfile(project_dir, 'config.ini');
restart_path = fullfile(project_dir, 'restart.mat');
result_png = fullfile(project_dir, 'SPH_Poiseuille_result.png');

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

DL = get_ini_numeric(cfg, 'physical', 'DL');
DH = get_ini_numeric(cfg, 'physical', 'DH');
dp = get_ini_numeric(cfg, 'physical', 'dp');
rho0 = get_ini_numeric(cfg, 'physical', 'rho0');
mu = get_ini_numeric(cfg, 'physical', 'mu');
U_f = get_ini_numeric(cfg, 'physical', 'U_f');
c_f = get_ini_numeric(cfg, 'physical', 'c_f');

t_end = get_ini_numeric(cfg, 'simulation', 'end_time');
output_interval = get_ini_numeric(cfg, 'simulation', 'output_interval');
sort_interval = round(get_ini_numeric(cfg, 'simulation', 'sort_interval'));
restart_from_file = round(get_ini_numeric(cfg, 'simulation', 'restart_from_file'));

gravity_g = 12.0 * mu * U_f / (rho0 * DH^2);
U_max = 1.5 * U_f;
h = 1.3 * dp;
BW = 4.0 * dp; %#ok<NASGU>
periodic_buffer = BW;
p0 = rho0 * c_f^2;
inv_sigma0 = dp^2;
nu = mu / rho0;

fprintf('参数: DL=%.3f, DH=%.3f, dp=%.4f, h=%.4f\n', DL, DH, dp, h);
fprintf('参数: rho0=%.3f, mu=%.3f, U_f=%.6f, c_f=%.3f\n', rho0, mu, U_f, c_f);
fprintf('派生: g=%.6f, Umax=%.6f, p0=%.6f\n', gravity_g, U_max, p0);
fprintf('仿真: end_time=%.3f, output_interval=%.3f, sort_interval=%d\n', t_end, output_interval, sort_interval);
fprintf('周期缓冲层: buffer=%.4f (%.2f*h)\n', periodic_buffer, periodic_buffer / h);

if sort_interval <= 0
    error('sort_interval 必须为正整数。');
end

%% S3: 粒子初始化
x_fluid = (dp/2 : dp : DL - dp/2)';
y_fluid = (dp/2 : dp : DH - dp/2)';
[X_fluid, Y_fluid] = meshgrid(x_fluid, y_fluid);
pos_fluid = [X_fluid(:), Y_fluid(:)];
n_fluid = size(pos_fluid, 1);

% 壁面粒子：仅在流体域 [0, DL] 范围内生成，避免周期性边界冲突
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
config_signature = create_config_signature(DL, DH, dp, rho0, mu, U_f, c_f, t_end, output_interval, sort_interval);
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
[pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij] = feval(neighbor_mex_name, pos, n_fluid, n_total, h, DL);
[rho, Vol, B] = feval(physics_mex_name, 'density_correction', ...
    pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij, ...
    mass, n_fluid, n_total, rho0, h, inv_sigma0);
p(1:n_fluid) = p0 * (rho(1:n_fluid) ./ rho0 - 1.0);
p(n_fluid+1:end) = 0.0;

%% S6: 主循环（双层时间步进）
while t < t_end - 1e-12
    target_time = min(t + output_interval, t_end);

    while t < target_time - 1e-12
        step = step + 1;

        Dt = advection_time_step(vel(1:n_fluid, :), force(1:n_fluid, :), force_prior(1:n_fluid, :), ...
            mass(1:n_fluid), h, mu, rho0, U_f);
        Dt = min(Dt, target_time - t);
        Dt = min(Dt, t_end - t);

        [rho, Vol, B] = feval(physics_mex_name, 'density_correction', ...
            pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij, ...
            mass, n_fluid, n_total, rho0, h, inv_sigma0);

        viscous_force = feval(physics_mex_name, 'viscous_force', ...
            pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
            vel, Vol, B, mu, h, n_fluid, n_total, mass, wall_vel);
        force_prior = viscous_force;
        force_prior(1:n_fluid, 1) = force_prior(1:n_fluid, 1) + mass(1:n_fluid) * gravity_g;

        pos = feval(physics_mex_name, 'transport_correction', ...
            pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
            Vol, B, pos, h, n_fluid, n_total);

        pos = bounding_from_wall(pos, n_fluid, DH, dp);

        relaxation_time = 0.0;
        while relaxation_time < Dt - 1e-12
            dt = acoustic_time_step(vel(1:n_fluid, :), c_f, h, Dt - relaxation_time);

            [rho, p, pos, force, drho_dt] = feval(physics_mex_name, 'integration_1st', ...
                pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
                Vol, B, rho, mass, pos, vel, drho_dt, force_prior, dt, ...
                n_fluid, n_total, rho0, p0, c_f, wall_vel);

            acc_fluid = (force_prior(1:n_fluid, :) + force(1:n_fluid, :)) ./ mass(1:n_fluid);
            vel(1:n_fluid, :) = vel(1:n_fluid, :) + acc_fluid * dt;

            [pos, drho_dt, ~] = feval(physics_mex_name, 'integration_2nd', ...
                pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
                Vol, rho, pos, vel, dt, n_fluid, n_total, wall_vel);
            rho(1:n_fluid) = rho(1:n_fluid) + drho_dt(1:n_fluid) * (dt / 2.0);

            [pos, wrapped] = periodic_bounding(pos, n_fluid, DL, periodic_buffer);
            if wrapped
                [pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij] = feval(neighbor_mex_name, pos, n_fluid, n_total, h, DL);
            end

            % BoundingFromWall: SPHinXsys 风格壁面防穿透
            pos = bounding_from_wall(pos, n_fluid, DH, dp);

            relaxation_time = relaxation_time + dt;
        end

        [pos, ~] = periodic_bounding(pos, n_fluid, DL, periodic_buffer);
        pos = bounding_from_wall(pos, n_fluid, DH, dp);
        vel(n_fluid+1:end, :) = wall_vel(n_fluid+1:end, :);

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
        'step', step);
    save_restart(restart_path, config_signature, state);
end

%% S7: 后处理与验证
fluid_pos = pos(1:n_fluid, :);
fluid_pos(:, 1) = mod(fluid_pos(:, 1), DL);
fluid_vel = vel(1:n_fluid, :);

n_bins = max(20, round(DH / dp));
[y_mid, u_mean] = compute_binned_profile_mean(fluid_pos(:, 2), fluid_vel(:, 1), 0.0, DH, n_bins);
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

fig = figure('Color', 'w', 'Position', [120, 120, 1200, 500]);
subplot(1, 2, 1);
plot(u_exact, y_mid, 'k-', 'LineWidth', 2.0); hold on;
plot(u_mean, y_mid, 'ro', 'MarkerSize', 4.5, 'LineWidth', 1.0);
grid on;
xlabel('u_x');
ylabel('y');
title(sprintf('Velocity Profile (L2=%.3f%%)', 100.0 * L2_error));
legend({'Analytical', 'SPH'}, 'Location', 'best');

subplot(1, 2, 2);
scatter(fluid_pos(:, 1), fluid_pos(:, 2), 9, fluid_vel(:, 1), 'filled'); hold on;
wall_pos_plot = pos(n_fluid+1:end, :);
wall_pos_plot(:, 1) = mod(wall_pos_plot(:, 1), DL);
scatter(wall_pos_plot(:, 1), wall_pos_plot(:, 2), 8, [0.25, 0.25, 0.25], 'filled');
axis equal;
xlim([0, DL]);
ylim([-4 * dp, DH + 4 * dp]);
grid on;
xlabel('x');
ylabel('y');
title('Particle Distribution & u_x');
colormap(turbo);
caxis([0, 1.2]);  % 设置颜色范围为 0 到 1.2
cb = colorbar;
cb.Label.String = 'u_x';

saveas(fig, result_png);
fprintf('结果图已保存: %s\n', result_png);

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

function sig = create_config_signature(DL, DH, dp, rho0, mu, U_f, c_f, t_end, output_interval, sort_interval)
    sig = sprintf('DL=%.12g|DH=%.12g|dp=%.12g|rho0=%.12g|mu=%.12g|Uf=%.12g|cf=%.12g|t=%.12g|oi=%.12g|si=%d', ...
        DL, DH, dp, rho0, mu, U_f, c_f, t_end, output_interval, sort_interval);
end

function Dt = advection_time_step(vel, force, force_prior, mass, h, mu, rho0, U_f)
    viscous_speed = mu / (rho0 * h);
    speed_ref = max(viscous_speed, U_f);

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
    signal = c_f + vecnorm(vel, 2, 2);
    max_signal = max(signal);
    max_signal = max(max_signal, 1e-12);
    dt = min(0.6 * h / max_signal, remain);
    dt = max(dt, 1e-12);
end

function [pos, vel, rho, mass, wall_vel, drho_dt, force_prior, force, p, Vol, B] = sort_particles_by_cell( ...
    pos, vel, rho, mass, wall_vel, drho_dt, force_prior, force, p, Vol, B, n_fluid, DL, h)

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
    edges = linspace(y_min, y_max, n_bins + 1);
    y_mid = 0.5 * (edges(1:end-1) + edges(2:end));
    y_mid = y_mid(:);
    bin_id = discretize(y_values, edges);

    sum_u = accumarray(bin_id(~isnan(bin_id)), u_values(~isnan(bin_id)), [n_bins, 1], @sum, 0);
    cnt_u = accumarray(bin_id(~isnan(bin_id)), 1, [n_bins, 1], @sum, 0);
    u_mean = sum_u ./ max(cnt_u, 1);
    u_mean(cnt_u == 0) = NaN;
end

function save_restart(restart_path, config_signature, state)
    save(restart_path, 'state', 'config_signature', '-v7.3');
end

function pos = bounding_from_wall(pos, n_fluid, DH, dp)
% BoundingFromWall (SPHinXsys 风格)
% 粒子到壁面距离 < 0.25*dp 时，弹回到 0.5*dp 处
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
