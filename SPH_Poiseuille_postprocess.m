function post_result = SPH_Poiseuille_postprocess(postprocess_mat_path)
%SPH_POISEUILLE_POSTPROCESS 读取后处理 mat，生成泊肃叶流验证图与误差指标。
    if nargin < 1 || isempty(postprocess_mat_path)
        project_dir = fileparts(mfilename('fullpath'));
        postprocess_mat_path = fullfile(project_dir, 'results', 'SPH_Poiseuille_postprocess.mat');
    end

    if ~exist(postprocess_mat_path, 'file')
        error('后处理数据文件不存在: %s', postprocess_mat_path);
    end

    loaded = load(postprocess_mat_path, 'postprocess_data');
    if ~isfield(loaded, 'postprocess_data')
        error('后处理数据文件缺少 postprocess_data: %s', postprocess_mat_path);
    end

    post = loaded.postprocess_data;
    validate_postprocess_input(post);

    cfg = post.cfg;
    n_fluid = post.geom.n_fluid;
    state = post.state;
    monitor = post.monitor;
    final_profile = get_final_profile(post);
    output = post.output;
    ensure_parent_dir(output.result_png);
    ensure_parent_dir(output.profile_evolution_png);

    fluid_pos = state.pos(1:n_fluid, :);
    fluid_pos(:, 1) = mod(fluid_pos(:, 1), cfg.DL);
    fluid_vel = state.vel(1:n_fluid, :);

    y_mid = final_profile.y_mid;
    u_mean = final_profile.u_mean;
    u_exact = final_profile.u_exact;

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

    y_norm = y_mid / cfg.DH;
    u_norm_exact = u_exact / cfg.U_max;
    u_norm_sph = u_mean / cfg.U_max;

    fig = figure('Color', 'w', 'Position', [100, 100, 1400, 520], 'Renderer', 'painters');
    ax1 = subplot(1, 2, 1);
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
    text(ax1, 0.05, 0.95, sprintf('$L_2 = %.2f\\%%$', 100 * L2_error), ...
        'Units', 'normalized', 'Interpreter', 'latex', 'FontSize', 12, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
        'FontName', 'Times New Roman');
    title(ax1, '(a) Velocity profile', 'FontName', 'Times New Roman', ...
        'FontSize', 13, 'FontWeight', 'normal');

    ax2 = subplot(1, 2, 2);
    [xg, yg, Ug] = build_velocity_field_grid(fluid_pos, fluid_vel(:, 1), cfg);
    draw_velocity_field(ax2, xg, yg, Ug, cfg);
    title(ax2, '(b) Velocity field', 'FontName', 'Times New Roman', ...
        'FontSize', 13, 'FontWeight', 'normal');

    saveas(fig, output.result_png);
    fprintf('结果图已保存: %s\n', output.result_png);
    close(fig);

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

    saveas(fig_evo, output.profile_evolution_png);
    fprintf('中间截面剖面演化图已保存: %s\n', output.profile_evolution_png);
    close(fig_evo);

    post_result = struct( ...
        'L2_error', L2_error, ...
        'result_png', output.result_png, ...
        'profile_evolution_png', output.profile_evolution_png);
end

function validate_postprocess_input(post)
    required_fields = {'cfg', 'geom', 'state', 'monitor', 'output'};
    for k = 1:numel(required_fields)
        if ~isfield(post, required_fields{k})
            error('后处理数据缺少字段: %s', required_fields{k});
        end
    end

    required_cfg = {'DL', 'DH', 'dp', 'U_max', 'gravity_g', 'nu', 'h', 'wall_thickness'};
    require_struct_fields(post.cfg, required_cfg, 'cfg');
    require_struct_fields(post.geom, {'n_fluid'}, 'geom');
    require_struct_fields(post.state, {'pos', 'vel'}, 'state');
    require_struct_fields(post.monitor, {'n_bins', 'profile_times', 'mid_profile_u'}, 'monitor');
    require_struct_fields(post.output, {'result_png', 'profile_evolution_png'}, 'output');
end

function require_struct_fields(data, required_fields, struct_name)
    for k = 1:numel(required_fields)
        if ~isfield(data, required_fields{k})
            error('后处理数据缺少字段: %s.%s', struct_name, required_fields{k});
        end
    end
end

function final_profile = get_final_profile(post)
    if isfield(post, 'final_profile')
        require_struct_fields(post.final_profile, {'y_mid', 'u_mean', 'u_exact'}, 'final_profile');
        final_profile = post.final_profile;
        return;
    end

    cfg = post.cfg;
    n_fluid = post.geom.n_fluid;
    fluid_pos = post.state.pos(1:n_fluid, :);
    fluid_pos(:, 1) = mod(fluid_pos(:, 1), cfg.DL);
    fluid_vel_x = post.state.vel(1:n_fluid, 1);
    [y_mid, u_mean] = compute_binned_profile_mean(fluid_pos(:, 2), fluid_vel_x, 0.0, cfg.DH, post.monitor.n_bins);
    final_profile = struct( ...
        'y_mid', y_mid, ...
        'u_mean', u_mean, ...
        'u_exact', cfg.gravity_g / (2.0 * cfg.nu) .* y_mid .* (cfg.DH - y_mid));
end

function ensure_parent_dir(file_path)
    parent_dir = fileparts(file_path);
    if ~isempty(parent_dir) && ~exist(parent_dir, 'dir')
        mkdir(parent_dir);
    end
end

function [xg, yg, Ug] = build_velocity_field_grid(fluid_pos, u_x, cfg)
    nx_grid = round(cfg.DL / cfg.dp) * 2;
    ny_grid = round(cfg.DH / cfg.dp) * 2;
    xg = linspace(0, cfg.DL, nx_grid);
    yg = linspace(0, cfg.DH, ny_grid);
    [Xg, Yg] = meshgrid(xg, yg);

    fp_x = mod(fluid_pos(:, 1), cfg.DL);
    fp_y = fluid_pos(:, 2);
    right_mask = fp_x > cfg.DL - 2 * cfg.h;
    left_mask = fp_x < 2 * cfg.h;
    fp_x_ext = [fp_x; fp_x(right_mask) - cfg.DL; fp_x(left_mask) + cfg.DL];
    fp_y_ext = [fp_y; fp_y(right_mask); fp_y(left_mask)];
    u_x_ext = [u_x; u_x(right_mask); u_x(left_mask)];

    interpolant = scatteredInterpolant(fp_x_ext, fp_y_ext, u_x_ext, 'natural', 'nearest');
    Ug = interpolant(Xg, Yg);
end

function draw_velocity_field(ax, xg, yg, Ug, cfg)
    wall_thick = cfg.wall_thickness;
    y_lo = -wall_thick;
    y_hi = cfg.DH + wall_thick;

    fill(ax, [0 cfg.DL cfg.DL 0], [y_lo y_lo 0 0], [0.75 0.75 0.75], ...
        'EdgeColor', 'none'); hold(ax, 'on');
    fill(ax, [0 cfg.DL cfg.DL 0], [cfg.DH cfg.DH y_hi y_hi], [0.75 0.75 0.75], ...
        'EdgeColor', 'none');
    plot(ax, [0 cfg.DL], [0 0], 'k-', 'LineWidth', 1.2);
    plot(ax, [0 cfg.DL], [cfg.DH cfg.DH], 'k-', 'LineWidth', 1.2);

    imagesc(ax, xg, yg, Ug);
    set(ax, 'YDir', 'normal');

    fill(ax, [0 cfg.DL cfg.DL 0], [y_lo y_lo 0 0], [0.75 0.75 0.75], 'EdgeColor', 'none');
    fill(ax, [0 cfg.DL cfg.DL 0], [cfg.DH cfg.DH y_hi y_hi], [0.75 0.75 0.75], 'EdgeColor', 'none');
    plot(ax, [0 cfg.DL], [0 0], 'k-', 'LineWidth', 1.2);
    plot(ax, [0 cfg.DL], [cfg.DH cfg.DH], 'k-', 'LineWidth', 1.2);
    text(ax, cfg.DL / 2, y_lo / 2, 'Wall', 'HorizontalAlignment', 'center', ...
        'FontName', 'Times New Roman', 'FontSize', 11, 'Color', [0.3 0.3 0.3]);
    text(ax, cfg.DL / 2, cfg.DH + wall_thick / 2, 'Wall', 'HorizontalAlignment', 'center', ...
        'FontName', 'Times New Roman', 'FontSize', 11, 'Color', [0.3 0.3 0.3]);
    hold(ax, 'off');

    axis(ax, 'equal');
    xlim(ax, [0, cfg.DL]);
    ylim(ax, [y_lo, y_hi]);
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 13, 'LineWidth', 1.0, ...
        'TickDir', 'in', 'TickLength', [0.015 0.015], 'Box', 'on');
    xlabel(ax, '$x$ (m)', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel(ax, '$y$ (m)', 'Interpreter', 'latex', 'FontSize', 14);
    colormap(ax, turbo);
    caxis(ax, [0, cfg.U_max * 1.1]);
    cb = colorbar(ax);
    cb.Label.String = '$u_x$ (m/s)';
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize = 13;
    set(cb, 'FontName', 'Times New Roman', 'FontSize', 11, ...
        'TickDirection', 'in', 'LineWidth', 1.0);
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
