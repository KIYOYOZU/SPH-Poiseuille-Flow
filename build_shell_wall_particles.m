function [pos_wall, wall_normal, wall_measure, wall_thickness_arr] = build_shell_wall_particles(DL, DH, dp, wall_thickness)
%BUILD_SHELL_WALL_PARTICLES 生成上下单层 shell 壁面粒子及其几何元数据。
    if nargin < 4
        error('build_shell_wall_particles 需要输入 DL, DH, dp, wall_thickness。');
    end
    if DL <= 0 || DH <= 0 || dp <= 0 || wall_thickness <= 0
        error('DL, DH, dp 和 wall_thickness 必须为正数。');
    end

    x_wall = (dp/2 : dp : DL - dp/2)';
    nx = numel(x_wall);

    pos_bottom = [x_wall, -0.5 * dp * ones(nx, 1)];
    pos_top = [x_wall, (DH + 0.5 * dp) * ones(nx, 1)];
    pos_wall = [pos_bottom; pos_top];

    wall_normal = [repmat([0.0, -1.0], nx, 1); repmat([0.0, 1.0], nx, 1)];
    wall_measure = dp * ones(2 * nx, 1);
    wall_thickness_arr = wall_thickness * ones(2 * nx, 1);
end
