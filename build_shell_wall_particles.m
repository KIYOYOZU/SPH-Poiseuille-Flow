function [pos_wall, wall_normal, wall_measure, wall_thickness_arr] = build_shell_wall_particles(DL, DH, dp, wall_thickness)
%BUILD_SHELL_WALL_PARTICLES 生成上下厚壁粒子区及其几何元数据。
    if nargin < 4
        error('build_shell_wall_particles 需要输入 DL, DH, dp, wall_thickness。');
    end
    if DL <= 0 || DH <= 0 || dp <= 0 || wall_thickness <= 0
        error('DL, DH, dp 和 wall_thickness 必须为正数。');
    end

    n_layers = round(wall_thickness / dp);
    if abs(n_layers * dp - wall_thickness) > 1e-12
        error('wall_thickness 必须是 dp 的整数倍。');
    end
    if n_layers < 1
        error('wall_thickness 至少需要一层粒子。');
    end

    x_wall = (dp / 2 : dp : DL - dp / 2)';
    y_bottom = ((-wall_thickness + dp / 2) : dp : (-dp / 2))';
    y_top = ((DH + dp / 2) : dp : (DH + wall_thickness - dp / 2))';

    [X_bottom, Y_bottom] = meshgrid(x_wall, y_bottom);
    [X_top, Y_top] = meshgrid(x_wall, y_top);

    pos_bottom = [X_bottom(:), Y_bottom(:)];
    pos_top = [X_top(:), Y_top(:)];
    pos_wall = [pos_bottom; pos_top];

    n_bottom = size(pos_bottom, 1);
    n_top = size(pos_top, 1);
    wall_normal = [repmat([0.0, -1.0], n_bottom, 1); repmat([0.0, 1.0], n_top, 1)];
    wall_measure = dp * ones(n_bottom + n_top, 1);
    wall_thickness_arr = dp * ones(n_bottom + n_top, 1);
end
