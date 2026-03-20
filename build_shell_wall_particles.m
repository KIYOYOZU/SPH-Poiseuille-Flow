function [pos_wall, wall_normal, wall_measure, wall_thickness_arr] = build_shell_wall_particles(DL, DH, dp, wall_thickness)
%BUILD_SHELL_WALL_PARTICLES 生成上下厚壁粒子区及其几何元数据。
% wall_measure .* wall_thickness_arr 给出壁粒子的有效体积，
% 便于 MATLAB 主脚本与 MEX 算子共享同一套壁面离散几何。
% 关键修复背景（3eb6de0）：这里保留 BW=4dp 厚壁配置，
% 避免回退到单层 shell 后再次出现壁面支撑不足的问题。
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
    % 法向数组按“下壁块、上壁块”顺序展开，便于后续按粒子块追踪。
    wall_normal = [repmat([0.0, -1.0], n_bottom, 1); repmat([0.0, 1.0], n_top, 1)];
    wall_measure = dp * ones(n_bottom + n_top, 1);
    wall_thickness_arr = dp * ones(n_bottom + n_top, 1);
end
