function wall_patch = build_wall_patch_metadata(pos_wall, wall_normal, wall_measure, DL, DH, patch_count_x)
%BUILD_WALL_PATCH_METADATA 构建上下壁 patch 元数据。
    if nargin < 6
        error('build_wall_patch_metadata 需要 pos_wall, wall_normal, wall_measure, DL, DH, patch_count_x。');
    end
    if patch_count_x < 1 || abs(patch_count_x - round(patch_count_x)) > 0
        error('patch_count_x 必须为正整数。');
    end

    n_wall = size(pos_wall, 1);
    if size(wall_normal, 1) ~= n_wall || numel(wall_measure) ~= n_wall
        error('壁面几何数组尺寸不一致。');
    end

    patch_count_x = round(patch_count_x);
    patch_count_total = 2 * patch_count_x;
    patch_id = zeros(n_wall, 1);
    patch_ix = zeros(n_wall, 1);
    patch_side = zeros(n_wall, 1);
    wall_local_indices_by_patch = cell(patch_count_total, 1);
    surface_local_indices_by_patch = cell(patch_count_total, 1);
    patch_length = zeros(patch_count_total, 1);

    x_wrapped = mod(pos_wall(:, 1), DL);
    x_wrapped(x_wrapped < 0.0) = x_wrapped(x_wrapped < 0.0) + DL;
    x_wrapped(x_wrapped >= DL) = x_wrapped(x_wrapped >= DL) - eps(DL);
    patch_width = DL / patch_count_x;

    bottom_mask = wall_normal(:, 2) < 0.0;
    top_mask = wall_normal(:, 2) > 0.0;
    if ~any(bottom_mask) || ~any(top_mask)
        error('必须同时存在上下壁粒子。');
    end

    bottom_surface_y = max(pos_wall(bottom_mask, 2));
    top_surface_y = min(pos_wall(top_mask, 2));
    tol = 1.0e-12;
    surface_mask = (bottom_mask & abs(pos_wall(:, 2) - bottom_surface_y) < tol) | ...
        (top_mask & abs(pos_wall(:, 2) - top_surface_y) < tol);

    for i = 1:n_wall
        ix = floor(x_wrapped(i) / patch_width);
        ix = min(max(ix, 0), patch_count_x - 1);
        side = wall_normal(i, 2) > 0.0;
        pid = side * patch_count_x + ix + 1;
        patch_id(i) = pid;
        patch_ix(i) = ix + 1;
        patch_side(i) = side;
        wall_local_indices_by_patch{pid}(end + 1, 1) = i;
        if surface_mask(i)
            surface_local_indices_by_patch{pid}(end + 1, 1) = i;
            patch_length(pid) = patch_length(pid) + wall_measure(i);
        end
    end

    for pid = 1:patch_count_total
        if isempty(wall_local_indices_by_patch{pid})
            error('存在空 patch，请降低 patch_count_x。');
        end
        if isempty(surface_local_indices_by_patch{pid}) || patch_length(pid) <= 0.0
            error('patch 表层识别失败，请检查厚壁几何。');
        end
    end

    wall_patch = struct( ...
        'patch_count_x', patch_count_x, ...
        'patch_count_total', patch_count_total, ...
        'patch_width', patch_width, ...
        'patch_id', patch_id, ...
        'patch_ix', patch_ix, ...
        'patch_side', patch_side, ...
        'surface_mask', surface_mask, ...
        'patch_length', patch_length, ...
        'wall_local_indices_by_patch', {wall_local_indices_by_patch}, ...
        'surface_local_indices_by_patch', {surface_local_indices_by_patch});
end
