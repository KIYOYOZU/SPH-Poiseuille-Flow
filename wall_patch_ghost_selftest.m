clear; clc;

project_dir = fileparts(mfilename('fullpath'));
if isempty(project_dir)
    project_dir = pwd;
end
build_dir = fullfile(project_dir, 'build');
if ~exist(build_dir, 'dir')
    mkdir(build_dir);
end
addpath(project_dir);
addpath(build_dir);

ensure_selftest_mex_compiled(fullfile(project_dir, 'mex', 'sph_physics_mex.c'), 'sph_physics_shell_mex', build_dir);

DL = 4.0;
DH = 1.0;
dp = 1.0;
wall_thickness = 2.0;
patch_count_x = 2;
[pos_wall, wall_normal, wall_measure, ~] = build_shell_wall_particles(DL, DH, dp, wall_thickness);
wall_patch = build_wall_patch_metadata(pos_wall, wall_normal, wall_measure, DL, DH, patch_count_x);

assert(numel(wall_patch.patch_length) == 2 * patch_count_x, 'patch 数量错误。');
assert(all(abs(wall_patch.patch_length - 2.0) < 1.0e-12), 'patch 物理长度错误。');
assert(isequal(unique(wall_patch.patch_id(wall_normal(:, 2) < 0.0))', [1, 2]), '下壁 patch 编号错误。');
assert(isequal(unique(wall_patch.patch_id(wall_normal(:, 2) > 0.0))', [3, 4]), '上壁 patch 编号错误。');

n_wall = size(pos_wall, 1);
n_fluid = 1;
n_total = n_fluid + n_wall;
contact_wall_local = wall_patch.surface_local_indices_by_patch{1}(1);
neighbor = struct( ...
    'pair_i', 1, ...
    'pair_j', n_fluid + contact_wall_local, ...
    'dx_ij', 1.0, ...
    'dy_ij', 0.0, ...
    'r_ij', 1.0, ...
    'dW_ij', 1.0);
vel = zeros(n_total, 2);
vel(1, 1) = 1.0;
Vol = ones(n_total, 1);
B = zeros(n_total, 4);
B(:, 1) = 1.0;
B(:, 4) = 1.0;
wall_vel = zeros(n_total, 2);
tau_target = 0.25;
[wall_vel_out, patch_state] = update_wall_patch_ghost_velocity( ...
    neighbor, vel, Vol, B, wall_vel, wall_patch, n_fluid, 1.0, 1.0, tau_target, 1.0);
expected_sum_k = 4.0 / 1.01;
expected_u_g = (expected_sum_k - (-tau_target * wall_patch.patch_length(1))) / expected_sum_k;
assert(abs(patch_state.sum_surface_K(1) - expected_sum_k) < 1.0e-12, 'sum_surface_K 重构错误。');
assert(abs(patch_state.sum_outer_force(1)) < 1.0e-12, 'sum_outer_force 对单层接触场景应为零。');
assert(abs(patch_state.ghost_u(1) - expected_u_g) < 1.0e-12, 'patch ghost velocity 重构错误。');
surface_patch1_global = n_fluid + wall_patch.surface_local_indices_by_patch{1};
assert(all(abs(wall_vel_out(surface_patch1_global, 1) - expected_u_g) < 1.0e-12), ...
    'patch ghost velocity 未广播到 patch 表层壁粒子。');
assert(all(abs(wall_vel_out(surface_patch1_global, 2)) < 1.0e-12), '壁面 ghost velocity 的 y 分量应为零。');
non_surface_local = setdiff(wall_patch.wall_local_indices_by_patch{1}, wall_patch.surface_local_indices_by_patch{1});
if ~isempty(non_surface_local)
    non_surface_global = n_fluid + non_surface_local;
    assert(all(abs(wall_vel_out(non_surface_global, :)) < 1.0e-12, 'all'), ...
        '非表层厚壁粒子在当前原型里应保持静止。');
end

disp('wall_patch_ghost_selftest: PASS');
clear mex;

function ensure_selftest_mex_compiled(src_path, out_name, build_dir)
    out_bin = fullfile(build_dir, [out_name, '.', mexext]);
    need_compile = exist(out_bin, 'file') ~= 3;
    if ~need_compile
        src_info = dir(src_path);
        bin_info = dir(out_bin);
        need_compile = src_info.datenum > bin_info.datenum;
    end
    if ~need_compile
        return;
    end

    if ispc
        mex('-R2018a', '-O', 'COMPFLAGS="$COMPFLAGS /openmp"', ...
            '-output', out_name, '-outdir', build_dir, src_path);
    else
        mex('-R2018a', '-O', ...
            'CFLAGS="$CFLAGS -fopenmp"', ...
            'LDFLAGS="$LDFLAGS -fopenmp"', ...
            '-output', out_name, '-outdir', build_dir, src_path);
    end
end


