project_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_dir, 'build'));

pair_i = 1;
pair_j = 2;
dx = 0.0;
dy = 0.04;
r = 0.04;
W = 1.0;
dW = 10.0;

mass = [1.6e-3; 1.6e-3];
pos = [0.0, 0.02; 0.0, -0.02];
vel = zeros(2, 2);
wall_vel = zeros(2, 2);
rho = [1.0; 1.0];
drho_dt = [0.0; 0.0];

dt = 1.0e-3;
n_fluid = 1;
n_total = 2;
rho0 = 1.0;
p0 = 225.0;
c_f = 15.0;
mu = 0.1;
h = 0.052;
inv_sigma0 = 0.04^2;
body_force_x = 0.8 * mass(1);

[rho_out, p_out, pos_out, vel_out, drho_out, force_out, force_prior_out, Vol_out, B_out] = ...
    sph_physics_shell_mex('advance_shell_step', ...
        pair_i, pair_j, dx, dy, r, W, dW, ...
        mass, pos, vel, wall_vel, rho, drho_dt, ...
        dt, n_fluid, n_total, rho0, p0, c_f, mu, h, inv_sigma0, body_force_x);

assert(isequal(size(rho_out), [n_total, 1]), 'advance_shell_step rho 输出尺寸错误。');
assert(isequal(size(p_out), [n_total, 1]), 'advance_shell_step p 输出尺寸错误。');
assert(isequal(size(pos_out), [n_total, 2]), 'advance_shell_step pos 输出尺寸错误。');
assert(isequal(size(vel_out), [n_total, 2]), 'advance_shell_step vel 输出尺寸错误。');
assert(isequal(size(drho_out), [n_total, 1]), 'advance_shell_step drho 输出尺寸错误。');
assert(isequal(size(force_out), [n_total, 2]), 'advance_shell_step force 输出尺寸错误。');
assert(isequal(size(force_prior_out), [n_total, 2]), 'advance_shell_step force_prior 输出尺寸错误。');
assert(isequal(size(Vol_out), [n_total, 1]), 'advance_shell_step Vol 输出尺寸错误。');
assert(isequal(size(B_out), [n_total, 4]), 'advance_shell_step B 输出尺寸错误。');

[tau_bottom, tau_top] = sph_physics_shell_mex('wall_shear_monitor', ...
    pair_i, pair_j, dx, dy, r, dW, pos, vel, wall_vel, Vol_out, B_out, ...
    n_fluid, DL_value(), DH_value(), mu, h);

assert(isscalar(tau_bottom) && isfinite(tau_bottom), 'wall_shear_monitor 下壁面输出无效。');
assert(isscalar(tau_top) && isfinite(tau_top), 'wall_shear_monitor 上壁面输出无效。');

disp('test_sph_physics_modes: PASS');

function value = DL_value()
value = 3.0;
end

function value = DH_value()
value = 1.0;
end
