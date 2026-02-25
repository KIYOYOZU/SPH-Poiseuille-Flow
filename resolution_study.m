%% resolution_study.m
% 粒子分辨率对壁面切应力施加精度的影响研究
% 以 dp 为循环变量，扫描 14 个分辨率点，统计稳态 L2 误差和切应力误差
% 不调用、不修改 SPH_Poiseuille.m

clear; clc; close all;

project_dir = fileparts(mfilename('fullpath'));
if isempty(project_dir), project_dir = pwd; end

build_dir = fullfile(project_dir, 'build');
mex_dir   = fullfile(project_dir, 'mex');
if ~exist(build_dir, 'dir'), mkdir(build_dir); end
addpath(build_dir);

neighbor_mex_name = 'sph_neighbor_search_mex';
physics_mex_name  = 'sph_physics_mex';
rs_ensure_mex(fullfile(mex_dir,'sph_neighbor_search_mex.c'), neighbor_mex_name, build_dir);
rs_ensure_mex(fullfile(mex_dir,'sph_physics_mex.c'),         physics_mex_name,  build_dir);

%% 固定物理参数（与 config.ini 一致）
DL_base  = 3.0;
DH_base  = 1.0;
rho0     = 1.0;
mu       = 0.1;
U_bulk   = 6.66667;
c_f      = 10.0;
end_time = 10.0;

%% dp 扫描列表
dp_coarse = 0.10 : -0.01 : 0.01;
dp_list   = dp_coarse;
N_dp      = numel(dp_list);

L2_err   = nan(1, N_dp);
tau_err  = nan(1, N_dp);

fprintf('开始分辨率扫描，共 %d 个 dp 点\n', N_dp);

for k = 1:N_dp
    dp = dp_list(k);
    fprintf('\n=== [%d/%d] dp = %.4f ===\n', k, N_dp, dp);

    % 几何对齐
    DL = round(DL_base / dp) * dp;
    DH = round(DH_base / dp) * dp;

    % 派生参数
    gravity_g  = 12.0 * mu * U_bulk / (rho0 * DH^2);
    U_max      = 1.5 * U_bulk;
    h          = 1.3 * dp;
    BW         = 4.0 * dp;
    periodic_buffer = BW;
    p0         = rho0 * c_f^2;
    inv_sigma0 = dp^2;
    nu         = mu / rho0;
    tau_target = gravity_g * rho0 * DH / 2;

    % 粒子初始化
    x_fluid = (dp/2 : dp : DL - dp/2)';
    y_fluid = (dp/2 : dp : DH - dp/2)';
    [X_fluid, Y_fluid] = meshgrid(x_fluid, y_fluid);
    pos_fluid = [X_fluid(:), Y_fluid(:)];
    n_fluid   = size(pos_fluid, 1);

    x_wall = (dp/2 : dp : DL - dp/2)';
    x_bottom=[]; y_bottom=[]; x_top=[]; y_top=[];
    for layer = 1:4
        yb = -((2*layer-1)*dp/2);
        yt =  DH + ((2*layer-1)*dp/2);
        x_bottom = [x_bottom; x_wall]; %#ok<AGROW>
        y_bottom = [y_bottom; yb*ones(size(x_wall))]; %#ok<AGROW>
        x_top    = [x_top;    x_wall]; %#ok<AGROW>
        y_top    = [y_top;    yt*ones(size(x_wall))]; %#ok<AGROW>
    end
    pos_wall = [x_bottom, y_bottom; x_top, y_top];
    n_wall   = size(pos_wall, 1);
    n_total  = n_fluid + n_wall;

    pos          = [pos_fluid; pos_wall];
    vel          = zeros(n_total, 2);
    wall_vel     = zeros(n_total, 2);
    rho          = rho0 * ones(n_total, 1);
    p            = zeros(n_total, 1);
    drho_dt      = zeros(n_total, 1);
    force        = zeros(n_total, 2);
    force_prior  = zeros(n_total, 2);
    mass_fluid   = rho0 * dp^2;
    mass_wall    = rho0 * dp^2;
    mass         = [mass_fluid*ones(n_fluid,1); mass_wall*ones(n_wall,1)];
    Vol          = mass ./ rho;
    B            = zeros(n_total, 4);
    B(:,1)       = 1.0;
    B(:,4)       = 1.0;

    fprintf('  粒子数: fluid=%d, wall=%d\n', n_fluid, n_wall);

    % 初始邻居搜索
    [pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij] = ...
        feval(neighbor_mex_name, pos, n_fluid, n_total, h, DL);
    [rho, Vol, B] = feval(physics_mex_name, 'density_correction', ...
        pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij, ...
        mass, n_fluid, n_total, rho0, h, inv_sigma0);
    p(1:n_fluid)       = p0 * (rho(1:n_fluid) ./ rho0 - 1.0);
    p(n_fluid+1:end)   = 0.0;

    % PI 控制器初始化
    I_bottom = 0.0; I_top = 0.0; I_max = 2.0;
    tau_num_bottom = 0.0; tau_num_top = 0.0;
    k_p = 0.5; k_i = 0.025;

    % 稳态统计累加器
    tau_bot_acc = 0.0; tau_top_acc = 0.0; t_acc = 0.0;

    t    = 0.0;
    step = 0;
    sort_interval = 100;

    %% 主循环
    while t < end_time - 1e-12
        step = step + 1;

        Dt = rs_advection_dt(vel(1:n_fluid,:), force(1:n_fluid,:), ...
            force_prior(1:n_fluid,:), mass(1:n_fluid), h, mu, rho0, U_bulk);
        Dt = min(Dt, end_time - t);
        if Dt < 1e-14
            fprintf('  警告: Dt 退化，跳过 dp=%.4f\n', dp);
            break;
        end

        [rho, Vol, B] = feval(physics_mex_name, 'density_correction', ...
            pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij, ...
            mass, n_fluid, n_total, rho0, h, inv_sigma0);

        viscous_force = feval(physics_mex_name, 'viscous_force', ...
            pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
            vel, Vol, B, mu, h, n_fluid, n_total, mass, wall_vel);
        force_prior = viscous_force;
        force_prior(1:n_fluid,1) = force_prior(1:n_fluid,1) + mass(1:n_fluid)*gravity_g;

        % === 方案 D+B：pair-wise tau 测量 + PI 输出 delta_tau 叠加 force_prior ===
        % 下壁面切应力
        is_bottom = (pair_j > n_fluid) & (pos(pair_j,2) < 0);
        i_b = pair_i(is_bottom); j_b = pair_j(is_bottom);
        if ~isempty(i_b)
            dx_b=dx_ij(is_bottom); dy_b=dy_ij(is_bottom);
            r_b=r_ij(is_bottom);   dW_b=dW_ij(is_bottom);
            ex_b=dx_b./r_b; ey_b=dy_b./r_b;
            eBe_b = ex_b.*(B(i_b,1).*ex_b+B(i_b,2).*ey_b) + ...
                    ey_b.*(B(i_b,3).*ex_b+B(i_b,4).*ey_b);
            dv_x_b = vel(i_b,1) - wall_vel(j_b,1);
            f_pair_b = 4*mu.*eBe_b.*dW_b.*Vol(j_b).*dv_x_b./(r_b+0.01*h).*Vol(i_b);
            tau_num_bottom = -sum(f_pair_b)/DL;
        else
            tau_num_bottom = 0.0;
        end

        % 上壁面切应力
        is_top = (pair_j > n_fluid) & (pos(pair_j,2) > DH);
        i_t = pair_i(is_top); j_t = pair_j(is_top);
        if ~isempty(i_t)
            dx_t=dx_ij(is_top); dy_t=dy_ij(is_top);
            r_t=r_ij(is_top);   dW_t=dW_ij(is_top);
            ex_t=dx_t./r_t; ey_t=dy_t./r_t;
            eBe_t = ex_t.*(B(i_t,1).*ex_t+B(i_t,2).*ey_t) + ...
                    ey_t.*(B(i_t,3).*ex_t+B(i_t,4).*ey_t);
            dv_x_t = vel(i_t,1) - wall_vel(j_t,1);
            f_pair_t = 4*mu.*eBe_t.*dW_t.*Vol(j_t).*dv_x_t./(r_t+0.01*h).*Vol(i_t);
            tau_num_top = -sum(f_pair_t)/DL;
        else
            tau_num_top = 0.0;
        end

        % PI 更新：下壁面 → delta_tau_b → 叠加 force_prior
        e_b = tau_target - tau_num_bottom;
        I_bottom = max(-I_max, min(I_max, I_bottom + e_b*Dt));
        delta_tau_b = k_p*e_b + k_i*I_bottom;
        near_bottom = pos(1:n_fluid,2) < 2*h;
        force_prior(near_bottom,1) = force_prior(near_bottom,1) + delta_tau_b*dp^2;

        % PI 更新：上壁面 → delta_tau_t → 叠加 force_prior
        e_t = tau_target - tau_num_top;
        I_top = max(-I_max, min(I_max, I_top + e_t*Dt));
        delta_tau_t = k_p*e_t + k_i*I_top;
        near_top = pos(1:n_fluid,2) > DH - 2*h;
        force_prior(near_top,1) = force_prior(near_top,1) + delta_tau_t*dp^2;
        % === 方案 D+B 结束 ===

        % 传输速度修正
        pos = feval(physics_mex_name, 'transport_correction', ...
            pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
            Vol, B, pos, h, n_fluid, n_total);
        pos = rs_bounding(pos, n_fluid, DH, dp);

        % 声学子步
        relax = 0.0;
        while relax < Dt - 1e-12
            dt = rs_acoustic_dt(vel(1:n_fluid,:), c_f, h, Dt-relax);
            [rho, p, pos, force, drho_dt] = feval(physics_mex_name, 'integration_1st', ...
                pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
                Vol, B, rho, mass, pos, vel, drho_dt, force_prior, dt, ...
                n_fluid, n_total, rho0, p0, c_f, wall_vel);
            acc_f = (force_prior(1:n_fluid,:)+force(1:n_fluid,:))./mass(1:n_fluid);
            vel(1:n_fluid,:) = vel(1:n_fluid,:) + acc_f*dt;
            [pos, drho_dt, ~] = feval(physics_mex_name, 'integration_2nd', ...
                pair_i, pair_j, dx_ij, dy_ij, r_ij, dW_ij, ...
                Vol, rho, pos, vel, dt, n_fluid, n_total, wall_vel);
            rho(1:n_fluid) = rho(1:n_fluid) + drho_dt(1:n_fluid)*(dt/2.0);
            [pos, wrapped] = rs_periodic(pos, n_fluid, DL, periodic_buffer);
            if wrapped
                [pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij] = ...
                    feval(neighbor_mex_name, pos, n_fluid, n_total, h, DL);
            end
            pos   = rs_bounding(pos, n_fluid, DH, dp);
            relax = relax + dt;
        end

        [pos, ~] = rs_periodic(pos, n_fluid, DL, periodic_buffer);
        pos = rs_bounding(pos, n_fluid, DH, dp);
        vel(n_fluid+1:end,:) = wall_vel(n_fluid+1:end,:);

        if mod(step, sort_interval)==0 && step~=1
            [pos,vel,rho,mass,wall_vel,drho_dt,force_prior,force,p,Vol,B] = ...
                rs_sort(pos,vel,rho,mass,wall_vel,drho_dt,force_prior,force,p,Vol,B,n_fluid,DL,h);
        end
        [pair_i, pair_j, dx_ij, dy_ij, r_ij, W_ij, dW_ij] = ...
            feval(neighbor_mex_name, pos, n_fluid, n_total, h, DL);

        t = t + Dt;

        % 稳态统计（t >= 0.8*end_time）
        if t >= 0.8 * end_time
            tau_bot_acc = tau_bot_acc + tau_num_bottom * Dt;
            tau_top_acc = tau_top_acc + tau_num_top    * Dt;
            t_acc       = t_acc + Dt;
        end

        if mod(step, 50)==0
            fprintf('  step=%d t=%.4f/%.1f tau_bot=%.4f tau_top=%.4f\n', ...
                step, t, end_time, tau_num_bottom, tau_num_top);
        end
    end % 主循环

    % 稳态切应力
    if t_acc > 0
        tau_bot_steady = tau_bot_acc / t_acc;
        tau_top_steady = tau_top_acc / t_acc;
    else
        tau_bot_steady = tau_num_bottom;
        tau_top_steady = tau_num_top;
    end
    tau_steady = (tau_bot_steady + tau_top_steady) / 2;
    tau_err(k) = abs(tau_steady - tau_target) / tau_target;

    % L2 误差
    fluid_pos = pos(1:n_fluid,:);
    fluid_vel = vel(1:n_fluid,:);
    n_bins = max(20, round(DH/dp));
    [y_mid, u_mean] = rs_binned_profile(fluid_pos(:,2), fluid_vel(:,1), 0.0, DH, n_bins);

    valid = ~isnan(u_mean);
    if any(valid)
        u_exact_rel = gravity_g/(2.0*nu) .* y_mid(valid) .* (DH - y_mid(valid));
        L2_err(k) = sqrt(sum((u_mean(valid) - u_exact_rel).^2) / max(sum(u_exact_rel.^2), eps));
    end

    fprintf('  dp=%.4f  L2=%.4f%%  tau_err=%.4f%%\n', dp, 100*L2_err(k), 100*tau_err(k));
end % dp 循环

%% 保存结果
save_path = fullfile(project_dir, 'resolution_results.mat');
save(save_path, 'dp_list', 'L2_err', 'tau_err');
fprintf('\n结果已保存: %s\n', save_path);

%% 双对数收敛曲线
valid_L2  = ~isnan(L2_err)  & L2_err  > 0;
valid_tau = ~isnan(tau_err) & tau_err > 0;

fig = figure('Color','w','Position',[100,100,1000,440]);

ax1 = subplot(1,2,1);
loglog(ax1, dp_list(valid_L2), L2_err(valid_L2), 'o-', ...
    'Color',[0.2 0.4 0.8],'LineWidth',1.5,'MarkerFaceColor',[0.2 0.4 0.8],'MarkerSize',6);
hold(ax1,'on');
if sum(valid_L2) >= 2
    cf_L2 = polyfit(log(dp_list(valid_L2)), log(L2_err(valid_L2)), 1);
    dp_fit = linspace(min(dp_list(valid_L2)), max(dp_list(valid_L2)), 50);
    loglog(ax1, dp_fit, exp(polyval(cf_L2, log(dp_fit))), '--', ...
        'Color',[0.5 0.5 0.5],'LineWidth',1.2);
    legend(ax1, {'SPH', sprintf('slope=%.2f', cf_L2(1))}, ...
        'Location','northwest','Box','off','FontSize',11);
    fprintf('L2 收敛阶数: %.3f\n', cf_L2(1));
end
hold(ax1,'off');
xlabel(ax1,'$dp$ (m)','Interpreter','latex','FontSize',13);
ylabel(ax1,'$L_2$ relative error','Interpreter','latex','FontSize',13);
title(ax1,'(a) Velocity profile convergence','FontName','Times New Roman','FontWeight','normal');
set(ax1,'FontName','Times New Roman','FontSize',12,'TickDir','in','Box','on');
% 手动设置 Y 轴刻度，确保数值清晰可读
yticks_L2 = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10];
set(ax1, 'YTick', yticks_L2, 'YTickLabel', arrayfun(@(v) sprintf('%.1f%%', v*100), yticks_L2, 'UniformOutput', false));
ylim(ax1, [0.004, 0.12]);

ax2 = subplot(1,2,2);
loglog(ax2, dp_list(valid_tau), tau_err(valid_tau), 's-', ...
    'Color',[0.8 0.2 0.2],'LineWidth',1.5,'MarkerFaceColor',[0.8 0.2 0.2],'MarkerSize',6);
hold(ax2,'on');
if sum(valid_tau) >= 2
    cf_tau = polyfit(log(dp_list(valid_tau)), log(tau_err(valid_tau)), 1);
    dp_fit2 = linspace(min(dp_list(valid_tau)), max(dp_list(valid_tau)), 50);
    loglog(ax2, dp_fit2, exp(polyval(cf_tau, log(dp_fit2))), '--', ...
        'Color',[0.5 0.5 0.5],'LineWidth',1.2);
    legend(ax2, {'SPH', sprintf('slope=%.2f', cf_tau(1))}, ...
        'Location','northwest','Box','off','FontSize',11);
    fprintf('tau 收敛阶数: %.3f\n', cf_tau(1));
end
hold(ax2,'off');
xlabel(ax2,'$dp$ (m)','Interpreter','latex','FontSize',13);
ylabel(ax2,'$\tau$ relative error','Interpreter','latex','FontSize',13);
title(ax2,'(b) Wall shear stress convergence','FontName','Times New Roman','FontWeight','normal');
set(ax2,'FontName','Times New Roman','FontSize',12,'TickDir','in','Box','on');

saveas(fig, fullfile(project_dir, 'resolution_convergence.png'));
fprintf('收敛曲线已保存: resolution_convergence.png\n');

%% ===== 局部函数 =====
function rs_ensure_mex(src, name, bdir)
    out = fullfile(bdir, [name,'.',mexext]);
    if exist(out,'file')
        si = dir(src); bi = dir(out);
        if bi.datenum >= si.datenum, return; end
    end
    fprintf('编译 MEX: %s\n', name);
    if ispc
        mex('-R2018a','-O','COMPFLAGS="$COMPFLAGS /openmp"', ...
            '-output',name,'-outdir',bdir,src);
    else
        mex('-R2018a','-O','CFLAGS="$CFLAGS -fopenmp"', ...
            'LDFLAGS="$LDFLAGS -fopenmp"','-output',name,'-outdir',bdir,src);
    end
end

function Dt = rs_advection_dt(vel, force, force_prior, mass, h, mu, rho0, U_bulk)
    viscous_speed = mu/(rho0*h);
    speed_ref = max(viscous_speed, U_bulk);
    v_max = max(vecnorm(vel,2,2));
    acc_vec = (force+force_prior)./mass;
    acc_max = max(4.0*h*vecnorm(acc_vec,2,2));
    speed_max = max(sqrt(max(v_max^2, acc_max)), speed_ref);
    speed_max = max(speed_max, 1e-12);
    Dt = 0.25*h/speed_max;
end

function dt = rs_acoustic_dt(vel, c_f, h, remain)
    sig = c_f + vecnorm(vel,2,2);
    dt  = min(0.6*h/max(max(sig),1e-12), remain);
    dt  = max(dt, 1e-12);
end

function [pos, wrapped] = rs_periodic(pos, n_fluid, DL, buf)
    x = pos(1:n_fluid,1); wrapped = false;
    lm = x < -buf;
    while any(lm), x(lm)=x(lm)+DL; wrapped=true; lm=x<-buf; end
    rm = x >= DL+buf;
    while any(rm), x(rm)=x(rm)-DL; wrapped=true; rm=x>=DL+buf; end
    pos(1:n_fluid,1) = x;
end

function pos = rs_bounding(pos, n_fluid, DH, dp)
    y = pos(1:n_fluid,2);
    y(y < 0.25*dp) = 0.5*dp;
    y(DH-y < 0.25*dp) = DH - 0.5*dp;
    pos(1:n_fluid,2) = y;
end

function [y_mid, u_mean] = rs_binned_profile(y_vals, u_vals, ymin, ymax, nb)
    edges = linspace(ymin, ymax, nb+1);
    y_mid = 0.5*(edges(1:end-1)+edges(2:end))';
    bid   = discretize(y_vals, edges);
    ok    = ~isnan(bid);
    su    = accumarray(bid(ok), u_vals(ok), [nb,1], @sum, 0);
    cnt   = accumarray(bid(ok), 1,          [nb,1], @sum, 0);
    u_mean = su ./ max(cnt,1);
    u_mean(cnt==0) = NaN;
end

function [pos,vel,rho,mass,wall_vel,drho_dt,fp,force,p,Vol,B] = ...
    rs_sort(pos,vel,rho,mass,wall_vel,drho_dt,fp,force,p,Vol,B,n_fluid,DL,h)
    n_total = size(pos,1);
    cell_sz = 2.0*h;
    cx = floor(mod(pos(1:n_fluid,1),DL)/cell_sz);
    y0 = min(pos(1:n_fluid,2));
    cy = floor((pos(1:n_fluid,2)-y0)/cell_sz);
    [~,idx_f] = sortrows([cy,cx],[1,2]);
    idx = [idx_f; (n_fluid+1:n_total)'];
    pos=pos(idx,:); vel=vel(idx,:); rho=rho(idx,:); mass=mass(idx,:);
    wall_vel=wall_vel(idx,:); drho_dt=drho_dt(idx,:);
    fp=fp(idx,:); force=force(idx,:); p=p(idx,:); Vol=Vol(idx,:); B=B(idx,:);
end
