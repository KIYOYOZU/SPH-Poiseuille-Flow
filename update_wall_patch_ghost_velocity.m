function [wall_vel, patch_state] = update_wall_patch_ghost_velocity( ...
    neighbor, vel, Vol, B, wall_vel, wall_patch, n_fluid, mu, h, tau_target, main_flow_sign)
%UPDATE_WALL_PATCH_GHOST_VELOCITY 按 patch 离散配平重构表层壁粒子 ghost velocity。
    patch_count_total = wall_patch.patch_count_total;
    sum_surface_K = zeros(patch_count_total, 1);
    sum_surface_Ku = zeros(patch_count_total, 1);
    sum_outer_force = zeros(patch_count_total, 1);
    ghost_u = zeros(patch_count_total, 1);
    fallback_flag = zeros(patch_count_total, 1);

    if main_flow_sign >= 0.0
        main_flow_sign = 1.0;
    else
        main_flow_sign = -1.0;
    end

    wall_vel(n_fluid + 1:end, :) = 0.0;
    for k = 1:numel(neighbor.pair_i)
        ii = neighbor.pair_i(k);
        jj = neighbor.pair_j(k);
        rk = neighbor.r_ij(k);
        if ii < 1 || ii > n_fluid || jj <= n_fluid || rk <= 1.0e-12
            continue;
        end

        local_j = jj - n_fluid;
        pid = wall_patch.patch_id(local_j);
        ex = neighbor.dx_ij(k) / rk;
        ey = neighbor.dy_ij(k) / rk;
        eBe = ex * (B(ii, 1) * ex + B(ii, 2) * ey) + ...
            ey * (B(ii, 3) * ex + B(ii, 4) * ey);
        coeff = 4.0 * mu * eBe * neighbor.dW_ij(k) * Vol(jj) * Vol(ii) / (rk + 0.01 * h);
        if wall_patch.surface_mask(local_j)
            sum_surface_K(pid) = sum_surface_K(pid) + coeff;
            sum_surface_Ku(pid) = sum_surface_Ku(pid) + coeff * vel(ii, 1);
        else
            sum_outer_force(pid) = sum_outer_force(pid) + coeff * vel(ii, 1);
        end
    end

    for pid = 1:patch_count_total
        target_force_x = -main_flow_sign * tau_target * wall_patch.patch_length(pid);
        if abs(sum_surface_K(pid)) > 1.0e-12
            ghost_u(pid) = (sum_surface_Ku(pid) + sum_outer_force(pid) - target_force_x) / sum_surface_K(pid);
        else
            ghost_u(pid) = 0.0;
            fallback_flag(pid) = 1;
        end
        surface_global_indices = n_fluid + wall_patch.surface_local_indices_by_patch{pid};
        wall_vel(surface_global_indices, 1) = ghost_u(pid);
    end

    patch_state = struct( ...
        'sum_surface_K', sum_surface_K, ...
        'sum_surface_Ku', sum_surface_Ku, ...
        'sum_outer_force', sum_outer_force, ...
        'ghost_u', ghost_u, ...
        'fallback_flag', fallback_flag);
end
