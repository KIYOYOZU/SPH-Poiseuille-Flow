/*
 * sph_step_mex.c
 * Fused MEX C + OpenMP accelerated SPH single-step computation
 * (neighbor search + density/pressure/forces/XSPH)
 *
 * Build:
 *   mex -R2018a -O COMPFLAGS="$COMPFLAGS /openmp" sph_step_mex.c
 *
 * Usage:
 *   [ax, ay, rho, p, vx_xsph, vy_xsph, n_pairs] = sph_step_mex(
 *       x, y, vx, vy, n_total, n_fluid, L, r_cut,
 *       y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, n_cells,
 *       h, alpha_kernel, W_self, mass, cs, rho0, mu, gx, do_shepard);
 */

#include "mex.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static void counting_sort(const int *cell_id, int n, int n_cells,
                          int *sorted_particles, int *cell_start, int *cell_end)
{
    int *count;
    int *offset;
    int i, cid;

    count = (int *)mxCalloc(n_cells, sizeof(int));

    for (i = 0; i < n; i++)
        count[cell_id[i]]++;

    cell_start[0] = 0;
    for (i = 1; i < n_cells; i++)
        cell_start[i] = cell_start[i - 1] + count[i - 1];

    for (i = 0; i < n_cells; i++) {
        if (count[i] > 0)
            cell_end[i] = cell_start[i] + count[i] - 1;
        else {
            cell_start[i] = -1;
            cell_end[i] = -2;
        }
    }

    offset = (int *)mxCalloc(n_cells, sizeof(int));
    for (i = 0; i < n_cells; i++)
        offset[i] = (cell_start[i] >= 0) ? cell_start[i] : 0;

    for (i = 0; i < n; i++) {
        cid = cell_id[i];
        sorted_particles[offset[cid]] = i;
        offset[cid]++;
    }

    mxFree(count);
    mxFree(offset);
}

static void free_thread_pair_buffers(int n_threads,
                                     int **t_pair_i,
                                     int **t_pair_j,
                                     double **t_dx,
                                     double **t_dy,
                                     double **t_r,
                                     int *t_count,
                                     int *t_cap)
{
    int t;

    if (t_pair_i) {
        for (t = 0; t < n_threads; t++) {
            if (t_pair_i[t]) free(t_pair_i[t]);
        }
        free(t_pair_i);
    }
    if (t_pair_j) {
        for (t = 0; t < n_threads; t++) {
            if (t_pair_j[t]) free(t_pair_j[t]);
        }
        free(t_pair_j);
    }
    if (t_dx) {
        for (t = 0; t < n_threads; t++) {
            if (t_dx[t]) free(t_dx[t]);
        }
        free(t_dx);
    }
    if (t_dy) {
        for (t = 0; t < n_threads; t++) {
            if (t_dy[t]) free(t_dy[t]);
        }
        free(t_dy);
    }
    if (t_r) {
        for (t = 0; t < n_threads; t++) {
            if (t_r[t]) free(t_r[t]);
        }
        free(t_r);
    }
    if (t_count) free(t_count);
    if (t_cap) free(t_cap);
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    const double *x;
    const double *y;
    const double *vx;
    const double *vy;
    int n_total;
    int n_fluid;
    double L;
    double r_cut;
    double y_min_domain;
    int n_cell_x;
    int n_cell_y;
    double cell_size_x;
    double cell_size_y;
    int n_cells;
    double h;
    double alpha_kernel;
    double W_self;
    double mass;
    double cs;
    double rho0;
    double mu;
    double gx;
    int do_shepard;

    double h_inv;
    double h_sq_001;
    double cs_sq;
    double r_cut_sq;

    int *cell_id;
    int *sorted_particles;
    int *cell_start;
    int *cell_end_arr;

    int n_threads;
    int max_pairs_per_thread;
    int **t_pair_i;
    int **t_pair_j;
    double **t_dx;
    double **t_dy;
    double **t_r;
    int *t_count;
    int *t_cap;

    int total_pairs;
    int *pair_i;
    int *pair_j;
    double *dx_pair;
    double *dy_pair;
    double *r_pair;
    double *W_pair;
    double *dWdr_pair;

    double *ax_out;
    double *ay_out;
    double *rho_out;
    double *p_out;
    double *vx_xsph_out;
    double *vy_xsph_out;

    static const int stencil_dx[5] = {0, 1, -1, 0, 1};
    static const int stencil_dy[5] = {0, 0, 1, 1, 1};

    if (nrhs != 23)
        mexErrMsgIdAndTxt("SPH:nrhs", "Requires 23 input arguments");
    if (nlhs != 7)
        mexErrMsgIdAndTxt("SPH:nlhs", "Requires 7 output arguments");

    x = mxGetDoubles(prhs[0]);
    y = mxGetDoubles(prhs[1]);
    vx = mxGetDoubles(prhs[2]);
    vy = mxGetDoubles(prhs[3]);
    n_total = (int)mxGetScalar(prhs[4]);
    n_fluid = (int)mxGetScalar(prhs[5]);
    L = mxGetScalar(prhs[6]);
    r_cut = mxGetScalar(prhs[7]);
    y_min_domain = mxGetScalar(prhs[8]);
    n_cell_x = (int)mxGetScalar(prhs[9]);
    n_cell_y = (int)mxGetScalar(prhs[10]);
    cell_size_x = mxGetScalar(prhs[11]);
    cell_size_y = mxGetScalar(prhs[12]);
    n_cells = (int)mxGetScalar(prhs[13]);
    h = mxGetScalar(prhs[14]);
    alpha_kernel = mxGetScalar(prhs[15]);
    W_self = mxGetScalar(prhs[16]);
    mass = mxGetScalar(prhs[17]);
    cs = mxGetScalar(prhs[18]);
    rho0 = mxGetScalar(prhs[19]);
    mu = mxGetScalar(prhs[20]);
    gx = mxGetScalar(prhs[21]);
    do_shepard = (int)mxGetScalar(prhs[22]);

    h_inv = 1.0 / h;
    h_sq_001 = 0.01 * h * h;
    cs_sq = cs * cs;
    r_cut_sq = r_cut * r_cut;

    plhs[0] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[5] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[6] = mxCreateDoubleScalar(0.0);

    ax_out = mxGetDoubles(plhs[0]);
    ay_out = mxGetDoubles(plhs[1]);
    rho_out = mxGetDoubles(plhs[2]);
    p_out = mxGetDoubles(plhs[3]);
    vx_xsph_out = mxGetDoubles(plhs[4]);
    vy_xsph_out = mxGetDoubles(plhs[5]);

    memset(ax_out, 0, n_total * sizeof(double));
    memset(ay_out, 0, n_total * sizeof(double));
    memset(rho_out, 0, n_total * sizeof(double));
    memset(vx_xsph_out, 0, n_total * sizeof(double));
    memset(vy_xsph_out, 0, n_total * sizeof(double));

    cell_id = (int *)mxMalloc(n_total * sizeof(int));
    sorted_particles = (int *)mxMalloc(n_total * sizeof(int));
    cell_start = (int *)mxMalloc(n_cells * sizeof(int));
    cell_end_arr = (int *)mxMalloc(n_cells * sizeof(int));

    pair_i = NULL;
    pair_j = NULL;
    dx_pair = NULL;
    dy_pair = NULL;
    r_pair = NULL;
    W_pair = NULL;
    dWdr_pair = NULL;
    t_pair_i = NULL;
    t_pair_j = NULL;
    t_dx = NULL;
    t_dy = NULL;
    t_r = NULL;
    t_count = NULL;
    t_cap = NULL;

    {
        int i;
        for (i = 0; i < n_total; i++) {
            double xmod = fmod(x[i], L);
            int cx;
            int cy;
            if (xmod < 0.0) xmod += L;
            cx = (int)floor(xmod / cell_size_x);
            cy = (int)floor((y[i] - y_min_domain) / cell_size_y);
            if (cx < 0) cx = 0;
            if (cx >= n_cell_x) cx = n_cell_x - 1;
            if (cy < 0) cy = 0;
            if (cy >= n_cell_y) cy = n_cell_y - 1;
            cell_id[i] = cy * n_cell_x + cx;
        }
    }

    counting_sort(cell_id, n_total, n_cells, sorted_particles, cell_start, cell_end_arr);

    n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    max_pairs_per_thread = n_total * 30 / n_threads + 1024;
    t_pair_i = (int **)calloc(n_threads, sizeof(int *));
    t_pair_j = (int **)calloc(n_threads, sizeof(int *));
    t_dx = (double **)calloc(n_threads, sizeof(double *));
    t_dy = (double **)calloc(n_threads, sizeof(double *));
    t_r = (double **)calloc(n_threads, sizeof(double *));
    t_count = (int *)calloc(n_threads, sizeof(int));
    t_cap = (int *)calloc(n_threads, sizeof(int));

    {
        int t;
        for (t = 0; t < n_threads; t++) {
            t_cap[t] = max_pairs_per_thread;
            t_pair_i[t] = (int *)malloc((size_t)t_cap[t] * sizeof(int));
            t_pair_j[t] = (int *)malloc((size_t)t_cap[t] * sizeof(int));
            t_dx[t] = (double *)malloc((size_t)t_cap[t] * sizeof(double));
            t_dy[t] = (double *)malloc((size_t)t_cap[t] * sizeof(double));
            t_r[t] = (double *)malloc((size_t)t_cap[t] * sizeof(double));
        }
    }

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int tid = 0;
        int local_count;
        int local_cap;
        int *lp_i;
        int *lp_j;
        double *ld_x;
        double *ld_y;
        double *ld_r;
        int cy;

#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        local_count = 0;
        local_cap = t_cap[tid];
        lp_i = t_pair_i[tid];
        lp_j = t_pair_j[tid];
        ld_x = t_dx[tid];
        ld_y = t_dy[tid];
        ld_r = t_r[tid];

#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 1)
#endif
        for (cy = 0; cy < n_cell_y; cy++) {
            int cx;
            for (cx = 0; cx < n_cell_x; cx++) {
                int cid1 = cy * n_cell_x + cx;
                int s;
                if (cell_start[cid1] < 0) continue;

                for (s = 0; s < 5; s++) {
                    int ncx = cx + stencil_dx[s];
                    int ncy = cy + stencil_dy[s];
                    double x_shift = 0.0;
                    int cid2;
                    int ii;

                    if (ncy < 0 || ncy >= n_cell_y) continue;

                    if (ncx < 0) {
                        ncx += n_cell_x;
                        x_shift = L;
                    } else if (ncx >= n_cell_x) {
                        ncx -= n_cell_x;
                        x_shift = -L;
                    }

                    cid2 = ncy * n_cell_x + ncx;
                    if (cell_start[cid2] < 0) continue;

                    for (ii = cell_start[cid1]; ii <= cell_end_arr[cid1]; ii++) {
                        int i = sorted_particles[ii];
                        double xi = x[i];
                        double yi = y[i];
                        int jj_start = (cid1 == cid2) ? (ii + 1) : cell_start[cid2];
                        int jj;

                        for (jj = jj_start; jj <= cell_end_arr[cid2]; jj++) {
                            int j = sorted_particles[jj];
                            double dxx = xi - x[j] + x_shift;
                            double dyy = yi - y[j];
                            double rr_sq = dxx * dxx + dyy * dyy;

                            if (rr_sq < r_cut_sq && rr_sq > 0.0) {
                                if (local_count >= local_cap) {
                                    local_cap *= 2;
                                    lp_i = (int *)realloc(lp_i, (size_t)local_cap * sizeof(int));
                                    lp_j = (int *)realloc(lp_j, (size_t)local_cap * sizeof(int));
                                    ld_x = (double *)realloc(ld_x, (size_t)local_cap * sizeof(double));
                                    ld_y = (double *)realloc(ld_y, (size_t)local_cap * sizeof(double));
                                    ld_r = (double *)realloc(ld_r, (size_t)local_cap * sizeof(double));
                                }

                                lp_i[local_count] = i;
                                lp_j[local_count] = j;
                                ld_x[local_count] = dxx;
                                ld_y[local_count] = dyy;
                                ld_r[local_count] = sqrt(rr_sq);
                                local_count++;
                            }
                        }
                    }
                }
            }
        }

        t_pair_i[tid] = lp_i;
        t_pair_j[tid] = lp_j;
        t_dx[tid] = ld_x;
        t_dy[tid] = ld_y;
        t_r[tid] = ld_r;
        t_count[tid] = local_count;
        t_cap[tid] = local_cap;
    }

    total_pairs = 0;
    {
        int t;
        for (t = 0; t < n_threads; t++)
            total_pairs += t_count[t];
    }

    mxGetDoubles(plhs[6])[0] = (double)total_pairs;

    if (total_pairs == 0) {
        int i;
        for (i = 0; i < n_total; i++) {
            rho_out[i] = mass * W_self;
            p_out[i] = cs_sq * (rho_out[i] - rho0);
        }
        for (i = 0; i < n_fluid; i++)
            ax_out[i] += gx;

        mxFree(cell_id);
        mxFree(sorted_particles);
        mxFree(cell_start);
        mxFree(cell_end_arr);
        free_thread_pair_buffers(n_threads, t_pair_i, t_pair_j, t_dx, t_dy, t_r, t_count, t_cap);
        return;
    }

    pair_i = (int *)mxMalloc((size_t)total_pairs * sizeof(int));
    pair_j = (int *)mxMalloc((size_t)total_pairs * sizeof(int));
    dx_pair = (double *)mxMalloc((size_t)total_pairs * sizeof(double));
    dy_pair = (double *)mxMalloc((size_t)total_pairs * sizeof(double));
    r_pair = (double *)mxMalloc((size_t)total_pairs * sizeof(double));

    {
        int off = 0;
        int t;
        for (t = 0; t < n_threads; t++) {
            int k;
            for (k = 0; k < t_count[t]; k++) {
                pair_i[off] = t_pair_i[t][k];
                pair_j[off] = t_pair_j[t][k];
                dx_pair[off] = t_dx[t][k];
                dy_pair[off] = t_dy[t][k];
                r_pair[off] = t_r[t][k];
                off++;
            }
        }
    }

    mxFree(cell_id);
    mxFree(sorted_particles);
    mxFree(cell_start);
    mxFree(cell_end_arr);
    free_thread_pair_buffers(n_threads, t_pair_i, t_pair_j, t_dx, t_dy, t_r, t_count, t_cap);

    W_pair = (double *)mxMalloc((size_t)total_pairs * sizeof(double));
    dWdr_pair = (double *)mxMalloc((size_t)total_pairs * sizeof(double));

    {
        int k;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (k = 0; k < total_pairs; k++) {
            double q = r_pair[k] * h_inv;
            if (q < 1.0) {
                W_pair[k] = alpha_kernel * (2.0 / 3.0 - q * q + 0.5 * q * q * q);
                dWdr_pair[k] = alpha_kernel * (-2.0 * q + 1.5 * q * q) * h_inv;
            } else if (q < 2.0) {
                double tmp = 2.0 - q;
                W_pair[k] = alpha_kernel * (1.0 / 6.0) * tmp * tmp * tmp;
                dWdr_pair[k] = alpha_kernel * (-0.5) * tmp * tmp * h_inv;
            } else {
                W_pair[k] = 0.0;
                dWdr_pair[k] = 0.0;
            }
        }
    }

    if (n_threads > 1) {
        double **rho_priv = (double **)malloc((size_t)n_threads * sizeof(double *));
        int t;

        for (t = 0; t < n_threads; t++)
            rho_priv[t] = (double *)calloc((size_t)n_total, sizeof(double));

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            int tid = 0;
            double *my_rho;
            int k;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            my_rho = rho_priv[tid];

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (k = 0; k < total_pairs; k++) {
                my_rho[pair_i[k]] += W_pair[k];
                my_rho[pair_j[k]] += W_pair[k];
            }
        }

        {
            int i;
            for (i = 0; i < n_total; i++) {
                double sum = 0.0;
                for (t = 0; t < n_threads; t++)
                    sum += rho_priv[t][i];
                rho_out[i] = mass * (sum + W_self);
            }
        }

        for (t = 0; t < n_threads; t++)
            free(rho_priv[t]);
        free(rho_priv);
    } else {
        int k;
        for (k = 0; k < total_pairs; k++) {
            rho_out[pair_i[k]] += W_pair[k];
            rho_out[pair_j[k]] += W_pair[k];
        }
        {
            int i;
            for (i = 0; i < n_total; i++)
                rho_out[i] = mass * (rho_out[i] + W_self);
        }
    }

    if (do_shepard) {
        double *shepard = (double *)mxCalloc((size_t)n_total, sizeof(double));
        int k;
        int i;

        for (k = 0; k < total_pairs; k++) {
            int ii = pair_i[k];
            int jj = pair_j[k];
            double vol_j_W = mass * W_pair[k] / rho_out[jj];
            double vol_i_W = mass * W_pair[k] / rho_out[ii];
            shepard[ii] += vol_j_W;
            shepard[jj] += vol_i_W;
        }

        for (i = 0; i < n_total; i++) {
            shepard[i] += mass * W_self / rho_out[i];
            if (shepard[i] < 0.5) shepard[i] = 0.5;
            if (shepard[i] > 1.5) shepard[i] = 1.5;
            rho_out[i] /= shepard[i];
        }

        mxFree(shepard);
    }

    {
        int i;
        for (i = 0; i < n_total; i++)
            p_out[i] = cs_sq * (rho_out[i] - rho0);
    }

    if (n_threads > 1) {
        double **ax_priv = (double **)malloc((size_t)n_threads * sizeof(double *));
        double **ay_priv = (double **)malloc((size_t)n_threads * sizeof(double *));
        double **vx_xsph_priv = (double **)malloc((size_t)n_threads * sizeof(double *));
        double **vy_xsph_priv = (double **)malloc((size_t)n_threads * sizeof(double *));
        int t;

        for (t = 0; t < n_threads; t++) {
            ax_priv[t] = (double *)calloc((size_t)n_total, sizeof(double));
            ay_priv[t] = (double *)calloc((size_t)n_total, sizeof(double));
            vx_xsph_priv[t] = (double *)calloc((size_t)n_total, sizeof(double));
            vy_xsph_priv[t] = (double *)calloc((size_t)n_total, sizeof(double));
        }

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            int tid = 0;
            double *my_ax;
            double *my_ay;
            double *my_vxx;
            double *my_vyy;
            int k;

#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            my_ax = ax_priv[tid];
            my_ay = ay_priv[tid];
            my_vxx = vx_xsph_priv[tid];
            my_vyy = vy_xsph_priv[tid];

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (k = 0; k < total_pairs; k++) {
                int ii = pair_i[k];
                int jj = pair_j[k];
                double rho_i = rho_out[ii];
                double rho_j = rho_out[jj];
                double p_i = p_out[ii];
                double p_j = p_out[jj];
                double r_safe = r_pair[k];
                double ex;
                double ey;
                double press_coef;
                double dvx;
                double dvy;
                double visc_common;
                double fx;
                double fy;
                double rho_avg_inv;
                double dvx_x;
                double dvy_x;

                if (r_safe < 1e-12) r_safe = 1e-12;
                ex = dx_pair[k] / r_safe;
                ey = dy_pair[k] / r_safe;

                press_coef = -mass * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * dWdr_pair[k];

                dvx = vx[ii] - vx[jj];
                dvy = vy[ii] - vy[jj];
                visc_common = mass * 2.0 * mu / (rho_i * rho_j) *
                    (r_pair[k] * dWdr_pair[k]) / (r_pair[k] * r_pair[k] + h_sq_001);

                fx = press_coef * ex + visc_common * dvx;
                fy = press_coef * ey + visc_common * dvy;

                my_ax[ii] += fx;
                my_ay[ii] += fy;
                my_ax[jj] -= fx;
                my_ay[jj] -= fy;

                rho_avg_inv = 2.0 / (rho_i + rho_j);
                dvx_x = mass * (vx[jj] - vx[ii]) * W_pair[k] * rho_avg_inv;
                dvy_x = mass * (vy[jj] - vy[ii]) * W_pair[k] * rho_avg_inv;
                my_vxx[ii] += dvx_x;
                my_vyy[ii] += dvy_x;
                my_vxx[jj] -= dvx_x;
                my_vyy[jj] -= dvy_x;
            }
        }

        {
            int i;
            for (i = 0; i < n_total; i++) {
                for (t = 0; t < n_threads; t++) {
                    ax_out[i] += ax_priv[t][i];
                    ay_out[i] += ay_priv[t][i];
                    vx_xsph_out[i] += vx_xsph_priv[t][i];
                    vy_xsph_out[i] += vy_xsph_priv[t][i];
                }
            }
        }

        for (t = 0; t < n_threads; t++) {
            free(ax_priv[t]);
            free(ay_priv[t]);
            free(vx_xsph_priv[t]);
            free(vy_xsph_priv[t]);
        }
        free(ax_priv);
        free(ay_priv);
        free(vx_xsph_priv);
        free(vy_xsph_priv);
    } else {
        int k;
        for (k = 0; k < total_pairs; k++) {
            int ii = pair_i[k];
            int jj = pair_j[k];
            double rho_i = rho_out[ii];
            double rho_j = rho_out[jj];
            double p_i = p_out[ii];
            double p_j = p_out[jj];
            double r_safe = r_pair[k];
            double ex;
            double ey;
            double press_coef;
            double dvx;
            double dvy;
            double visc_common;
            double fx;
            double fy;
            double rho_avg_inv;
            double dvx_x;
            double dvy_x;

            if (r_safe < 1e-12) r_safe = 1e-12;
            ex = dx_pair[k] / r_safe;
            ey = dy_pair[k] / r_safe;

            press_coef = -mass * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * dWdr_pair[k];

            dvx = vx[ii] - vx[jj];
            dvy = vy[ii] - vy[jj];
            visc_common = mass * 2.0 * mu / (rho_i * rho_j) *
                (r_pair[k] * dWdr_pair[k]) / (r_pair[k] * r_pair[k] + h_sq_001);

            fx = press_coef * ex + visc_common * dvx;
            fy = press_coef * ey + visc_common * dvy;

            ax_out[ii] += fx;
            ay_out[ii] += fy;
            ax_out[jj] -= fx;
            ay_out[jj] -= fy;

            rho_avg_inv = 2.0 / (rho_i + rho_j);
            dvx_x = mass * (vx[jj] - vx[ii]) * W_pair[k] * rho_avg_inv;
            dvy_x = mass * (vy[jj] - vy[ii]) * W_pair[k] * rho_avg_inv;
            vx_xsph_out[ii] += dvx_x;
            vy_xsph_out[ii] += dvy_x;
            vx_xsph_out[jj] -= dvx_x;
            vy_xsph_out[jj] -= dvy_x;
        }
    }

    {
        int i;
        for (i = 0; i < n_fluid; i++)
            ax_out[i] += gx;
    }

    mxFree(pair_i);
    mxFree(pair_j);
    mxFree(dx_pair);
    mxFree(dy_pair);
    mxFree(r_pair);
    mxFree(W_pair);
    mxFree(dWdr_pair);
}

