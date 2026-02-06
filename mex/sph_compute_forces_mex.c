/*
 * sph_compute_forces_mex.c
 * MEX C + OpenMP accelerated SPH force computation
 *
 * Build: mex -O COMPFLAGS="$COMPFLAGS /openmp" sph_compute_forces_mex.c
 *
 * Usage:
 *   [ax, ay, rho, p, vx_xsph, vy_xsph] = ...
 *       sph_compute_forces_mex(pair_i, pair_j, dx_pair, dy_pair, r_pair, n_pairs, ...
 *       vx, vy, n_total, n_fluid, h, alpha_kernel, W_self, mass, ...
 *       cs, rho0, mu, gx, epsilon_xsph, do_shepard);
 */

#include "mex.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* all variable declarations at top (C89 compatible for MSVC) */
    const double *pair_i_d;
    const double *pair_j_d;
    const int32_T *pair_i_i32;
    const int32_T *pair_j_i32;
    const double *dx_pair;
    const double *dy_pair;
    const double *r_pair;
    int    n_pairs;
    const double *vx;
    const double *vy;
    int    n_total;
    int    n_fluid;
    double h;
    double alpha_kernel;
    double W_self;
    double mass;
    double cs;
    double rho0;
    double mu;
    double gx;
    double epsilon_xsph;
    int    do_shepard;
    double h_inv, h_sq_001, cs_sq;
    int pair_idx_is_int32;
    int *pi, *pj;
    double *W_pair, *dWdr_pair;
    double *rho;
    int n_threads;
    double *p_arr;
    double *ax_out, *ay_out, *vx_xsph_out, *vy_xsph_out;

    /* parameter check */
    if (nrhs != 20)
        mexErrMsgIdAndTxt("SPH:nrhs", "Requires 20 input arguments");
    if (nlhs != 6)
        mexErrMsgIdAndTxt("SPH:nlhs", "Requires 6 output arguments");

    /* read inputs */
    pair_i_d      = NULL;
    pair_j_d      = NULL;
    pair_i_i32    = NULL;
    pair_j_i32    = NULL;
    dx_pair       = mxGetDoubles(prhs[2]);
    dy_pair       = mxGetDoubles(prhs[3]);
    r_pair        = mxGetDoubles(prhs[4]);
    n_pairs       = (int)mxGetScalar(prhs[5]);
    vx            = mxGetDoubles(prhs[6]);
    vy            = mxGetDoubles(prhs[7]);
    n_total       = (int)mxGetScalar(prhs[8]);
    n_fluid       = (int)mxGetScalar(prhs[9]);
    h             = mxGetScalar(prhs[10]);
    alpha_kernel  = mxGetScalar(prhs[11]);
    W_self        = mxGetScalar(prhs[12]);
    mass          = mxGetScalar(prhs[13]);
    cs            = mxGetScalar(prhs[14]);
    rho0          = mxGetScalar(prhs[15]);
    mu            = mxGetScalar(prhs[16]);
    gx            = mxGetScalar(prhs[17]);
    epsilon_xsph  = mxGetScalar(prhs[18]);
    do_shepard    = (int)mxGetScalar(prhs[19]);
    (void)epsilon_xsph;

    if (mxIsInt32(prhs[0]) && mxIsInt32(prhs[1])) {
        pair_idx_is_int32 = 1;
        pair_i_i32 = (const int32_T *)mxGetData(prhs[0]);
        pair_j_i32 = (const int32_T *)mxGetData(prhs[1]);
    } else if (mxIsDouble(prhs[0]) && mxIsDouble(prhs[1])) {
        pair_idx_is_int32 = 0;
        pair_i_d = mxGetDoubles(prhs[0]);
        pair_j_d = mxGetDoubles(prhs[1]);
    } else {
        mexErrMsgIdAndTxt("SPH:pairType", "pair_i/pair_j must both be int32 or double");
    }

    h_inv = 1.0 / h;
    h_sq_001 = 0.01 * h * h;
    cs_sq = cs * cs;

    /* convert pair indices to 0-based int */
    pi = (int *)mxMalloc(n_pairs * sizeof(int));
    pj = (int *)mxMalloc(n_pairs * sizeof(int));
    {
        int k;
        for (k = 0; k < n_pairs; k++) {
            if (pair_idx_is_int32) {
                pi[k] = (int)pair_i_i32[k] - 1;
                pj[k] = (int)pair_j_i32[k] - 1;
            } else {
                pi[k] = (int)pair_i_d[k] - 1;
                pj[k] = (int)pair_j_d[k] - 1;
            }
        }
    }

    /* ---- create outputs and use as working buffers ---- */
    plhs[0] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[5] = mxCreateDoubleMatrix(n_total, 1, mxREAL);

    ax_out      = mxGetDoubles(plhs[0]);
    ay_out      = mxGetDoubles(plhs[1]);
    rho         = mxGetDoubles(plhs[2]);
    p_arr       = mxGetDoubles(plhs[3]);
    vx_xsph_out = mxGetDoubles(plhs[4]);
    vy_xsph_out = mxGetDoubles(plhs[5]);

    memset(ax_out, 0, n_total * sizeof(double));
    memset(ay_out, 0, n_total * sizeof(double));
    memset(rho, 0, n_total * sizeof(double));
    memset(vx_xsph_out, 0, n_total * sizeof(double));
    memset(vy_xsph_out, 0, n_total * sizeof(double));

    /* ---- kernel function computation ---- */
    W_pair    = (double *)mxMalloc(n_pairs * sizeof(double));
    dWdr_pair = (double *)mxMalloc(n_pairs * sizeof(double));

    {
        int k;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (k = 0; k < n_pairs; k++) {
            double q = r_pair[k] * h_inv;
            if (q < 1.0) {
                W_pair[k]    = alpha_kernel * (2.0/3.0 - q*q + 0.5*q*q*q);
                dWdr_pair[k] = alpha_kernel * (-2.0*q + 1.5*q*q) * h_inv;
            } else if (q < 2.0) {
                double tmp = 2.0 - q;
                W_pair[k]    = alpha_kernel * (1.0/6.0) * tmp*tmp*tmp;
                dWdr_pair[k] = alpha_kernel * (-0.5) * tmp*tmp * h_inv;
            } else {
                W_pair[k]    = 0.0;
                dWdr_pair[k] = 0.0;
            }
        }
    }

    /* ---- density summation (thread-private arrays + reduction) ---- */

    n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif

    if (n_threads > 1) {
        double **rho_priv = (double **)malloc(n_threads * sizeof(double *));
        {
            int t;
            for (t = 0; t < n_threads; t++)
                rho_priv[t] = (double *)calloc(n_total, sizeof(double));
        }

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
            for (k = 0; k < n_pairs; k++) {
                my_rho[pi[k]] += W_pair[k];
                my_rho[pj[k]] += W_pair[k];
            }
        }

        /* reduction */
        {
            int i, t;
            for (i = 0; i < n_total; i++) {
                double sum = 0.0;
                for (t = 0; t < n_threads; t++)
                    sum += rho_priv[t][i];
                rho[i] = mass * (sum + W_self);
            }
            for (t = 0; t < n_threads; t++)
                free(rho_priv[t]);
        }
        free(rho_priv);
    } else {
        /* single thread */
        int k;
        for (k = 0; k < n_pairs; k++) {
            rho[pi[k]] += W_pair[k];
            rho[pj[k]] += W_pair[k];
        }
        {
            int i;
            for (i = 0; i < n_total; i++)
                rho[i] = mass * (rho[i] + W_self);
        }
    }

    /* ---- Shepard density correction ---- */
    if (do_shepard) {
        double *shepard = (double *)mxCalloc(n_total, sizeof(double));
        {
            int k;
            for (k = 0; k < n_pairs; k++) {
                double vol_j_W = mass * W_pair[k] / rho[pj[k]];
                double vol_i_W = mass * W_pair[k] / rho[pi[k]];
                shepard[pi[k]] += vol_j_W;
                shepard[pj[k]] += vol_i_W;
            }
        }
        {
            int i;
            for (i = 0; i < n_total; i++) {
                shepard[i] += mass * W_self / rho[i];
                if (shepard[i] < 0.5) shepard[i] = 0.5;
                if (shepard[i] > 1.5) shepard[i] = 1.5;
                rho[i] /= shepard[i];
            }
        }
        mxFree(shepard);
    }

    /* ---- pressure ---- */
    {
        int i;
        for (i = 0; i < n_total; i++)
            p_arr[i] = cs_sq * (rho[i] - rho0);
    }

    /* ---- force computation + XSPH (thread-private + reduction) ---- */
    if (n_threads > 1) {
        double **ax_priv      = (double **)malloc(n_threads * sizeof(double *));
        double **ay_priv      = (double **)malloc(n_threads * sizeof(double *));
        double **vx_xsph_priv = (double **)malloc(n_threads * sizeof(double *));
        double **vy_xsph_priv = (double **)malloc(n_threads * sizeof(double *));
        {
            int t;
            for (t = 0; t < n_threads; t++) {
                ax_priv[t]      = (double *)calloc(n_total, sizeof(double));
                ay_priv[t]      = (double *)calloc(n_total, sizeof(double));
                vx_xsph_priv[t] = (double *)calloc(n_total, sizeof(double));
                vy_xsph_priv[t] = (double *)calloc(n_total, sizeof(double));
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            int tid = 0;
            double *my_ax, *my_ay, *my_vxx, *my_vxy;
            int k;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            #endif
            my_ax  = ax_priv[tid];
            my_ay  = ay_priv[tid];
            my_vxx = vx_xsph_priv[tid];
            my_vxy = vy_xsph_priv[tid];

            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (k = 0; k < n_pairs; k++) {
                int i = pi[k];
                int j = pj[k];
                double rho_i = rho[i];
                double rho_j = rho[j];
                double p_i   = p_arr[i];
                double p_j   = p_arr[j];
                double r_safe = r_pair[k];
                double ex, ey, press_coef, dvx, dvy, visc_common, fx, fy;
                double rho_avg_inv, dvx_xsph, dvy_xsph;

                if (r_safe < 1e-12) r_safe = 1e-12;
                ex = dx_pair[k] / r_safe;
                ey = dy_pair[k] / r_safe;

                /* pressure term */
                press_coef = -mass * (p_i/(rho_i*rho_i) + p_j/(rho_j*rho_j)) * dWdr_pair[k];

                /* viscosity (Morris 1997) */
                dvx = vx[i] - vx[j];
                dvy = vy[i] - vy[j];
                visc_common = mass * 2.0 * mu / (rho_i * rho_j) *
                    (r_pair[k] * dWdr_pair[k]) / (r_pair[k]*r_pair[k] + h_sq_001);

                fx = press_coef * ex + visc_common * dvx;
                fy = press_coef * ey + visc_common * dvy;

                /* symmetric accumulation (Newton's 3rd law) */
                my_ax[i] += fx;
                my_ay[i] += fy;
                my_ax[j] -= fx;
                my_ay[j] -= fy;

                /* XSPH */
                rho_avg_inv = 2.0 / (rho_i + rho_j);
                dvx_xsph = mass * (vx[j] - vx[i]) * W_pair[k] * rho_avg_inv;
                dvy_xsph = mass * (vy[j] - vy[i]) * W_pair[k] * rho_avg_inv;
                my_vxx[i] += dvx_xsph;
                my_vxy[i] += dvy_xsph;
                my_vxx[j] -= dvx_xsph;
                my_vxy[j] -= dvy_xsph;
            }
        }

        /* reduction */
        {
            int i, t;
            for (i = 0; i < n_total; i++) {
                for (t = 0; t < n_threads; t++) {
                    ax_out[i]      += ax_priv[t][i];
                    ay_out[i]      += ay_priv[t][i];
                    vx_xsph_out[i] += vx_xsph_priv[t][i];
                    vy_xsph_out[i] += vy_xsph_priv[t][i];
                }
            }
            for (t = 0; t < n_threads; t++) {
                free(ax_priv[t]);
                free(ay_priv[t]);
                free(vx_xsph_priv[t]);
                free(vy_xsph_priv[t]);
            }
        }
        free(ax_priv);
        free(ay_priv);
        free(vx_xsph_priv);
        free(vy_xsph_priv);
    } else {
        /* single thread */
        int k;
        for (k = 0; k < n_pairs; k++) {
            int i = pi[k];
            int j = pj[k];
            double rho_i = rho[i];
            double rho_j = rho[j];
            double p_i   = p_arr[i];
            double p_j   = p_arr[j];
            double r_safe = r_pair[k];
            double ex, ey, press_coef, dvx, dvy, visc_common, fx, fy;
            double rho_avg_inv, dvx_xsph, dvy_xsph;

            if (r_safe < 1e-12) r_safe = 1e-12;
            ex = dx_pair[k] / r_safe;
            ey = dy_pair[k] / r_safe;

            press_coef = -mass * (p_i/(rho_i*rho_i) + p_j/(rho_j*rho_j)) * dWdr_pair[k];

            dvx = vx[i] - vx[j];
            dvy = vy[i] - vy[j];
            visc_common = mass * 2.0 * mu / (rho_i * rho_j) *
                (r_pair[k] * dWdr_pair[k]) / (r_pair[k]*r_pair[k] + h_sq_001);

            fx = press_coef * ex + visc_common * dvx;
            fy = press_coef * ey + visc_common * dvy;

            ax_out[i] += fx;
            ay_out[i] += fy;
            ax_out[j] -= fx;
            ay_out[j] -= fy;

            rho_avg_inv = 2.0 / (rho_i + rho_j);
            dvx_xsph = mass * (vx[j] - vx[i]) * W_pair[k] * rho_avg_inv;
            dvy_xsph = mass * (vy[j] - vy[i]) * W_pair[k] * rho_avg_inv;
            vx_xsph_out[i] += dvx_xsph;
            vy_xsph_out[i] += dvy_xsph;
            vx_xsph_out[j] -= dvx_xsph;
            vy_xsph_out[j] -= dvy_xsph;
        }
    }

    /* body force (fluid particles only) */
    {
        int i;
        for (i = 0; i < n_fluid; i++)
            ax_out[i] += gx;
    }

    /* free memory */
    mxFree(pi);
    mxFree(pj);
    mxFree(W_pair);
    mxFree(dWdr_pair);
}
