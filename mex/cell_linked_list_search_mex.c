/*
 * cell_linked_list_search_mex.c
 * MEX C + OpenMP accelerated Cell-Linked List neighbor search
 *
 * Build: mex -O COMPFLAGS="$COMPFLAGS /openmp" cell_linked_list_search_mex.c
 *
 * Usage:
 *   [pair_i, pair_j, dx_pair, dy_pair, r_pair, n_pairs] = ...
 *       cell_linked_list_search_mex(x, y, n_total, L, r_cut, ...
 *       y_min_domain, n_cell_x, n_cell_y, cell_size_x, cell_size_y, n_cells);
 *
 * Output pair_i, pair_j are MATLAB 1-based indices
 */

#include "mex.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- Counting sort: sort particles by cell_id ---- */
static void counting_sort(const int *cell_id, int n, int n_cells,
                          int *sorted_particles, int *cell_start, int *cell_end)
{
    int *count;
    int *offset;
    int i, cid;

    count = (int *)mxCalloc(n_cells, sizeof(int));

    for (i = 0; i < n; i++)
        count[cell_id[i]]++;

    /* prefix sum -> cell_start (0-based) */
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

    /* place particles */
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

/* ---- MEX entry point ---- */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* all variable declarations at top (C89 compatible for MSVC) */
    const double *x;
    const double *y;
    int    n_total;
    double L;
    double r_cut;
    double y_min_domain;
    int    n_cell_x;
    int    n_cell_y;
    double cell_size_x;
    double cell_size_y;
    int    n_cells;
    double r_cut_sq;
    int *cell_id;
    int *sorted_particles;
    int *cell_start;
    int *cell_end_arr;
    int n_threads;
    int max_pairs_per_thread;
    int    **t_pair_i;
    int    **t_pair_j;
    double **t_dx;
    double **t_dy;
    double **t_r;
    int     *t_count;
    int     *t_cap;
    int total_pairs;
    int32_T *out_pi, *out_pj;
    double *out_dx, *out_dy, *out_r;

    /* half-stencil: [self, right, upper-left, up, upper-right] */
    static const int stencil_dx[5] = { 0,  1, -1, 0, 1};
    static const int stencil_dy[5] = { 0,  0,  1, 1, 1};

    /* parameter check */
    if (nrhs != 11)
        mexErrMsgIdAndTxt("SPH:nrhs", "Requires 11 input arguments");
    if (nlhs != 6)
        mexErrMsgIdAndTxt("SPH:nlhs", "Requires 6 output arguments");

    /* read inputs */
    x             = mxGetDoubles(prhs[0]);
    y             = mxGetDoubles(prhs[1]);
    n_total       = (int)mxGetScalar(prhs[2]);
    L             = mxGetScalar(prhs[3]);
    r_cut         = mxGetScalar(prhs[4]);
    y_min_domain  = mxGetScalar(prhs[5]);
    n_cell_x      = (int)mxGetScalar(prhs[6]);
    n_cell_y      = (int)mxGetScalar(prhs[7]);
    cell_size_x   = mxGetScalar(prhs[8]);
    cell_size_y   = mxGetScalar(prhs[9]);
    n_cells       = (int)mxGetScalar(prhs[10]);

    r_cut_sq = r_cut * r_cut;

    /* ---- particle -> cell mapping (0-based cell index) ---- */
    cell_id = (int *)mxMalloc(n_total * sizeof(int));
    {
        int i;
        for (i = 0; i < n_total; i++) {
            double xmod = fmod(x[i], L);
            int cx, cy;
            if (xmod < 0) xmod += L;
            cx = (int)floor(xmod / cell_size_x);
            cy = (int)floor((y[i] - y_min_domain) / cell_size_y);
            if (cx < 0) cx = 0;
            if (cx >= n_cell_x) cx = n_cell_x - 1;
            if (cy < 0) cy = 0;
            if (cy >= n_cell_y) cy = n_cell_y - 1;
            cell_id[i] = cy * n_cell_x + cx;
        }
    }

    /* ---- counting sort ---- */
    sorted_particles = (int *)mxMalloc(n_total * sizeof(int));
    cell_start       = (int *)mxMalloc(n_cells * sizeof(int));
    cell_end_arr     = (int *)mxMalloc(n_cells * sizeof(int));
    counting_sort(cell_id, n_total, n_cells, sorted_particles, cell_start, cell_end_arr);

    /* ---- determine thread count ---- */
    n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    /* ---- per-thread pair buffers (use stdlib malloc for OpenMP safety) ---- */
    max_pairs_per_thread = n_total * 30 / n_threads + 1024;
    t_pair_i = (int **)   malloc(n_threads * sizeof(int *));
    t_pair_j = (int **)   malloc(n_threads * sizeof(int *));
    t_dx     = (double **)malloc(n_threads * sizeof(double *));
    t_dy     = (double **)malloc(n_threads * sizeof(double *));
    t_r      = (double **)malloc(n_threads * sizeof(double *));
    t_count  = (int *)   calloc(n_threads, sizeof(int));
    t_cap    = (int *)   malloc(n_threads * sizeof(int));

    {
        int t;
        for (t = 0; t < n_threads; t++) {
            t_cap[t] = max_pairs_per_thread;
            t_pair_i[t] = (int *)   malloc(t_cap[t] * sizeof(int));
            t_pair_j[t] = (int *)   malloc(t_cap[t] * sizeof(int));
            t_dx[t]     = (double *)malloc(t_cap[t] * sizeof(double));
            t_dy[t]     = (double *)malloc(t_cap[t] * sizeof(double));
            t_r[t]      = (double *)malloc(t_cap[t] * sizeof(double));
        }
    }

    /* ---- OpenMP parallel neighbor search (distribute by cy row) ---- */
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int tid = 0;
        int local_count, local_cap;
        int    *lp_i, *lp_j;
        double *ld_x, *ld_y, *ld_r;
        int cy;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        local_count = 0;
        local_cap   = t_cap[tid];
        lp_i  = t_pair_i[tid];
        lp_j  = t_pair_j[tid];
        ld_x  = t_dx[tid];
        ld_y  = t_dy[tid];
        ld_r  = t_r[tid];

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
                    double x_shift;
                    int cid2, ii;

                    /* skip if y out of bounds */
                    if (ncy < 0 || ncy >= n_cell_y) continue;

                    /* periodic wrap in x */
                    x_shift = 0.0;
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
                        int pi_idx = sorted_particles[ii];
                        double xi = x[pi_idx];
                        double yi = y[pi_idx];
                        int jj_start = (cid1 == cid2) ? (ii + 1) : cell_start[cid2];
                        int jj;

                        for (jj = jj_start; jj <= cell_end_arr[cid2]; jj++) {
                            int pj_idx = sorted_particles[jj];
                            double dxx = xi - x[pj_idx] + x_shift;
                            double dyy = yi - y[pj_idx];
                            double rr_sq = dxx * dxx + dyy * dyy;

                            if (rr_sq < r_cut_sq && rr_sq > 0.0) {
                                /* grow buffer if needed */
                                if (local_count >= local_cap) {
                                    local_cap *= 2;
                                    lp_i = (int *)   realloc(lp_i, local_cap * sizeof(int));
                                    lp_j = (int *)   realloc(lp_j, local_cap * sizeof(int));
                                    ld_x = (double *)realloc(ld_x, local_cap * sizeof(double));
                                    ld_y = (double *)realloc(ld_y, local_cap * sizeof(double));
                                    ld_r = (double *)realloc(ld_r, local_cap * sizeof(double));
                                }
                                lp_i[local_count] = pi_idx + 1;  /* 1-based */
                                lp_j[local_count] = pj_idx + 1;
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

        /* write back thread-local results */
        t_pair_i[tid] = lp_i;
        t_pair_j[tid] = lp_j;
        t_dx[tid]     = ld_x;
        t_dy[tid]     = ld_y;
        t_r[tid]      = ld_r;
        t_count[tid]  = local_count;
        t_cap[tid]    = local_cap;
    }

    /* ---- merge all thread results ---- */
    total_pairs = 0;
    {
        int t;
        for (t = 0; t < n_threads; t++)
            total_pairs += t_count[t];
    }

    /* create output arrays */
    plhs[0] = mxCreateNumericMatrix(total_pairs, 1, mxINT32_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(total_pairs, 1, mxINT32_CLASS, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(total_pairs, 1, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(total_pairs, 1, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(total_pairs, 1, mxREAL);
    plhs[5] = mxCreateDoubleScalar((double)total_pairs);

    out_pi = (int32_T *)mxGetData(plhs[0]);
    out_pj = (int32_T *)mxGetData(plhs[1]);
    out_dx = mxGetDoubles(plhs[2]);
    out_dy = mxGetDoubles(plhs[3]);
    out_r  = mxGetDoubles(plhs[4]);

    {
        int off = 0;
        int t, k;
        for (t = 0; t < n_threads; t++) {
            for (k = 0; k < t_count[t]; k++) {
                out_pi[off] = (int32_T)t_pair_i[t][k];
                out_pj[off] = (int32_T)t_pair_j[t][k];
                out_dx[off] = t_dx[t][k];
                out_dy[off] = t_dy[t][k];
                out_r[off]  = t_r[t][k];
                off++;
            }
            free(t_pair_i[t]);
            free(t_pair_j[t]);
            free(t_dx[t]);
            free(t_dy[t]);
            free(t_r[t]);
        }
    }

    /* free memory */
    mxFree(cell_id);
    mxFree(sorted_particles);
    mxFree(cell_start);
    mxFree(cell_end_arr);
    free(t_pair_i);
    free(t_pair_j);
    free(t_dx);
    free(t_dy);
    free(t_r);
    free(t_count);
    free(t_cap);
}
