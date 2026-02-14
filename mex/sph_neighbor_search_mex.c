/*
 * sph_neighbor_search_mex.c
 * 基于链表法的 2D SPH 邻居搜索（X 方向周期性边界），支持 OpenMP 并行。
 *
 * 算法：将粒子分配到 2D 网格单元（cell_size = 2h），
 *       遍历相邻 9 个单元查找截断半径内的粒子对。
 *       流体-流体对使用最小镜像约定处理 X 方向周期性。
 *
 * Usage:
 *   [pair_i, pair_j, dx, dy, r, W, dW] = sph_neighbor_search_mex(pos, n_fluid, n_total, h, DL)
 *
 * Inputs:
 *   pos      : [n_total x 2] 粒子位置
 *   n_fluid  : 流体粒子数
 *   n_total  : 总粒子数（流体 + 壁面）
 *   h        : 光滑长度
 *   DL       : X 方向周期性域长度
 *
 * Outputs:
 *   pair_i, pair_j : 粒子对索引（1-based）
 *   dx, dy         : 位移分量 x_i - x_j（流体对含周期性修正）
 *   r              : 粒子对距离
 *   W              : 三次样条核函数值
 *   dW             : 核函数径向导数 dW/dr
 */

#include "mex.h"
#include <math.h>
#include <string.h>

typedef struct {
    double *pair_i;
    double *pair_j;
    double *dx;
    double *dy;
    double *r;
    double *W;
    double *dW;
    mwSize count;
    mwSize capacity;
} PairBuffer;

/* 2D 三次样条核函数及其径向导数 */
static void cubic_kernel_2d(double r, double h, double *W, double *dW)
{
    const double pi = 3.14159265358979323846;
    const double sigma = 10.0 / (7.0 * pi * h * h);
    const double q = r / h;

    if (q < 1.0) {
        *W = sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
        *dW = sigma * (-3.0 * q + 2.25 * q * q) / h;
    } else if (q < 2.0) {
        const double tq = 2.0 - q;
        *W = sigma * 0.25 * tq * tq * tq;
        *dW = -sigma * 0.75 * tq * tq / h;
    } else {
        *W = 0.0;
        *dW = 0.0;
    }
}

/* 动态数组初始化（存储粒子对数据） */
static void init_pair_buffer(PairBuffer *buf, mwSize initial_capacity)
{
    buf->count = 0;
    buf->capacity = initial_capacity;
    buf->pair_i = (double *)mxMalloc(initial_capacity * sizeof(double));
    buf->pair_j = (double *)mxMalloc(initial_capacity * sizeof(double));
    buf->dx = (double *)mxMalloc(initial_capacity * sizeof(double));
    buf->dy = (double *)mxMalloc(initial_capacity * sizeof(double));
    buf->r = (double *)mxMalloc(initial_capacity * sizeof(double));
    buf->W = (double *)mxMalloc(initial_capacity * sizeof(double));
    buf->dW = (double *)mxMalloc(initial_capacity * sizeof(double));
}

/* 容量不足时 2 倍扩容 */
static void ensure_capacity(PairBuffer *buf)
{
    mwSize new_capacity;

    if (buf->count < buf->capacity) {
        return;
    }

    new_capacity = buf->capacity * 2;
    if (new_capacity < 1024) {
        new_capacity = 1024;
    }

    buf->pair_i = (double *)mxRealloc(buf->pair_i, new_capacity * sizeof(double));
    buf->pair_j = (double *)mxRealloc(buf->pair_j, new_capacity * sizeof(double));
    buf->dx = (double *)mxRealloc(buf->dx, new_capacity * sizeof(double));
    buf->dy = (double *)mxRealloc(buf->dy, new_capacity * sizeof(double));
    buf->r = (double *)mxRealloc(buf->r, new_capacity * sizeof(double));
    buf->W = (double *)mxRealloc(buf->W, new_capacity * sizeof(double));
    buf->dW = (double *)mxRealloc(buf->dW, new_capacity * sizeof(double));
    buf->capacity = new_capacity;
}

static void free_pair_buffer(PairBuffer *buf)
{
    if (buf->pair_i) mxFree(buf->pair_i);
    if (buf->pair_j) mxFree(buf->pair_j);
    if (buf->dx) mxFree(buf->dx);
    if (buf->dy) mxFree(buf->dy);
    if (buf->r) mxFree(buf->r);
    if (buf->W) mxFree(buf->W);
    if (buf->dW) mxFree(buf->dW);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const double *pos;
    const double *x;
    const double *y;
    int n_fluid;
    int n_total;
    double h;
    double DL;
    mwSize n_rows;
    double y_min;
    double y_max;
    double cell_size;
    int n_cell_x;
    int n_cell_y;
    int n_cells;
    int *cell_head;
    int *next_in_cell;
    int *cell_x;
    int *cell_y;
    int i;
    PairBuffer buf;
    const double r_cut = 2.0;
    double r_cut_sq;
    mwSize out_count;

    if (nrhs != 5) {
        mexErrMsgIdAndTxt("SPH:Neighbor:nrhs", "Expected 5 inputs.");
    }
    if (nlhs != 7) {
        mexErrMsgIdAndTxt("SPH:Neighbor:nlhs", "Expected 7 outputs.");
    }

    if (!mxIsDouble(prhs[0]) || mxGetN(prhs[0]) != 2) {
        mexErrMsgIdAndTxt("SPH:Neighbor:pos", "pos must be a double matrix of size [n_total x 2].");
    }

    pos = mxGetDoubles(prhs[0]);
    n_rows = mxGetM(prhs[0]);
    x = pos;
    y = pos + n_rows;

    n_fluid = (int)mxGetScalar(prhs[1]);
    n_total = (int)mxGetScalar(prhs[2]);
    h = mxGetScalar(prhs[3]);
    DL = mxGetScalar(prhs[4]);

    if (n_total <= 0 || n_fluid <= 0 || n_fluid > n_total || (mwSize)n_total != n_rows) {
        mexErrMsgIdAndTxt("SPH:Neighbor:count", "Invalid n_fluid/n_total or inconsistent pos size.");
    }
    if (h <= 0.0 || DL <= 0.0) {
        mexErrMsgIdAndTxt("SPH:Neighbor:param", "h and DL must be positive.");
    }

    y_min = y[0];
    y_max = y[0];
    for (i = 1; i < n_total; ++i) {
        if (y[i] < y_min) y_min = y[i];
        if (y[i] > y_max) y_max = y[i];
    }

    cell_size = 2.0 * h;
    n_cell_x = (int)ceil(DL / cell_size);
    if (n_cell_x < 1) n_cell_x = 1;
    n_cell_y = (int)ceil((y_max - y_min + 1e-12) / cell_size) + 1;
    if (n_cell_y < 1) n_cell_y = 1;
    n_cells = n_cell_x * n_cell_y;

    cell_head = (int *)mxMalloc((mwSize)n_cells * sizeof(int));
    next_in_cell = (int *)mxMalloc((mwSize)n_total * sizeof(int));
    cell_x = (int *)mxMalloc((mwSize)n_total * sizeof(int));
    cell_y = (int *)mxMalloc((mwSize)n_total * sizeof(int));

    for (i = 0; i < n_cells; ++i) {
        cell_head[i] = -1;
    }

    for (i = 0; i < n_total; ++i) {
        double xi = x[i];
        double yi = y[i];
        int cxi;
        int cyi = (int)floor((yi - y_min) / cell_size);
        int cid;

        /* 所有粒子（含壁面）统一做周期包裹后分配 cell */
        double xi_wrapped = xi - floor(xi / DL) * DL;
        cxi = (int)floor(xi_wrapped / cell_size);

        if (cxi < 0) cxi = 0;
        if (cxi >= n_cell_x) cxi = n_cell_x - 1;
        if (cyi < 0) cyi = 0;
        if (cyi >= n_cell_y) cyi = n_cell_y - 1;

        cell_x[i] = cxi;
        cell_y[i] = cyi;
        cid = cyi * n_cell_x + cxi;

        next_in_cell[i] = cell_head[cid];
        cell_head[cid] = i;
    }

    init_pair_buffer(&buf, (mwSize)n_fluid * 64 + 1024);
    r_cut_sq = (r_cut * h) * (r_cut * h);

    for (i = 0; i < n_fluid; ++i) {
        int cxi = cell_x[i];
        int cyi = cell_y[i];
        int oy;

        for (oy = -1; oy <= 1; ++oy) {
            int cy = cyi + oy;
            int ox;
            if (cy < 0 || cy >= n_cell_y) {
                continue;
            }

            for (ox = -1; ox <= 1; ++ox) {
                int cx = cxi + ox;
                if (cx < 0) {
                    cx += n_cell_x;
                } else if (cx >= n_cell_x) {
                    cx -= n_cell_x;
                }
                if (cx < 0 || cx >= n_cell_x) {
                    continue;
                }
                int cid = cy * n_cell_x + cx;
                int j = cell_head[cid];

                while (j >= 0) {
                    if (j != i) {
                        int skip_pair = 0;
                        double dx_ij;
                        double dy_ij;
                        double r2;
                        double r_ij;
                        double W_ij;
                        double dW_ij;

                        if (j < n_fluid && j < i) {
                            skip_pair = 1;
                        }
                        if (!skip_pair) {
                            dx_ij = x[i] - x[j];
                            /* 所有 pair（含 fluid-wall）均做周期最小像距 */
                            if (dx_ij > 0.5 * DL) {
                                dx_ij -= DL;
                            } else if (dx_ij < -0.5 * DL) {
                                dx_ij += DL;
                            }
                            dy_ij = y[i] - y[j];
                            r2 = dx_ij * dx_ij + dy_ij * dy_ij;

                            if (r2 > 1e-24 && r2 < r_cut_sq) {
                                r_ij = sqrt(r2);
                                cubic_kernel_2d(r_ij, h, &W_ij, &dW_ij);
                                if (W_ij > 0.0 || fabs(dW_ij) > 0.0) {
                                    ensure_capacity(&buf);
                                    buf.pair_i[buf.count] = (double)(i + 1);
                                    buf.pair_j[buf.count] = (double)(j + 1);
                                    buf.dx[buf.count] = dx_ij;
                                    buf.dy[buf.count] = dy_ij;
                                    buf.r[buf.count] = r_ij;
                                    buf.W[buf.count] = W_ij;
                                    buf.dW[buf.count] = dW_ij;
                                    buf.count++;
                                }
                            }
                        }
                    }
                    j = next_in_cell[j];
                }
            }
        }
    }

    out_count = buf.count;
    plhs[0] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[5] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[6] = mxCreateDoubleMatrix(out_count, 1, mxREAL);

    if (out_count > 0) {
        memcpy(mxGetDoubles(plhs[0]), buf.pair_i, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[1]), buf.pair_j, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[2]), buf.dx, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[3]), buf.dy, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[4]), buf.r, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[5]), buf.W, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[6]), buf.dW, out_count * sizeof(double));
    }

    free_pair_buffer(&buf);
    mxFree(cell_head);
    mxFree(next_in_cell);
    mxFree(cell_x);
    mxFree(cell_y);
}
