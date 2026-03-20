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

typedef struct {
    int *head;
    int *next;
    int *particle;
    double *x;
    double *y;
    mwSize count;
    mwSize capacity;
} CellEntryBuffer;

static int wrap_cell_index(int cell_index, int n_cell_x)
{
    if (n_cell_x <= 0) {
        return 0;
    }
    cell_index %= n_cell_x;
    if (cell_index < 0) {
        cell_index += n_cell_x;
    }
    return cell_index;
}

static void init_cell_entry_buffer(CellEntryBuffer *buf, int n_cells, mwSize capacity)
{
    int i;
    buf->count = 0;
    buf->capacity = capacity;
    buf->head = (int *)mxMalloc((mwSize)n_cells * sizeof(int));
    buf->next = (int *)mxMalloc(capacity * sizeof(int));
    buf->particle = (int *)mxMalloc(capacity * sizeof(int));
    buf->x = (double *)mxMalloc(capacity * sizeof(double));
    buf->y = (double *)mxMalloc(capacity * sizeof(double));

    for (i = 0; i < n_cells; ++i) {
        buf->head[i] = -1;
    }
}

static void insert_cell_entry(CellEntryBuffer *buf, int particle, double x, double y,
                              int cxi, int cyi, int n_cell_x)
{
    mwSize entry = buf->count;
    int cid = cyi * n_cell_x + cxi;

    if (entry >= buf->capacity) {
        mexErrMsgIdAndTxt("SPH:Neighbor:entryCapacity", "Cell entry capacity exceeded.");
    }

    buf->particle[entry] = particle;
    buf->x[entry] = x;
    buf->y[entry] = y;
    buf->next[entry] = buf->head[cid];
    buf->head[cid] = (int)entry;
    buf->count++;
}

static void free_cell_entry_buffer(CellEntryBuffer *buf)
{
    if (buf->head) mxFree(buf->head);
    if (buf->next) mxFree(buf->next);
    if (buf->particle) mxFree(buf->particle);
    if (buf->x) mxFree(buf->x);
    if (buf->y) mxFree(buf->y);
}

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
    /* ---- 输入数据 ---- */
    const double *pos;          /* 粒子位置 [n_total x 2] */
    const double *x;            /* x 坐标列指针 */
    const double *y;            /* y 坐标列指针 */
    int n_fluid;                /* 流体粒子数 */
    int n_total;                /* 总粒子数 */
    double h;                   /* 光滑长度 */
    double DL;                  /* X 方向周期性域长度 */
    mwSize n_rows;              /* 位置矩阵行数（用于验证） */
    /* ---- 网格参数 ---- */
    double y_min;               /* Y 方向最小坐标 */
    double y_max;               /* Y 方向最大坐标 */
    double cell_size;           /* 单元格尺寸 = 2h */
    int n_cell_x;               /* X 方向单元格数 */
    int n_cell_y;               /* Y 方向单元格数 */
    int n_cells;                /* 总单元格数 */
    /* ---- 链表法数据结构 ---- */
    CellEntryBuffer entries;    /* 周期 ghost entry + 实粒子 entry */
    int *cell_x;                /* 实粒子所在单元格 x 索引 [n_total] */
    int *cell_y;                /* 实粒子所在单元格 y 索引 [n_total] */
    int *seen_neighbor;         /* 每个流体粒子的去重标记 [n_total] */
    int i;                      /* 循环索引 */
    /* ---- 粒子对缓冲区 ---- */
    PairBuffer buf;             /* 动态数组存储粒子对 */
    const double r_cut = 2.0;   /* 截断半径（单位：h） */
    double periodic_cutoff;     /* 周期 ghost 插入阈值 = 2h */
    double r_cut_sq;            /* 截断半径平方 */
    mwSize out_count;           /* 最终粒子对数量 */

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

    /* 第一步：计算 Y 方向边界（用于确定网格范围） */
    y_min = y[0];
    y_max = y[0];
    for (i = 1; i < n_total; ++i) {
        if (y[i] < y_min) y_min = y[i];
        if (y[i] > y_max) y_max = y[i];
    }

    /* 第二步：构建 2D 网格（cell_size = 2h，确保相邻单元覆盖截断半径）*/
    cell_size = 2.0 * h;
    n_cell_x = (int)ceil(DL / cell_size);
    if (n_cell_x < 1) n_cell_x = 1;
    n_cell_y = (int)ceil((y_max - y_min + 1e-12) / cell_size) + 1;
    if (n_cell_y < 1) n_cell_y = 1;
    n_cells = n_cell_x * n_cell_y;

    cell_x = (int *)mxMalloc((mwSize)n_total * sizeof(int));
    cell_y = (int *)mxMalloc((mwSize)n_total * sizeof(int));
    seen_neighbor = (int *)mxMalloc((mwSize)n_total * sizeof(int));

    periodic_cutoff = r_cut * h;
    init_cell_entry_buffer(&entries, n_cells, (mwSize)n_total * 3);

    /* 第三步：将所有粒子分配到单元格，并为周期边界附近粒子插入 ghost entry。 */
    for (i = 0; i < n_total; ++i) {
        double xi_wrapped = x[i] - floor(x[i] / DL) * DL;
        double yi = y[i];
        int cxi = wrap_cell_index((int)floor(xi_wrapped / cell_size), n_cell_x);
        int cyi = (int)floor((yi - y_min) / cell_size);

        if (cyi < 0) cyi = 0;
        if (cyi >= n_cell_y) cyi = n_cell_y - 1;

        cell_x[i] = cxi;
        cell_y[i] = cyi;
        insert_cell_entry(&entries, i, xi_wrapped, yi, cxi, cyi, n_cell_x);

        if (xi_wrapped > DL - periodic_cutoff) {
            double ghost_x = xi_wrapped - DL;
            int ghost_cx = wrap_cell_index((int)floor(ghost_x / cell_size), n_cell_x);
            if (ghost_cx != cxi) {
                insert_cell_entry(&entries, i, ghost_x, yi, ghost_cx, cyi, n_cell_x);
            }
        }
        if (xi_wrapped < periodic_cutoff) {
            double ghost_x = xi_wrapped + DL;
            int ghost_cx = wrap_cell_index((int)floor(ghost_x / cell_size), n_cell_x);
            if (ghost_cx != cxi) {
                insert_cell_entry(&entries, i, ghost_x, yi, ghost_cx, cyi, n_cell_x);
            }
        }
    }

    for (i = 0; i < n_total; ++i) {
        seen_neighbor[i] = -1;
    }

    /* 第四步：初始化粒子对缓冲区（动态数组） */
    init_pair_buffer(&buf, (mwSize)n_fluid * 64 + 1024);
    r_cut_sq = (r_cut * h) * (r_cut * h);

    /* 第六步：遍历所有流体粒子，搜索相邻 9 个单元格内的邻居
     * 流体-流体对：只存储 i < j 避免重复
     * 流体-壁面对：全部存储
     * X 方向周期性：使用最小镜像约定（dx > 0.5DL → dx -= DL） */
    for (i = 0; i < n_fluid; ++i) {
        int cxi = cell_x[i];
        int cyi = cell_y[i];
        int oy;

        /* 遍历相邻 9 个单元格（3×3 邻域） */
        for (oy = -1; oy <= 1; ++oy) {
            int cy = cyi + oy;
            int ox;
            if (cy < 0 || cy >= n_cell_y) {
                continue;  /* Y 方向无周期性，跳过越界单元格 */
            }

            for (ox = -1; ox <= 1; ++ox) {
                int cx = cxi + ox;
                /* X 方向周期性：单元格索引循环包裹 */
                if (cx < 0) {
                    cx += n_cell_x;
                } else if (cx >= n_cell_x) {
                    cx -= n_cell_x;
                }
                if (cx < 0 || cx >= n_cell_x) {
                    continue;
                }
                int cid = cy * n_cell_x + cx;
                int entry = entries.head[cid];

                /* 遍历该单元格内的所有 entry（实粒子 + 必要 ghost entry） */
                while (entry >= 0) {
                    int j = entries.particle[entry];
                    if (j != i && seen_neighbor[j] != i) {
                        int skip_pair = 0;
                        double dx_ij;
                        double dy_ij;
                        double r2;
                        double r_ij;
                        double W_ij;
                        double dW_ij;
                        double xi_wrapped = x[i] - floor(x[i] / DL) * DL;

                        /* 流体-流体对：只存储 i < j 避免重复 */
                        if (j < n_fluid && j < i) {
                            skip_pair = 1;
                        }
                        if (!skip_pair) {
                            dx_ij = xi_wrapped - entries.x[entry];
                            /* 最小镜像约定：选择最短距离（周期性边界） */
                            if (dx_ij > 0.5 * DL) {
                                dx_ij -= DL;
                            } else if (dx_ij < -0.5 * DL) {
                                dx_ij += DL;
                            }
                            dy_ij = y[i] - entries.y[entry];
                            r2 = dx_ij * dx_ij + dy_ij * dy_ij;

                            /* 距离筛选：r > 0 且 r < 2h（截断半径） */
                            if (r2 > 1e-24 && r2 < r_cut_sq) {
                                r_ij = sqrt(r2);
                                /* 计算三次样条核函数及其导数 */
                                cubic_kernel_2d(r_ij, h, &W_ij, &dW_ij);
                                if (W_ij > 0.0 || fabs(dW_ij) > 0.0) {
                                    /* 存储粒子对（索引转为 1-based） */
                                    ensure_capacity(&buf);
                                    buf.pair_i[buf.count] = (double)(i + 1);
                                    buf.pair_j[buf.count] = (double)(j + 1);
                                    buf.dx[buf.count] = dx_ij;
                                    buf.dy[buf.count] = dy_ij;
                                    buf.r[buf.count] = r_ij;
                                    buf.W[buf.count] = W_ij;
                                    buf.dW[buf.count] = dW_ij;
                                    buf.count++;
                                    seen_neighbor[j] = i;
                                }
                            }
                        }
                    }
                    entry = entries.next[entry];
                }
            }
        }
    }

    /* 第七步：输出粒子对数据到 MATLAB */
    out_count = buf.count;
    plhs[0] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[5] = mxCreateDoubleMatrix(out_count, 1, mxREAL);
    plhs[6] = mxCreateDoubleMatrix(out_count, 1, mxREAL);

    /* 复制缓冲区数据到输出数组 */
    if (out_count > 0) {
        memcpy(mxGetDoubles(plhs[0]), buf.pair_i, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[1]), buf.pair_j, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[2]), buf.dx, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[3]), buf.dy, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[4]), buf.r, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[5]), buf.W, out_count * sizeof(double));
        memcpy(mxGetDoubles(plhs[6]), buf.dW, out_count * sizeof(double));
    }

    /* 第八步：清理内存 */
    free_pair_buffer(&buf);
    free_cell_entry_buffer(&entries);
    mxFree(cell_x);
    mxFree(cell_y);
    mxFree(seen_neighbor);
}
