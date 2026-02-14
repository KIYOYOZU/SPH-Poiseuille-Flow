/*
 * sph_physics_mex.c
 * SPH 物理算子 MEX 实现，支持 OpenMP 并行加速。
 * 通过模式字符串分发到不同物理计算：
 *
 * Modes:
 *   - density_correction   : 密度重初始化 + 核梯度修正矩阵 B
 *   - viscous_force        : 层流粘性力（含壁面无滑移镜像速度）
 *   - transport_correction : 传输速度修正（抑制张力不稳定性）
 *   - integration_1st      : 第一阶段积分（密度演化 + 压力 + 位置半步 + 压力梯度力）
 *   - integration_2nd      : 第二阶段积分（位置修正 + 密度散度修正）
 *
 * Build with OpenMP:
 *   Windows (MSVC): mex -R2018a -O COMPFLAGS="$COMPFLAGS /openmp" sph_physics_mex.c
 *   Linux/macOS:    mex -R2018a -O CFLAGS="$CFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" sph_physics_mex.c
 */

#include "mex.h"
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define EPS_REG 1e-8

/* 2D 三次样条核函数在 r=0 处的值 W(0,h) */
static double cubic_kernel_w0(double h)
{
    const double pi = 3.14159265358979323846;
    const double sigma = 10.0 / (7.0 * pi * h * h);
    return sigma;
}

static void require_count(int cond, const char *id, const char *msg)
{
    if (!cond) {
        mexErrMsgIdAndTxt(id, msg);
    }
}

static void get_pair_data(const mxArray *arr_i, const mxArray *arr_j,
                          const mxArray *arr_dx, const mxArray *arr_dy,
                          const mxArray *arr_r, const mxArray *arr_dW,
                          const double **pair_i, const double **pair_j,
                          const double **dx, const double **dy,
                          const double **r, const double **dW,
                          mwSize *n_pairs)
{
    mwSize ni = mxGetNumberOfElements(arr_i);
    mwSize nj = mxGetNumberOfElements(arr_j);
    mwSize ndx = mxGetNumberOfElements(arr_dx);
    mwSize ndy = mxGetNumberOfElements(arr_dy);
    mwSize nr = mxGetNumberOfElements(arr_r);
    mwSize ndw = mxGetNumberOfElements(arr_dW);

    require_count(ni == nj && ni == ndx && ni == ndy && ni == nr && ni == ndw,
                  "SPH:Physics:pairs", "Pair arrays must have same length.");

    *pair_i = mxGetDoubles(arr_i);
    *pair_j = mxGetDoubles(arr_j);
    *dx = mxGetDoubles(arr_dx);
    *dy = mxGetDoubles(arr_dy);
    *r = mxGetDoubles(arr_r);
    *dW = mxGetDoubles(arr_dW);
    *n_pairs = ni;
}

/* 密度重初始化 + 核梯度修正矩阵 B
 * 输出: rho(密度), Vol(粒子体积), B(2x2修正矩阵，展平为4列) */
static void mode_density_correction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* ---- 粒子对数据（由邻居搜索预计算） ---- */
    const double *pair_i;       /* 粒子对 i 索引 (1-based) */
    const double *pair_j;       /* 粒子对 j 索引 (1-based) */
    const double *dx;           /* 位移 x 分量: x_i - x_j */
    const double *dy;           /* 位移 y 分量: y_i - y_j */
    const double *r;            /* 粒子对距离 |r_ij| */
    const double *W;            /* 核函数值 W(r_ij, h) */
    const double *dW;           /* 核函数径向导数 dW/dr */
    const double *mass;         /* 粒子质量 [n_total] */
    mwSize n_pairs;             /* 粒子对总数 */
    /* ---- 物理/网格参数 ---- */
    int n_fluid;                /* 流体粒子数 */
    int n_total;                /* 总粒子数 = 流体 + 壁面 */
    double rho0;                /* 参考密度 ρ₀ */
    double h;                   /* 光滑长度 */
    double inv_sigma0;          /* 核函数归一化倒数 1/σ₀，用于密度修正 */
    double W0;                  /* 核函数自身贡献 W(0,h) */
    /* ---- 输出数组 ---- */
    double *rho_out;            /* 修正后密度 [n_total] */
    double *Vol_out;            /* 粒子体积 V=m/ρ [n_total] */
    double *B_out;              /* 核梯度修正矩阵 B，展平为 [n_total x 4] */
    /* ---- 中间累加数组 ---- */
    double *sigma_inner;        /* 流体-流体核函数求和 Σ W_ij */
    double *sigma_contact;      /* 流体-壁面接触项核函数求和 */
    double *A11;                /* 梯度修正矩阵 A 的 (1,1) 分量 */
    double *A12;                /* A 的 (1,2) 分量 */
    double *A21;                /* A 的 (2,1) 分量 */
    double *A22;                /* A 的 (2,2) 分量 */
    int k;                      /* 粒子对循环索引 (int 兼容 OpenMP) */
    int i;

    require_count(nrhs == 14, "SPH:Physics:density:nrhs",
                  "density_correction expects 13 inputs after mode.");
    require_count(nlhs == 3, "SPH:Physics:density:nlhs",
                  "density_correction expects 3 outputs.");

    pair_i = mxGetDoubles(prhs[1]);
    pair_j = mxGetDoubles(prhs[2]);
    dx = mxGetDoubles(prhs[3]);
    dy = mxGetDoubles(prhs[4]);
    r = mxGetDoubles(prhs[5]);
    W = mxGetDoubles(prhs[6]);
    dW = mxGetDoubles(prhs[7]);
    n_pairs = mxGetNumberOfElements(prhs[1]);

    require_count(mxGetNumberOfElements(prhs[2]) == n_pairs &&
                  mxGetNumberOfElements(prhs[3]) == n_pairs &&
                  mxGetNumberOfElements(prhs[4]) == n_pairs &&
                  mxGetNumberOfElements(prhs[5]) == n_pairs &&
                  mxGetNumberOfElements(prhs[6]) == n_pairs &&
                  mxGetNumberOfElements(prhs[7]) == n_pairs,
                  "SPH:Physics:density:pairs", "Pair arrays mismatch.");

    mass = mxGetDoubles(prhs[8]);
    n_fluid = (int)mxGetScalar(prhs[9]);
    n_total = (int)mxGetScalar(prhs[10]);
    rho0 = mxGetScalar(prhs[11]);
    h = mxGetScalar(prhs[12]);
    inv_sigma0 = mxGetScalar(prhs[13]);

    require_count(n_fluid > 0 && n_total >= n_fluid, "SPH:Physics:density:count",
                  "Invalid n_fluid/n_total.");
    require_count((int)mxGetNumberOfElements(prhs[8]) == n_total,
                  "SPH:Physics:density:mass", "mass size mismatch.");
    require_count(rho0 > 0.0 && h > 0.0, "SPH:Physics:density:param",
                  "rho0 and h must be positive.");

    plhs[0] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(n_total, 4, mxREAL);
    rho_out = mxGetDoubles(plhs[0]);
    Vol_out = mxGetDoubles(plhs[1]);
    B_out = mxGetDoubles(plhs[2]);

    sigma_inner = (double *)mxCalloc((mwSize)n_fluid, sizeof(double));
    sigma_contact = (double *)mxCalloc((mwSize)n_fluid, sizeof(double));
    A11 = (double *)mxCalloc((mwSize)n_fluid, sizeof(double));
    A12 = (double *)mxCalloc((mwSize)n_fluid, sizeof(double));
    A21 = (double *)mxCalloc((mwSize)n_fluid, sizeof(double));
    A22 = (double *)mxCalloc((mwSize)n_fluid, sizeof(double));

    /* 初始化：每个流体粒子的核函数求和包含自身贡献 W(0,h) */
    W0 = cubic_kernel_w0(h);
    for (i = 0; i < n_fluid; ++i) {
        sigma_inner[i] = W0;
    }

    /* 第一遍遍历：累加核函数值
     * 流体-流体对: σ_i += W_ij, σ_j += W_ij（对称）
     * 流体-壁面对: σ_contact_i += W_ij × (m_j/ρ₀) */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (k = 0; k < n_pairs; ++k) {
        int ii = (int)pair_i[k] - 1;
        int jj = (int)pair_j[k] - 1;
        double wk = W[k];

        if (ii < 0 || ii >= n_fluid || jj < 0 || jj >= n_total) {
            continue;
        }

        if (jj < n_fluid) {
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            sigma_inner[ii] += wk;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            sigma_inner[jj] += wk;
        } else {
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            sigma_contact[ii] += wk * (mass[jj] / rho0);
        }
    }

    for (i = 0; i < n_total; ++i) {
        rho_out[i] = rho0;
    }

    /* 密度重初始化公式:
     * ρ_i = (Σ_j W_ij) × ρ₀ / σ₀  +  Σ_wall W_iw × (m_w/ρ₀) × ρ₀²/(σ₀ × m_i)
     * 其中 σ₀ = Σ W(r_uniform) 为均匀分布下的核函数求和 */
    for (i = 0; i < n_fluid; ++i) {
        rhoi += sigma_contact[i] * rho0 * rho0 * inv_sigma0 / mass[i];
        if (rhoi <= 1e-12) {
            rhoi = rho0;
        }
        rho_out[i] = rhoi;
    }

    /* 计算粒子体积 V_i = m_i / ρ_i */
    for (i = 0; i < n_total; ++i) {
            rhoi = rho0;
            rho_out[i] = rhoi;
        }
        Vol_out[i] = mass[i] / rhoi;
    }

    /* 第二遍遍历：构建核梯度修正矩阵 A
     * A_i = Σ_j V_j × (dW/dr) × (e ⊗ r_ij)
     * 其中 e = r_ij/|r_ij| 为单位方向向量
     * 最终 B = A⁻¹ 用于修正核梯度 */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif

        if (ii < 0 || ii >= n_fluid || jj < 0 || jj >= n_total || rk <= 1e-12) {
            continue;
        }

        ex = dx[k] / rk;
        ey = dy[k] / rk;

        if (jj < n_fluid) {
            double vj = Vol_out[jj];
            double vi = Vol_out[ii];
            double fxj = dWk * vj;
            double fxi = dWk * vi;

            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A11[ii] -= dx[k] * (fxj * ex);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A12[ii] -= dx[k] * (fxj * ey);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A21[ii] -= dy[k] * (fxj * ex);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A22[ii] -= dy[k] * (fxj * ey);

            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A11[jj] -= dx[k] * (fxi * ex);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A12[jj] -= dx[k] * (fxi * ey);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A21[jj] -= dy[k] * (fxi * ex);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A22[jj] -= dy[k] * (fxi * ey);
        } else {
            double vj = Vol_out[jj];
            double fxj = dWk * vj;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A11[ii] -= dx[k] * (fxj * ex);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A12[ii] -= dx[k] * (fxj * ey);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A21[ii] -= dy[k] * (fxj * ex);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            A22[ii] -= dy[k] * (fxj * ey);
        }
    }

    /* 初始化 B 为单位矩阵（壁面粒子不做修正） */
    for (i = 0; i < n_total; ++i) {
        B_out[i] = 1.0;
        B_out[i + n_total] = 0.0;
        B_out[i + 2 * n_total] = 0.0;
        B_out[i + 3 * n_total] = 1.0;
    }

    /* 计算核梯度修正矩阵 B = A⁻¹（伪逆 + 混合权重）
     * 1. 构建 AᵀA 并求逆 → (AᵀA)⁻¹
     * 2. 伪逆 P = (AᵀA)⁻¹ × Aᵀ
     * 3. 混合权重: w₁ = det(A)/(det(A)+max(1-det(A),0))
     *    当 det(A)≈1 时 B≈P（修正生效），det(A)≈0 时 B≈I（退化为标准SPH） */
    for (i = 0; i < n_fluid; ++i) {
        double a11 = A11[i], a12 = A12[i], a21 = A21[i], a22 = A22[i];
        double ata11 = a11 * a11 + a21 * a21 + EPS_REG;
        double ata12 = a11 * a12 + a21 * a22;
        double ata22 = a12 * a12 + a22 * a22 + EPS_REG;
        double det_m = ata11 * ata22 - ata12 * ata12;
        double im11, im12, im22;
        double p11, p12, p21, p22;
        double det_a;
        double det_sqr;
        double denom;
        double w1;
        double w2;

        if (fabs(det_m) < 1e-20) {
            p11 = 1.0;
            p12 = 0.0;
            p21 = 0.0;
            p22 = 1.0;
        } else {
            im11 = ata22 / det_m;
            im12 = -ata12 / det_m;
            im22 = ata11 / det_m;

            p11 = im11 * a11 + im12 * a12;
            p12 = im11 * a21 + im12 * a22;
            p21 = im12 * a11 + im22 * a12;
            p22 = im12 * a21 + im22 * a22;
        }

        det_a = a11 * a22 - a12 * a21;
        det_sqr = fmax(1.0 - det_a, 0.0);
        denom = det_a + det_sqr;
        if (fabs(denom) < 1e-12) {
            w1 = 0.0;
            w2 = 1.0;
        } else {
            w1 = det_a / denom;
            w2 = det_sqr / denom;
        }

        B_out[i] = w1 * p11 + w2;
        B_out[i + n_total] = w1 * p12;
        B_out[i + 2 * n_total] = w1 * p21;
        B_out[i + 3 * n_total] = w1 * p22 + w2;
    }

    mxFree(sigma_inner);
    mxFree(sigma_contact);
    mxFree(A11);
    mxFree(A12);
    mxFree(A21);
    mxFree(A22);
}

/* 粘性力计算：基于修正核梯度的层流粘性模型
 * 壁面粒子使用镜像速度实现无滑移边界条件 */
static void mode_viscous_force(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* ---- 粒子对数据 ---- */
    const double *pair_i;       /* 粒子对 i 索引 */
    const double *pair_j;       /* 粒子对 j 索引 */
    const double *dx;           /* 位移 x 分量 */
    const double *dy;           /* 位移 y 分量 */
    const double *r;            /* 粒子对距离 */
    const double *dW;           /* 核函数径向导数 */
    mwSize n_pairs;             /* 粒子对总数 */
    /* ---- 物理量输入 ---- */
    const double *vel;          /* 速度场 [n_total x 2] */
    const double *Vol;          /* 粒子体积 [n_total] */
    const double *B;            /* 核梯度修正矩阵 [n_total x 4] */
    const double *mass;         /* 粒子质量 [n_total] */
    const double *wall_vel;     /* 壁面粒子速度（无滑移镜像） [n_total x 2] */
    int n_fluid;                /* 流体粒子数 */
    int n_total;                /* 总粒子数 */
    double mu;                  /* 动力粘度 μ */
    double h;                   /* 光滑长度 */
    /* ---- 速度分量指针（列优先布局） ---- */
    const double *vel_x;        /* 流体 x 速度 */
    const double *vel_y;        /* 流体 y 速度 */
    const double *wall_vel_x;   /* 壁面 x 速度 */
    const double *wall_vel_y;   /* 壁面 y 速度 */
    /* ---- 输出 ---- */
    double *force_out;          /* 粘性力输出 [n_total x 2] */
    double *force_x;            /* 粘性力 x 分量 */
    double *force_y;            /* 粘性力 y 分量 */
    /* ---- 中间累加 ---- */
    double *acc_x;              /* 粘性加速度 x 累加 */
    double *acc_y;              /* 粘性加速度 y 累加 */
    int k;                      /* 循环索引 (int 兼容 OpenMP) */

    require_count(nrhs == 16, "SPH:Physics:viscous:nrhs",
                  "viscous_force expects 16 inputs after mode.");
    require_count(nlhs == 1, "SPH:Physics:viscous:nlhs",
                  "viscous_force expects 1 output.");

    get_pair_data(prhs[1], prhs[2], prhs[3], prhs[4], prhs[5], prhs[6],
                  &pair_i, &pair_j, &dx, &dy, &r, &dW, &n_pairs);
    vel = mxGetDoubles(prhs[7]);
    Vol = mxGetDoubles(prhs[8]);
    B = mxGetDoubles(prhs[9]);
    mu = mxGetScalar(prhs[10]);
    h = mxGetScalar(prhs[11]);
    n_fluid = (int)mxGetScalar(prhs[12]);
    n_total = (int)mxGetScalar(prhs[13]);
    mass = mxGetDoubles(prhs[14]);
    wall_vel = mxGetDoubles(prhs[15]);

    require_count(mxGetM(prhs[7]) == (mwSize)n_total && mxGetN(prhs[7]) == 2,
                  "SPH:Physics:viscous:vel", "vel size mismatch.");
    require_count(mxGetNumberOfElements(prhs[8]) == (mwSize)n_total,
                  "SPH:Physics:viscous:Vol", "Vol size mismatch.");
    require_count(mxGetM(prhs[9]) == (mwSize)n_total && mxGetN(prhs[9]) == 4,
                  "SPH:Physics:viscous:B", "B size mismatch.");
    require_count(mxGetNumberOfElements(prhs[14]) == (mwSize)n_total,
                  "SPH:Physics:viscous:mass", "mass size mismatch.");
    require_count(mxGetM(prhs[15]) == (mwSize)n_total && mxGetN(prhs[15]) == 2,
                  "SPH:Physics:viscous:wallvel", "wall_vel size mismatch.");

    plhs[0] = mxCreateDoubleMatrix(n_total, 2, mxREAL);
    force_out = mxGetDoubles(plhs[0]);
    force_x = force_out;
    force_y = force_out + n_total;

    vel_x = vel;
    vel_y = vel + n_total;
    wall_vel_x = wall_vel;
    wall_vel_y = wall_vel + n_total;

    acc_x = (double *)mxCalloc((mwSize)n_total, sizeof(double));
    acc_y = (double *)mxCalloc((mwSize)n_total, sizeof(double));

    /* 粘性力计算主循环：基于修正核梯度的层流粘性模型
     * 公式: F_visc = μ × Σ_j (B_i + B_j) × (dW/dr) × V_j × (v_i - v_j) / (r + 0.01h)
     * 其中 B 为核梯度修正矩阵，确保一阶精度 */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (k = 0; k < n_pairs; ++k) {
        int ii = (int)pair_i[k] - 1;
        int jj = (int)pair_j[k] - 1;
        double rk = r[k];
        double ex, ey;
        double dWk = dW[k];
        double b11i, b12i, b21i, b22i;

        if (ii < 0 || ii >= n_fluid || jj < 0 || jj >= n_total || rk <= 1e-12) {
            continue;
        }

        ex = dx[k] / rk;
        ey = dy[k] / rk;

        b11i = B[ii];
        b12i = B[ii + n_total];
        b21i = B[ii + 2 * n_total];
        b22i = B[ii + 3 * n_total];

        /* 流体-流体对：对称粘性力 */
        /* 流体-流体对：对称粘性力 */
        if (jj < n_fluid) {
            double b11j = B[jj];
            double b12j = B[jj + n_total];
            double b21j = B[jj + 2 * n_total];
            double b22j = B[jj + 3 * n_total];
            double bs11 = b11i + b11j;
            double bs12 = b12i + b12j;
            double bs21 = b21i + b21j;
            double bs22 = b22i + b22j;
            /* 修正核梯度投影: eᵀ(B_i+B_j)e，确保一阶精度 */
            double eBe = ex * (bs11 * ex + bs12 * ey) + ey * (bs21 * ex + bs22 * ey);
            double denom = rk + 0.01 * h;  /* 分母正则化，避免奇异 */
            double dvx = vel_x[ii] - vel_x[jj];
            double dvy = vel_y[ii] - vel_y[jj];
            double coeff_i = eBe * mu * dWk * Vol[jj] / denom;
            double coeff_j = eBe * mu * dWk * Vol[ii] / denom;

            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            acc_x[ii] += coeff_i * dvx;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            acc_y[ii] += coeff_i * dvy;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            acc_x[jj] -= coeff_j * dvx;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            acc_y[jj] -= coeff_j * dvy;
        } else {
            /* 流体-壁面对：无滑移边界条件（镜像速度）
             * 系数 × 4 来自壁面粒子的镜像对称性 */
            double eBe = ex * (b11i * ex + b12i * ey) + ey * (b21i * ex + b22i * ey);
            double denom = rk + 0.01 * h;
            double dvx = vel_x[ii] - wall_vel_x[jj];
            double dvy = vel_y[ii] - wall_vel_y[jj];
            double coeff = 4.0 * eBe * mu * dWk * Vol[jj] / denom;

            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            acc_x[ii] += coeff * dvx;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            acc_y[ii] += coeff * dvy;
        }
    }

    /* 转换加速度为力: F = a × V（体积力形式） */
    for (int i = 0; i < n_fluid; ++i) {
        force_x[i] = acc_x[i] * Vol[i];
        force_y[i] = acc_y[i] * Vol[i];
    }
    for (int i = n_fluid; i < n_total; ++i) {
        force_x[i] = 0.0;
        force_y[i] = 0.0;
    }

    (void)mass;
    mxFree(acc_x);
    mxFree(acc_y);
}

/* 传输速度修正：抑制粒子聚集/空洞（张力不稳定性）
 * 通过额外的位置修正项保持粒子分布均匀 */
static void mode_transport_correction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* ---- 粒子对数据 ---- */
    const double *pair_i;       /* 粒子对 i 索引 */
    const double *pair_j;       /* 粒子对 j 索引 */
    const double *dx;           /* 位移 x 分量 */
    const double *dy;           /* 位移 y 分量 */
    const double *r;            /* 粒子对距离 */
    const double *dW;           /* 核函数径向导数 */
    mwSize n_pairs;             /* 粒子对总数 */
    /* ---- 物理量输入 ---- */
    const double *Vol;          /* 粒子体积 [n_total] */
    const double *B;            /* 核梯度修正矩阵 [n_total x 4] */
    const double *pos;          /* 粒子位置 [n_total x 2] */
    int n_fluid;                /* 流体粒子数 */
    int n_total;                /* 总粒子数 */
    double h;                   /* 光滑长度 */
    /* ---- 位置分量指针 ---- */
    const double *pos_x;        /* x 坐标 */
    const double *pos_y;        /* y 坐标 */
    /* ---- 输出 ---- */
    double *pos_out;            /* 修正后位置 [n_total x 2] */
    double *pos_out_x;          /* 修正后 x 坐标 */
    double *pos_out_y;          /* 修正后 y 坐标 */
    /* ---- 中间累加 ---- */
    double *inc_x;              /* 传输修正增量 x */
    double *inc_y;              /* 传输修正增量 y */
    int k;                      /* 循环索引 (int 兼容 OpenMP) */

    require_count(nrhs == 13, "SPH:Physics:transport:nrhs",
                  "transport_correction expects 13 inputs after mode.");
    require_count(nlhs == 1, "SPH:Physics:transport:nlhs",
                  "transport_correction expects 1 output.");

    get_pair_data(prhs[1], prhs[2], prhs[3], prhs[4], prhs[5], prhs[6],
                  &pair_i, &pair_j, &dx, &dy, &r, &dW, &n_pairs);
    Vol = mxGetDoubles(prhs[7]);
    B = mxGetDoubles(prhs[8]);
    pos = mxGetDoubles(prhs[9]);
    h = mxGetScalar(prhs[10]);
    n_fluid = (int)mxGetScalar(prhs[11]);
    n_total = (int)mxGetScalar(prhs[12]);

    require_count(mxGetNumberOfElements(prhs[7]) == (mwSize)n_total,
                  "SPH:Physics:transport:Vol", "Vol size mismatch.");
    require_count(mxGetM(prhs[8]) == (mwSize)n_total && mxGetN(prhs[8]) == 4,
                  "SPH:Physics:transport:B", "B size mismatch.");
    require_count(mxGetM(prhs[9]) == (mwSize)n_total && mxGetN(prhs[9]) == 2,
                  "SPH:Physics:transport:pos", "pos size mismatch.");

    plhs[0] = mxCreateDoubleMatrix(n_total, 2, mxREAL);
    pos_out = mxGetDoubles(plhs[0]);
    pos_out_x = pos_out;
    pos_out_y = pos_out + n_total;
    pos_x = pos;
    pos_y = pos + n_total;

    memcpy(pos_out_x, pos_x, (mwSize)n_total * sizeof(double));
    memcpy(pos_out_y, pos_y, (mwSize)n_total * sizeof(double));

    inc_x = (double *)mxCalloc((mwSize)n_total, sizeof(double));
    inc_y = (double *)mxCalloc((mwSize)n_total, sizeof(double));

    /* 传输速度修正主循环：
     * Δr_i = -Σ_j V_j × (B_i + B_j) × (dW/dr) × e_ij
     * 作用：将粒子推向低密度区域，抑制张力不稳定性 */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (k = 0; k < n_pairs; ++k) {
        int ii = (int)pair_i[k] - 1;
        int jj = (int)pair_j[k] - 1;
        double rk = r[k];
        double ex, ey;
        double dWk = dW[k];
        double b11i, b12i, b21i, b22i;

        if (ii < 0 || ii >= n_fluid || jj < 0 || jj >= n_total || rk <= 1e-12) {
            continue;
        }

        ex = dx[k] / rk;
        ey = dy[k] / rk;

        b11i = B[ii];
        b12i = B[ii + n_total];
        b21i = B[ii + 2 * n_total];
        b22i = B[ii + 3 * n_total];

        /* 流体-流体对：对称修正 */
        if (jj < n_fluid) {
            double b11j = B[jj];
            double b12j = B[jj + n_total];
            double b21j = B[jj + 2 * n_total];
            double b22j = B[jj + 3 * n_total];
            double bs11 = b11i + b11j;
            double bs12 = b12i + b12j;
            double bs21 = b21i + b21j;
            double bs22 = b22i + b22j;
            /* 修正核梯度方向: t = (B_i + B_j) × e */
            double tx = bs11 * ex + bs12 * ey;
            double ty = bs21 * ex + bs22 * ey;
            double coeff_i = -dWk * Vol[jj];
            double coeff_j = dWk * Vol[ii];

            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            inc_x[ii] += coeff_i * tx;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            inc_y[ii] += coeff_i * ty;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            inc_x[jj] += coeff_j * tx;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            inc_y[jj] += coeff_j * ty;
        } else {
            /* 流体-壁面对：系数 × 2（镜像对称） */
            double tx = b11i * ex + b12i * ey;
            double ty = b21i * ex + b22i * ey;
            double coeff = -2.0 * dWk * Vol[jj];
            inc_x[ii] += coeff * tx;
            inc_y[ii] += coeff * ty;
        }
    }

    /* 应用修正增量: r_new = r_old + ε × Δr，ε = 0.5h 为修正强度 */
    for (int i = 0; i < n_fluid; ++i) {
        double n2 = inc_x[i] * inc_x[i] + inc_y[i] * inc_y[i];
        double limiter = 100.0 * n2 / (h * h);
        double scale = 0.2 * h * h;
        if (limiter > 1.0) limiter = 1.0;
        if (limiter < 0.0) limiter = 0.0;
        pos_out_x[i] += scale * limiter * inc_x[i];
        pos_out_y[i] += scale * limiter * inc_y[i];
    }

    mxFree(inc_x);
    mxFree(inc_y);
}

/* 第一阶段积分（Verlet 分裂格式前半步）：
 * 密度连续性方程演化 → 弱可压缩状态方程求压力 → 位置半步推进 → 压力梯度力 */
static void mode_integration_1st(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* ---- 粒子对数据 ---- */
    const double *pair_i;       /* 粒子对 i 索引 */
    const double *pair_j;       /* 粒子对 j 索引 */
    const double *dx;           /* 位移 x 分量 */
    const double *dy;           /* 位移 y 分量 */
    const double *r;            /* 粒子对距离 */
    const double *dW;           /* 核函数径向导数 */
    mwSize n_pairs;             /* 粒子对总数 */
    /* ---- 物理量输入 ---- */
    const double *Vol;          /* 粒子体积 [n_total] */
    const double *B;            /* 核梯度修正矩阵 [n_total x 4] */
    const double *rho_in;       /* 输入密度 [n_total] */
    const double *mass;         /* 粒子质量 [n_total] */
    const double *pos_in;       /* 输入位置 [n_total x 2] */
    const double *vel;          /* 速度场 [n_total x 2] */
    const double *drho_in;      /* 上一步密度变化率 dρ/dt [n_total] */
    const double *force_prior;  /* 上一步合力（粘性+体积力）[n_total x 2] */
    double dt;                  /* 时间步长 Δt */
    int n_fluid;                /* 流体粒子数 */
    int n_total;                /* 总粒子数 */
    double rho0;                /* 参考密度 ρ₀ */
    double p0;                  /* 状态方程压力系数 p₀ = ρ₀c² */
    double c_f;                 /* 人工声速 c */
    const double *wall_vel;     /* 壁面粒子速度 [n_total x 2] */
    /* ---- 输出数组 ---- */
    double *rho_out;            /* 半步密度 ρ^(n+1/2) [n_total] */
    double *p_out;              /* 压力 p = p₀(ρ/ρ₀ - 1) [n_total] */
    double *pos_out;            /* 半步位置 [n_total x 2] */
    double *force_out;          /* 压力梯度力 [n_total x 2] */
    double *drho_out;           /* 密度散度修正项 [n_total] */
    /* ---- 速度/位置分量指针（列优先） ---- */
    const double *vel_x;
    const double *vel_y;
    const double *pos_x;
    const double *pos_y;
    const double *force_prior_x;
    const double *force_prior_y;
    const double *wall_vel_x;
    const double *wall_vel_y;
    double *pos_out_x;
    double *pos_out_y;
    double *force_x;            /* 压力梯度力 x 分量 */
    double *force_y;            /* 压力梯度力 y 分量 */
    double *diss;               /* 密度耗散累加项（Molteni-Colagrossi δ-SPH） */
    int k;                      /* 循环索引 (int 兼容 OpenMP) */

    require_count(nrhs == 22, "SPH:Physics:int1:nrhs",
                  "integration_1st expects 21 inputs after mode.");
    require_count(nlhs == 5, "SPH:Physics:int1:nlhs",
                  "integration_1st expects 5 outputs.");

    get_pair_data(prhs[1], prhs[2], prhs[3], prhs[4], prhs[5], prhs[6],
                  &pair_i, &pair_j, &dx, &dy, &r, &dW, &n_pairs);
    Vol = mxGetDoubles(prhs[7]);
    B = mxGetDoubles(prhs[8]);
    rho_in = mxGetDoubles(prhs[9]);
    mass = mxGetDoubles(prhs[10]);
    pos_in = mxGetDoubles(prhs[11]);
    vel = mxGetDoubles(prhs[12]);
    drho_in = mxGetDoubles(prhs[13]);
    force_prior = mxGetDoubles(prhs[14]);
    dt = mxGetScalar(prhs[15]);
    n_fluid = (int)mxGetScalar(prhs[16]);
    n_total = (int)mxGetScalar(prhs[17]);
    rho0 = mxGetScalar(prhs[18]);
    p0 = mxGetScalar(prhs[19]);
    c_f = mxGetScalar(prhs[20]);
    wall_vel = mxGetDoubles(prhs[21]);

    require_count(mxGetM(prhs[8]) == (mwSize)n_total && mxGetN(prhs[8]) == 4,
                  "SPH:Physics:int1:B", "B size mismatch.");
    require_count(mxGetNumberOfElements(prhs[7]) == (mwSize)n_total,
                  "SPH:Physics:int1:Vol", "Vol size mismatch.");
    require_count(mxGetNumberOfElements(prhs[9]) == (mwSize)n_total,
                  "SPH:Physics:int1:rho", "rho size mismatch.");
    require_count(mxGetNumberOfElements(prhs[10]) == (mwSize)n_total,
                  "SPH:Physics:int1:mass", "mass size mismatch.");
    require_count(mxGetM(prhs[11]) == (mwSize)n_total && mxGetN(prhs[11]) == 2,
                  "SPH:Physics:int1:pos", "pos size mismatch.");
    require_count(mxGetM(prhs[12]) == (mwSize)n_total && mxGetN(prhs[12]) == 2,
                  "SPH:Physics:int1:vel", "vel size mismatch.");
    require_count(mxGetNumberOfElements(prhs[13]) == (mwSize)n_total,
                  "SPH:Physics:int1:drho", "drho size mismatch.");
    require_count(mxGetM(prhs[14]) == (mwSize)n_total && mxGetN(prhs[14]) == 2,
                  "SPH:Physics:int1:force_prior", "force_prior size mismatch.");
    require_count(mxGetM(prhs[21]) == (mwSize)n_total && mxGetN(prhs[21]) == 2,
                  "SPH:Physics:int1:wall_vel", "wall_vel size mismatch.");

    plhs[0] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(n_total, 2, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(n_total, 2, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    rho_out = mxGetDoubles(plhs[0]);
    p_out = mxGetDoubles(plhs[1]);
    pos_out = mxGetDoubles(plhs[2]);
    force_out = mxGetDoubles(plhs[3]);
    drho_out = mxGetDoubles(plhs[4]);

    vel_x = vel;
    vel_y = vel + n_total;
    pos_x = pos_in;
    pos_y = pos_in + n_total;
    pos_out_x = pos_out;
    pos_out_y = pos_out + n_total;
    force_x = force_out;
    force_y = force_out + n_total;
    force_prior_x = force_prior;
    force_prior_y = force_prior + n_total;
    wall_vel_x = wall_vel;
    wall_vel_y = wall_vel + n_total;

    memcpy(rho_out, rho_in, (mwSize)n_total * sizeof(double));
    memcpy(pos_out_x, pos_x, (mwSize)n_total * sizeof(double));
    memcpy(pos_out_y, pos_y, (mwSize)n_total * sizeof(double));
    memset(p_out, 0, (mwSize)n_total * sizeof(double));
    memset(force_x, 0, (mwSize)n_total * sizeof(double));
    memset(force_y, 0, (mwSize)n_total * sizeof(double));
    memset(drho_out, 0, (mwSize)n_total * sizeof(double));

    diss = (double *)mxCalloc((mwSize)n_total, sizeof(double));

    /* 第一步：密度半步推进 + 状态方程求压力 + 位置半步推进
     * ρ^(n+1/2) = ρ^n + 0.5 × Δt × (dρ/dt)^n
     * p^(n+1/2) = p₀ × (ρ^(n+1/2)/ρ₀ - 1)  [弱可压缩状态方程]
     * r^(n+1/2) = r^n + 0.5 × Δt × v^n */
    for (int i = 0; i < n_fluid; ++i) {
        rho_out[i] = rho_out[i] + 0.5 * dt * drho_in[i];
        if (rho_out[i] < 1e-10) {
            rho_out[i] = rho0;
        }
        p_out[i] = p0 * (rho_out[i] / rho0 - 1.0);
        pos_out_x[i] += 0.5 * dt * vel_x[i];
        pos_out_y[i] += 0.5 * dt * vel_y[i];
    }

    /* 第二步：计算压力梯度力（对称形式）+ 密度耗散项（δ-SPH）
     * F_pressure = -Σ_j (p_i × B_j + p_j × B_i) × (dW/dr) × V_j × e_ij
     * δ-SPH 耗散: dρ/dt += 2δhc × Σ_j (p_i - p_j)/(ρ₀c) × (dW/dr) × V_j */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (k = 0; k < n_pairs; ++k) {
        int ii = (int)pair_i[k] - 1;
        int jj = (int)pair_j[k] - 1;
        double rk = r[k];
        double ex, ey;
        double dWk = dW[k];

        if (ii < 0 || ii >= n_fluid || jj < 0 || jj >= n_total || rk <= 1e-12) {
            continue;
        }

        ex = dx[k] / rk;
        ey = dy[k] / rk;

        /* 流体-流体对：对称压力梯度 + 密度耗散 */
        if (jj < n_fluid) {
            double p_i = p_out[ii];
            double p_j = p_out[jj];
            double b11i = B[ii], b12i = B[ii + n_total], b21i = B[ii + 2 * n_total], b22i = B[ii + 3 * n_total];
            double b11j = B[jj], b12j = B[jj + n_total], b21j = B[jj + 2 * n_total], b22j = B[jj + 3 * n_total];
            /* 压力梯度修正矩阵: M = p_i × B_j + p_j × B_i */
            double m11 = p_i * b11j + p_j * b11i;
            double m12 = p_i * b12j + p_j * b12i;
            double m21 = p_i * b21j + p_j * b21i;
            double m22 = p_i * b22j + p_j * b22i;
            double tx = m11 * ex + m12 * ey;
            double ty = m21 * ex + m22 * ey;
            double dWVj = dWk * Vol[jj];
            double dWVi = dWk * Vol[ii];
            double p_diff = p_i - p_j;

            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_x[ii] -= tx * dWVj;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_y[ii] -= ty * dWVj;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_x[jj] += tx * dWVi;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_y[jj] += ty * dWVi;

            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            diss[ii] += (p_diff / (rho0 * c_f)) * dWVj;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            diss[jj] += (-p_diff / (rho0 * c_f)) * dWVi;
        } else {
            /* 流体-壁面对：壁面压力外推（考虑体积力）
             * p_wall = p_i + ρ_i × r × max(0, -a·e)
             * 其中 a 为上一步加速度，确保壁面压力非负 */
            double p_i = p_out[ii];
            double rho_i = rho_out[ii];
            double b11i = B[ii], b12i = B[ii + n_total], b21i = B[ii + 2 * n_total], b22i = B[ii + 3 * n_total];
            double dWVj = dWk * Vol[jj];
            double ax = force_prior_x[ii] / mass[ii];
            double ay = force_prior_y[ii] / mass[ii];
            double face_wall_ext_acc = -(ax * ex + ay * ey);
            double p_wall = p_i + rho_i * rk * fmax(0.0, face_wall_ext_acc);
            double tx = b11i * ex + b12i * ey;
            double ty = b21i * ex + b22i * ey;

            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_x[ii] -= (p_i + p_wall) * dWVj * tx;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_y[ii] -= (p_i + p_wall) * dWVj * ty;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            diss[ii] += ((p_i - p_wall) / (rho0 * c_f)) * dWVj;
        }
    }

    /* 第三步：转换为体积力形式 + 应用密度耗散
     * F = a × V，dρ/dt = ρ × (散度项) */
    for (int i = 0; i < n_fluid; ++i) {
        force_x[i] *= Vol[i];
        force_y[i] *= Vol[i];
        drho_out[i] = diss[i] * rho_out[i];
    }
    for (int i = n_fluid; i < n_total; ++i) {
        force_x[i] = 0.0;
        force_y[i] = 0.0;
        drho_out[i] = 0.0;
    }

    (void)wall_vel_x;
    (void)wall_vel_y;
    mxFree(diss);
}

/* 第二阶段积分（Verlet 分裂格式后半步）：
 * 位置修正（完成全步推进）+ 密度散度修正（抑制密度振荡） */
static void mode_integration_2nd(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* ---- 粒子对数据 ---- */
    const double *pair_i;       /* 粒子对 i 索引 */
    const double *pair_j;       /* 粒子对 j 索引 */
    const double *dx;           /* 位移 x 分量 */
    const double *dy;           /* 位移 y 分量 */
    const double *r;            /* 粒子对距离 */
    const double *dW;           /* 核函数径向导数 */
    mwSize n_pairs;             /* 粒子对总数 */
    /* ---- 物理量输入 ---- */
    const double *Vol;          /* 粒子体积 [n_total] */
    const double *rho;          /* 当前密度 [n_total] */
    const double *pos_in;       /* 输入位置 [n_total x 2] */
    const double *vel;          /* 速度场 [n_total x 2] */
    double dt;                  /* 时间步长 Δt */
    int n_fluid;                /* 流体粒子数 */
    int n_total;                /* 总粒子数 */
    const double *wall_vel;     /* 壁面粒子速度 [n_total x 2] */
    /* ---- 速度/位置分量指针 ---- */
    const double *vel_x;
    const double *vel_y;
    const double *pos_x;
    const double *pos_y;
    const double *wall_vel_x;
    const double *wall_vel_y;
    /* ---- 输出数组 ---- */
    double *pos_out;            /* 全步位置 [n_total x 2] */
    double *pos_out_x;
    double *pos_out_y;
    double *drho_out;           /* 密度连续性方程变化率 dρ/dt [n_total] */
    double *force_out;          /* 占位力输出（本阶段为零）[n_total x 2] */
    /* ---- 中间累加 ---- */
    double *drho_rate;          /* 速度散度累加 ∇·v [n_total] */
    int k;                      /* 循环索引 (int 兼容 OpenMP) */

    require_count(nrhs == 15, "SPH:Physics:int2:nrhs",
                  "integration_2nd expects 15 inputs after mode.");
    require_count(nlhs == 3, "SPH:Physics:int2:nlhs",
                  "integration_2nd expects 3 outputs.");

    get_pair_data(prhs[1], prhs[2], prhs[3], prhs[4], prhs[5], prhs[6],
                  &pair_i, &pair_j, &dx, &dy, &r, &dW, &n_pairs);
    Vol = mxGetDoubles(prhs[7]);
    rho = mxGetDoubles(prhs[8]);
    pos_in = mxGetDoubles(prhs[9]);
    vel = mxGetDoubles(prhs[10]);
    dt = mxGetScalar(prhs[11]);
    n_fluid = (int)mxGetScalar(prhs[12]);
    n_total = (int)mxGetScalar(prhs[13]);
    wall_vel = mxGetDoubles(prhs[14]);

    require_count(mxGetNumberOfElements(prhs[7]) == (mwSize)n_total,
                  "SPH:Physics:int2:Vol", "Vol size mismatch.");
    require_count(mxGetNumberOfElements(prhs[8]) == (mwSize)n_total,
                  "SPH:Physics:int2:rho", "rho size mismatch.");
    require_count(mxGetM(prhs[9]) == (mwSize)n_total && mxGetN(prhs[9]) == 2,
                  "SPH:Physics:int2:pos", "pos size mismatch.");
    require_count(mxGetM(prhs[10]) == (mwSize)n_total && mxGetN(prhs[10]) == 2,
                  "SPH:Physics:int2:vel", "vel size mismatch.");
    require_count(mxGetM(prhs[14]) == (mwSize)n_total && mxGetN(prhs[14]) == 2,
                  "SPH:Physics:int2:wall_vel", "wall_vel size mismatch.");

    plhs[0] = mxCreateDoubleMatrix(n_total, 2, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(n_total, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(n_total, 2, mxREAL);
    pos_out = mxGetDoubles(plhs[0]);
    drho_out = mxGetDoubles(plhs[1]);
    force_out = mxGetDoubles(plhs[2]);

    vel_x = vel;
    vel_y = vel + n_total;
    pos_x = pos_in;
    pos_y = pos_in + n_total;
    wall_vel_x = wall_vel;
    wall_vel_y = wall_vel + n_total;
    pos_out_x = pos_out;
    pos_out_y = pos_out + n_total;

    memcpy(pos_out_x, pos_x, (mwSize)n_total * sizeof(double));
    memcpy(pos_out_y, pos_y, (mwSize)n_total * sizeof(double));
    memset(drho_out, 0, (mwSize)n_total * sizeof(double));
    memset(force_out, 0, (mwSize)n_total * 2 * sizeof(double));

    /* 第一步：位置全步推进（完成 Verlet 后半步）
     * r^(n+1) = r^(n+1/2) + 0.5 × Δt × v^(n+1) */
    for (int i = 0; i < n_fluid; ++i) {
        pos_out_x[i] += 0.5 * dt * vel_x[i];
        pos_out_y[i] += 0.5 * dt * vel_y[i];
    }

    drho_rate = (double *)mxCalloc((mwSize)n_total, sizeof(double));

    /* 第二步：计算密度散度修正（抑制密度振荡）
     * dρ/dt = ρ × Σ_j V_j × (v_i - v_j)·e_ij × (dW/dr) */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (k = 0; k < n_pairs; ++k) {
        int ii = (int)pair_i[k] - 1;
        int jj = (int)pair_j[k] - 1;
        double rk = r[k];
        double ex, ey;
        double dWk = dW[k];

        if (ii < 0 || ii >= n_fluid || jj < 0 || jj >= n_total || rk <= 1e-12) {
            continue;
        }

        ex = dx[k] / rk;
        ey = dy[k] / rk;

        /* 流体-流体对：对称速度散度 */
        if (jj < n_fluid) {
            double u_jump = (vel_x[ii] - vel_x[jj]) * ex + (vel_y[ii] - vel_y[jj]) * ey;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            drho_rate[ii] += u_jump * dWk * Vol[jj];
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            drho_rate[jj] += u_jump * dWk * Vol[ii];
        } else {
            /* 流体-壁面对：法向速度散度（考虑壁面法向）
             * 自动识别壁面法向（y<0 → 下壁面，y>0 → 上壁面）
             * 系数 × 2 来自镜像对称 */
            double nwx = 0.0;
            double nwy = (pos_y[jj] < 0.0) ? 1.0 : -1.0;
            double sign_en = ex * nwx + ey * nwy;
            double face_nx = (sign_en >= 0.0) ? nwx : -nwx;
            double face_ny = (sign_en >= 0.0) ? nwy : -nwy;
            double u_jump = 2.0 * ((vel_x[ii] - wall_vel_x[jj]) * face_nx + (vel_y[ii] - wall_vel_y[jj]) * face_ny);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            drho_rate[ii] += u_jump * dWk * Vol[jj];
        }
    }

    /* 第三步：应用密度散度修正
     * dρ/dt = ρ × (速度散度) */
    for (int i = 0; i < n_fluid; ++i) {
        drho_out[i] = drho_rate[i] * rho[i];
    }
    for (int i = n_fluid; i < n_total; ++i) {
        drho_out[i] = 0.0;
    }

    mxFree(drho_rate);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char mode[64];

    require_count(nrhs >= 1, "SPH:Physics:nrhs", "At least mode input is required.");
    require_count(mxIsChar(prhs[0]), "SPH:Physics:mode", "First input must be mode string.");
    mxGetString(prhs[0], mode, sizeof(mode));

    if (strcmp(mode, "density_correction") == 0) {
        mode_density_correction(nlhs, plhs, nrhs, prhs);
    } else if (strcmp(mode, "viscous_force") == 0) {
        mode_viscous_force(nlhs, plhs, nrhs, prhs);
    } else if (strcmp(mode, "transport_correction") == 0) {
        mode_transport_correction(nlhs, plhs, nrhs, prhs);
    } else if (strcmp(mode, "integration_1st") == 0) {
        mode_integration_1st(nlhs, plhs, nrhs, prhs);
    } else if (strcmp(mode, "integration_2nd") == 0) {
        mode_integration_2nd(nlhs, plhs, nrhs, prhs);
    } else {
        mexErrMsgIdAndTxt("SPH:Physics:mode", "Unsupported mode.");
    }
}
