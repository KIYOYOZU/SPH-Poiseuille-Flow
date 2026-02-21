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

/* 2D 三次样条核函数在 r=0 处的值 W(0,h) = 10/(7π h²) */
static double cubic_kernel_w0(double h)
{
    const double pi = 3.14159265358979323846;
    const double sigma = 10.0 / (7.0 * pi * h * h);
    return sigma;
}

/* 断言辅助：条件不满足时抛出 MATLAB 错误 */
static void require_count(int cond, const char *id, const char *msg)
{
    if (!cond) {
        mexErrMsgIdAndTxt(id, msg);
    }
}

/* 从 prhs 中提取粒子对数组指针，并校验长度一致性 */
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

/*
 * mode_density_correction
 * 密度重初始化 + 核梯度修正矩阵 B。
 *
 * 算法：
 *   1. 对每个流体粒子 i，累加核函数值 sigma_inner（流体-流体对）
 *      和 sigma_contact（流体-壁面对，按体积加权）。
 *   2. 由 sigma 重算密度：rho_i = (sigma_inner + sigma_contact * rho0 / m_i) * rho0 * inv_sigma0
 *   3. 计算体积 Vol = mass / rho。
 *   4. 构造核梯度矩阵 A，通过最小二乘伪逆得到修正矩阵 B，
 *      用加权混合（det(A) 接近 1 时偏向 B，否则退化为单位阵）。
 *
 * 输入（prhs[1..13]）：
 *   pair_i, pair_j, dx, dy, r, W, dW  — 粒子对数据
 *   mass[n_total], n_fluid, n_total, rho0, h, inv_sigma0
 * 输出（plhs[0..2]）：rho[n_total], Vol[n_total], B[n_total×4]
 */
static void mode_density_correction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const double *pair_i;
    const double *pair_j;
    const double *dx;
    const double *dy;
    const double *r;
    const double *W;
    const double *dW;
    const double *mass;
    mwSize n_pairs;
    int n_fluid;
    int n_total;
    double rho0;
    double h;
    double inv_sigma0;
    double W0;
    double *rho_out;
    double *Vol_out;
    double *B_out;
    double *sigma_inner;
    double *sigma_contact;
    double *A11;
    double *A12;
    double *A21;
    double *A22;
    int k;  /* Changed from mwSize to int for OpenMP compatibility */
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

    W0 = cubic_kernel_w0(h);
    for (i = 0; i < n_fluid; ++i) {
        sigma_inner[i] = W0;
    }

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

    for (i = 0; i < n_fluid; ++i) {
        double rhoi = sigma_inner[i] * rho0 * inv_sigma0;
        rhoi += sigma_contact[i] * rho0 * rho0 * inv_sigma0 / mass[i];
        if (rhoi <= 1e-12) {
            rhoi = rho0;
        }
        rho_out[i] = rhoi;
    }

    for (i = 0; i < n_total; ++i) {
        double rhoi = rho_out[i];
        if (rhoi <= 1e-12) {
            rhoi = rho0;
            rho_out[i] = rhoi;
        }
        Vol_out[i] = mass[i] / rhoi;
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (k = 0; k < n_pairs; ++k) {
        int ii = (int)pair_i[k] - 1;
        int jj = (int)pair_j[k] - 1;
        double rk = r[k];
        double dWk = dW[k];
        double ex, ey;

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

    for (i = 0; i < n_total; ++i) {
        B_out[i] = 1.0;
        B_out[i + n_total] = 0.0;
        B_out[i + 2 * n_total] = 0.0;
        B_out[i + 3 * n_total] = 1.0;
    }

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

/*
 * mode_viscous_force
 * 计算层流粘性力（Laplacian 形式，含核梯度修正矩阵 B）。
 *
 * 流体-流体对：
 *   coeff = eBe * mu * dW * Vol_j / (r + 0.01h)
 *   acc_i += coeff * (v_i - v_j)，acc_j -= coeff * (v_i - v_j)
 * 流体-壁面对（镜像无滑移，系数 ×4 补偿单侧积分）：
 *   coeff = 4 * eBe * mu * dW * Vol_j / (r + 0.01h)
 *   acc_i += coeff * (v_i - v_wall_j)
 *
 * 返回值 force = acc * Vol（单位：N/m²·m³ = N·m，
 * 调用方再除以 mass 得加速度，与 SPH 弱形式一致）。
 *
 * 输入（prhs[1..15]）：
 *   pair_i, pair_j, dx, dy, r, dW  — 粒子对数据
 *   vel[n_total×2], Vol[n_total], B[n_total×4],
 *   mu, h, n_fluid, n_total, mass[n_total], wall_vel[n_total×2]
 * 输出（plhs[0]）：force[n_total×2]
 */
static void mode_viscous_force(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const double *pair_i;
    const double *pair_j;
    const double *dx;
    const double *dy;
    const double *r;
    const double *dW;
    mwSize n_pairs;
    const double *vel;
    const double *Vol;
    const double *B;
    const double *mass;
    const double *wall_vel;
    int n_fluid;
    int n_total;
    double mu;
    double h;
    const double *vel_x;
    const double *vel_y;
    const double *wall_vel_x;
    const double *wall_vel_y;
    double *force_out;
    double *force_x;
    double *force_y;
    double *acc_x;
    double *acc_y;
    int k;  /* Changed from mwSize to int for OpenMP compatibility */

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

        if (jj < n_fluid) {
            double b11j = B[jj];
            double b12j = B[jj + n_total];
            double b21j = B[jj + 2 * n_total];
            double b22j = B[jj + 3 * n_total];
            double bs11 = b11i + b11j;
            double bs12 = b12i + b12j;
            double bs21 = b21i + b21j;
            double bs22 = b22i + b22j;
            double eBe = ex * (bs11 * ex + bs12 * ey) + ey * (bs21 * ex + bs22 * ey);
            double denom = rk + 0.01 * h;
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

/*
 * mode_transport_correction
 * 传输速度修正（Shifting），抑制粒子聚集引起的张力不稳定性。
 *
 * 对每个流体粒子 i 计算位移增量 inc：
 *   流体-流体对：inc_i += -dW * Vol_j * B·e，inc_j += dW * Vol_i * B·e
 *   流体-壁面对：inc_i += -2 * dW * Vol_j * B_i·e（单侧，系数 2 来自镜像对称）
 *
 * 限幅：limiter = clamp(100 * |inc|² / h², 0, 1)
 *   pos_i += 0.2 * h² * limiter * inc_i
 *
 * 输入（prhs[1..12]）：
 *   pair_i, pair_j, dx, dy, r, dW  — 粒子对数据
 *   Vol[n_total], B[n_total×4], pos[n_total×2], h, n_fluid, n_total
 * 输出（plhs[0]）：pos[n_total×2]（修正后位置）
 */
static void mode_transport_correction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const double *pair_i;
    const double *pair_j;
    const double *dx;
    const double *dy;
    const double *r;
    const double *dW;
    mwSize n_pairs;
    const double *Vol;
    const double *B;
    const double *pos;
    int n_fluid;
    int n_total;
    double h;
    const double *pos_x;
    const double *pos_y;
    double *pos_out;
    double *pos_out_x;
    double *pos_out_y;
    double *inc_x;
    double *inc_y;
    int k;  /* Changed from mwSize to int for OpenMP compatibility */

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

        if (jj < n_fluid) {
            double b11j = B[jj];
            double b12j = B[jj + n_total];
            double b21j = B[jj + 2 * n_total];
            double b22j = B[jj + 3 * n_total];
            double bs11 = b11i + b11j;
            double bs12 = b12i + b12j;
            double bs21 = b21i + b21j;
            double bs22 = b22i + b22j;
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
            /* 流体-壁面对：单侧修正（系数 2 来自镜像对称） */
            double tx = b11i * ex + b12i * ey;
            double ty = b21i * ex + b22i * ey;
            double coeff = -2.0 * dWk * Vol[jj];
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            inc_x[ii] += coeff * tx;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            inc_y[ii] += coeff * ty;
        }
    }

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

/*
 * mode_integration_1st
 * 第一阶段积分：密度演化 + 压力更新 + 位置半步推进 + 压力梯度力。
 *
 * 步骤：
 *   1. 对每对粒子计算速度散度贡献，累加到 drho_rate（连续方程）。
 *   2. 密度半步推进：rho += drho_rate * rho * (dt/2)。
 *   3. 弱可压缩状态方程：p = p0 * (rho/rho0 - 1)。
 *   4. 位置半步推进：pos += vel * (dt/2)。
 *   5. 计算压力梯度力（反对称 SPH 形式，含壁面 Riemann 压力修正）。
 *
 * 壁面压力：p_wall = p_i + rho_i * r * max(0, -a·e)
 *   （法向加速度分量为负时增加壁面压力，防止粒子穿透）
 *
 * 输入（prhs[1..18]）：
 *   pair_i, pair_j, dx, dy, r, dW  — 粒子对数据
 *   Vol, B, rho, mass, pos, vel, drho_dt, force_prior,
 *   dt, n_fluid, n_total, rho0, p0, c_f, wall_vel
 * 输出（plhs[0..4]）：rho, p, pos, force, drho_dt
 */
static void mode_integration_1st(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const double *pair_i;
    const double *pair_j;
    const double *dx;
    const double *dy;
    const double *r;
    const double *dW;
    mwSize n_pairs;
    const double *Vol;
    const double *B;
    const double *rho_in;
    const double *mass;
    const double *pos_in;
    const double *vel;
    const double *drho_in;
    const double *force_prior;
    double dt;
    int n_fluid;
    int n_total;
    double rho0;
    double p0;
    double c_f;
    const double *wall_vel;

    double *rho_out;
    double *p_out;
    double *pos_out;
    double *force_out;
    double *drho_out;
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
    double *force_x;
    double *force_y;
    double *diss;
    int k;  /* Changed from mwSize to int for OpenMP compatibility */

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

    for (int i = 0; i < n_fluid; ++i) {
        rho_out[i] = rho_out[i] + 0.5 * dt * drho_in[i];
        if (rho_out[i] < 1e-10) {
            rho_out[i] = rho0;
        }
        p_out[i] = p0 * (rho_out[i] / rho0 - 1.0);
        pos_out_x[i] += 0.5 * dt * vel_x[i];
        pos_out_y[i] += 0.5 * dt * vel_y[i];
    }

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

        if (jj < n_fluid) {
            double p_i = p_out[ii];
            double p_j = p_out[jj];
            double b11i = B[ii], b12i = B[ii + n_total], b21i = B[ii + 2 * n_total], b22i = B[ii + 3 * n_total];
            double b11j = B[jj], b12j = B[jj + n_total], b21j = B[jj + 2 * n_total], b22j = B[jj + 3 * n_total];
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

/*
 * mode_integration_2nd
 * 第二阶段积分：位置修正 + 密度散度修正。
 *
 * 步骤：
 *   1. 对每对粒子计算速度散度贡献，累加到 drho_rate（连续方程）。
 *      流体-流体对：对称累加（i 和 j 均贡献）。
 *      流体-壁面对：仅流体粒子 i 贡献（壁面粒子速度由 wall_vel 给定）。
 *   2. 密度散度修正：drho_out = drho_rate * rho。
 *   3. 位置全步推进：pos += vel * (dt/2)（完成第一阶段的半步）。
 *
 * 输入（prhs[1..13]）：
 *   pair_i, pair_j, dx, dy, r, dW  — 粒子对数据
 *   Vol, rho, pos, vel, dt, n_fluid, n_total, wall_vel
 * 输出（plhs[0..2]）：pos, drho_dt, drho_rate（调试用）
 */
static void mode_integration_2nd(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const double *pair_i;
    const double *pair_j;
    const double *dx;
    const double *dy;
    const double *r;
    const double *dW;
    mwSize n_pairs;
    const double *Vol;
    const double *rho;
    const double *pos_in;
    const double *vel;
    double dt;
    int n_fluid;
    int n_total;
    const double *wall_vel;
    const double *vel_x;
    const double *vel_y;
    const double *pos_x;
    const double *pos_y;
    const double *wall_vel_x;
    const double *wall_vel_y;
    double *pos_out;
    double *pos_out_x;
    double *pos_out_y;
    double *drho_out;
    double *force_out;
    double *drho_rate;
    int k;  /* Changed from mwSize to int for OpenMP compatibility */

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

    for (int i = 0; i < n_fluid; ++i) {
        pos_out_x[i] += 0.5 * dt * vel_x[i];
        pos_out_y[i] += 0.5 * dt * vel_y[i];
    }

    drho_rate = (double *)mxCalloc((mwSize)n_total, sizeof(double));

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
