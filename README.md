# SPH 平板泊肃叶流动模拟

基于**光滑粒子流体动力学 (Smoothed Particle Hydrodynamics, SPH)** 方法的二维平板泊肃叶流 (Poiseuille Flow) 数值模拟程序。

## 物理问题

模拟两平行板之间的压力驱动层流（泊肃叶流动），该问题具有解析解，适合用于验证 SPH 方法的准确性。

**解析解：**

$$u(y) = \frac{1}{2\mu} \left|\frac{dp}{dx}\right| y(H - y)$$

**模拟参数：**

| 参数 | 数值 |
|------|------|
| 板间距 H | 1.0 m |
| 通道长度 L | 2.0 m |
| 参考密度 ρ₀ | 1.0 kg/m³ |
| 动力粘度 μ | 1.0 Pa·s |
| 压力梯度 dp/dx | -8.0 Pa/m |
| 人工声速 cs | 10 m/s |

## 数值方法

- **核函数：** 2D Cubic Spline
- **邻居搜索：** Cell-Linked List（网格链表法），支持周期性边界
- **时间积分：** Velocity Verlet 格式 + 自适应时间步长（CFL & 粘性条件）
- **密度计算：** 核函数求和 + Shepard 密度修正
- **压力方程：** 弱可压缩状态方程 p = cs²(ρ - ρ₀)
- **粘性模型：** Morris (1997) 粘性项
- **速度修正：** XSPH 修正（ε = 0.5）
- **边界条件：** 壁面镜像速度无滑移 + x 方向周期性边界

## 模拟结果

![模拟结果](SPH_Poiseuille_result.png)

- **左图：** 速度剖面对比（蓝线 = 解析解，红点 = SPH 结果）
- **中图：** 粒子分布与速度场
- **右图：** 收敛历史

**L2 相对误差 < 1%**，SPH 结果与解析解高度吻合。

## 运行方式

### 环境要求

- MATLAB R2018b 或更高版本

### 运行

在 MATLAB 中打开并运行 `SPH_Poiseuille.m` 即可，无需额外依赖。

```matlab
run('SPH_Poiseuille.m')
```

程序将自动完成粒子初始化、SPH 模拟计算（约 5 秒物理时间）并输出结果对比图。

## 文件说明

| 文件 | 说明 |
|------|------|
| `SPH_Poiseuille.m` | 主程序，包含完整的 SPH 模拟代码 |
| `SPH_Poiseuille_result.png` | 模拟结果可视化 |

## 参考文献

- Morris, J. P., Fox, P. J., & Zhu, Y. (1997). Modeling low Reynolds number incompressible flows using SPH. *Journal of Computational Physics*, 136(1), 214-226.
- Liu, G. R., & Liu, M. B. (2003). *Smoothed Particle Hydrodynamics: A Meshfree Particle Method*. World Scientific.

## 许可证

本项目仅供学习交流使用。
