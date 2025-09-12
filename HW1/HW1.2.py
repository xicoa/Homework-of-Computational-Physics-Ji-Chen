import numpy
import matplotlib.pyplot as plt


# 方程组求解，A行列式为0会导致不可预料结果
def solve_linear_system(A: numpy.ndarray, b: numpy.ndarray):
    if b.size == 0:
        return numpy.array([])
    exchange_row = 0
    # 部分支点遴选
    for i in range(1, b.size):
        if abs(A[i][0]) > abs(A[exchange_row][0]):
            exchange_row = i
    for i in range(b.size):
        A[exchange_row][i], A[0][i] = A[0][i], A[exchange_row][i]
    b[exchange_row], b[0] = b[0], b[exchange_row]
    # 消元
    for i in range(1, b.size):
        coef = -A[i][0] / A[0][0]
        A[i][0] = 0
        b[i] += b[0] * coef
        for j in range(1, b.size):
            A[i][j] += A[0][j] * coef
    # 迭代并求解第一行值
    sub_solve: numpy.ndarray = solve_linear_system(A[1:, 1:], b[1:])
    first_solve = (b[0] - (A[0, 1:] * sub_solve).sum()) / A[0][0]
    solve = numpy.hstack((first_solve, sub_solve))
    return solve


xs = numpy.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75])
ys = numpy.array(
    [
        1,
        -0.65592,
        0.19014,
        0.19024,
        -0.37702,
        0.36707,
        -0.22881,
        0.05411,
        0.08239,
        -0.14436,
        0.13414,
        -0.07921,
    ]
)


# 高次多项式插值
A = numpy.empty([xs.size, xs.size], dtype=float)
for i in range(xs.size):
    A[:, i] = numpy.pow(xs, i)
b = ys.copy()
cs = solve_linear_system(A, b)

xs_plot: numpy.ndarray = numpy.linspace(xs.min(), xs.max(), 20000)
ys_plot = numpy.zeros(xs_plot.size)
ys_exact_plot = numpy.exp(-0.8 * xs_plot) * numpy.cos(10 * xs_plot)
for i in range(xs.size):
    ys_plot += numpy.pow(xs_plot, i) * cs[i]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(xs_plot, ys_plot, color='blue', label='11次插值多项式图像')
plt.plot(xs_plot, ys_exact_plot, color='red', label='原始函数图像f(t)')
plt.scatter(xs, ys, color='darkblue', label='采样点')
plt.xlabel('t')
plt.ylabel('震荡信号')
plt.legend()
plt.show()


# 三次样条插值
# setup_spline_equations 将返回不包含边界条件的 A 和 b, 即首行和末行的 A 和 b 为 0
def setup_spline_equations(t: numpy.ndarray, f: numpy.ndarray):
    A = numpy.zeros([t.size, t.size], dtype=float)
    b = numpy.zeros([t.size], dtype=float)
    h = numpy.zeros([t.size], dtype=float)
    for i in range(0, t.size - 1):
        h[i + 1] = t[i + 1] - t[i]
    for i in range(1, t.size - 1):
        A[i][i - 1] = h[i] / 6
        A[i][i] = (h[i] + h[i + 1]) / 3
        A[i][i + 1] = h[i + 1] / 6
        b[i] = (f[i + 1] - f[i]) / h[i + 1] - (f[i] - f[i - 1]) / h[i]
    return (A, b)


# 为避免重复计算, 此处仅得到一系列 M 值, 在 cubic_spline_interpolation 函数中再计算具体对应数值
def cubic_spline_interpolation_solveM(t_points: numpy.ndarray, f_points: numpy.ndarray, f_p_start: float, f_p_end: float):
    (A, b) = setup_spline_equations(t_points, f_points)
    # boundary condition at start
    A[0][0] = -(t_points[1] - t_points[0]) / 3
    A[0][1] = -(t_points[1] - t_points[0]) / 6
    b[0] = f_p_start - (f_points[1] - f_points[0]) / (t_points[1] - t_points[0])
    # boundary condition at end
    n = t_points.size - 1
    A[n][n - 1] = (t_points[n] - t_points[n - 1]) / 6
    A[n][n] = (t_points[n] - t_points[n - 1]) / 3
    b[n] = f_p_end - (f_points[n] - f_points[n - 1]) / (t_points[n] - t_points[n - 1])
    return solve_linear_system(A, b)


# 计算插值
def cubic_spline_interpolation(t_points: numpy.ndarray, f_points: numpy.ndarray, M: numpy.ndarray, t_eval: float):
    for i in range(t_points.size - 1):
        if t_points[i] <= t_eval <= t_points[i + 1]:
            hi1 = t_points[i + 1] - t_points[i]
            Ai = (f_points[i + 1] - f_points[i]) / hi1 - (M[i + 1] - M[i]) * hi1 / 6
            Bi = f_points[i] - M[i] * numpy.power(hi1, 2) / 6
            S = (M[i] * numpy.power(t_points[i + 1] - t_eval, 3) + M[i + 1] * numpy.power(t_eval - t_points[i], 3)) / (6 * hi1) + Ai * (t_eval - t_points[i]) + Bi
            return S
    return 0.0


M = cubic_spline_interpolation_solveM(xs, ys, -0.8, -0.7114)
ys_cubic_plot = numpy.fromiter(map(lambda x: cubic_spline_interpolation(xs, ys, M, x), xs_plot), dtype=float)
plt.plot(xs_plot, ys_cubic_plot, color='blue', label='三次样条插值图像S(t)')
plt.plot(xs_plot, ys_exact_plot, color='red', label='原始函数图像f(t)')
plt.scatter(xs, ys, color='darkblue', label='采样点')
plt.xlabel('t')
plt.ylabel('震荡信号')
plt.legend()
plt.show()


# 计算误差
def abs_diff(t_points: numpy.ndarray, f_points: numpy.ndarray, M: numpy.ndarray, t_eval: float):
    exact = numpy.exp(-0.8 * t_eval) * numpy.cos(10 * t_eval)
    interpolation = cubic_spline_interpolation(t_points, f_points, M, t_eval)
    return abs(exact - interpolation)


diff = numpy.fromiter(map(lambda x: abs_diff(xs, ys, M, x), xs_plot), dtype=float)
plt.plot(xs_plot, diff, label='|S(t)-f(t)|')
plt.xlabel('t')
plt.ylabel('绝对误差')
plt.legend()
plt.show()
