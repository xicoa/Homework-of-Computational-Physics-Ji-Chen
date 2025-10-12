import numpy
import matplotlib.pyplot as plot
import scipy

# region 3.1.a

f1 = lambda x, y: numpy.power(x, 2) + 12 * numpy.power(y, 2) + 3 * x * y + 7 * x + 15 * y


def gradient(f, x, y, delta=1e-8):
    fx = (f(x + delta, y) - f(x - delta, y)) / (2 * delta)
    fy = (f(x, y + delta) - f(x, y - delta)) / (2 * delta)
    return numpy.array([fx, fy])


def hessian(f, x, y, delta=1e-5):
    fxx = (f(x + delta, y) + f(x - delta, y) - 2 * f(x, y)) / (delta**2)
    fyy = (f(x, y + delta) + f(x, y - delta) - 2 * f(x, y)) / (delta**2)
    fxy = (f(x + delta, y + delta) + f(x - delta, y - delta) - f(x + delta, y - delta) - f(x - delta, y + delta)) / (4 * delta**2)
    fyx = fxy
    return numpy.array([[fxx, fxy], [fyx, fyy]])


def steepest_descent(f, x0, y0, epsilon=1e-5):
    xmin = x0
    ymin = y0
    xlist = [x0]
    ylist = [y0]
    value = f(xmin, ymin)
    value_new = value + 2 * epsilon
    while abs(value_new - value) >= epsilon:
        p = -gradient(f, xmin, ymin)
        g = lambda alpha: f(xmin + alpha * p[0], ymin + alpha * p[1])

        alpha_min = 0
        alpha_mid = 1e-5
        while True:
            if g(alpha_mid * 1.618) < g(alpha_mid):
                alpha_mid = alpha_mid * 1.618
            else:
                break
        alpha_max = alpha_mid * 1.618

        while min(g(alpha_max), g(alpha_min)) - g(alpha_mid) > epsilon:
            alpha_try = alpha_max - alpha_mid + alpha_min
            if g(alpha_try) <= g(alpha_mid):
                if alpha_try < alpha_mid:
                    alpha_max = alpha_mid
                    alpha_mid = alpha_try
                else:
                    alpha_min = alpha_mid
                    alpha_mid = alpha_try
            else:
                if alpha_try < alpha_mid:
                    alpha_min = alpha_try
                else:
                    alpha_max = alpha_try
        xmin = xmin + alpha_mid * p[0]
        ymin = ymin + alpha_mid * p[1]
        value = value_new
        value_new = f(xmin, ymin)
        xlist.append(xmin)
        ylist.append(ymin)
    return (xlist, ylist, xmin, ymin)


f1_steepest_xlist, f1_steepest_ylist, f1_steepest_xmin, f1_steepest_ymin = steepest_descent(f1, 0, 0, 1e-8)
print(f1_steepest_xmin, f1_steepest_ymin)


def conjugate_gradient(f, x0, y0, epsilon=1e-5):
    xmin = x0
    ymin = y0
    xlist = [x0]
    ylist = [y0]
    value = f(xmin, ymin)
    value_new = value + 2 * epsilon
    r_old = -gradient(f, xmin, ymin)
    p_old = r_old
    while abs(value_new - value) >= epsilon:
        alpha = (r_old @ r_old) / (p_old @ (hessian(f, xmin, ymin) @ p_old))
        xmin = xmin + alpha * p_old[0]
        ymin = ymin + alpha * p_old[1]
        r = -gradient(f, xmin, ymin)
        beta = beta = (r @ r) / (r_old @ r_old)
        p = r + beta * p_old
        r_old = r
        p_old = p
        value = value_new
        value_new = f(xmin, ymin)
        xlist.append(xmin)
        ylist.append(ymin)
    return (xlist, ylist, xmin, ymin)


f1_conj_xlist, f1_conj_ylist, f1_conj_xmin, f1_conj_ymin = conjugate_gradient(f1, 0, 0, 1e-8)
print(f1_conj_xmin, f1_conj_ymin)


lattice = 2**10
x = numpy.linspace(-5, 0, lattice)
y = numpy.linspace(-1, 0, lattice)
X, Y = numpy.meshgrid(x, y)
plot.plot(f1_steepest_xlist, f1_steepest_ylist, label='steepest_descent_path')
plot.plot(f1_conj_xlist, f1_conj_ylist, label='conjugate_gradient_path')
plot.contourf(X, Y, f1(X, Y), 8, alpha=0.8, cmap=plot.cm.hot)
C = plot.contour(X, Y, f1(X, Y), 8, alpha=0.8, colors='black')
plot.clabel(C, inline=True, fontsize=10)
plot.legend()
plot.show()

# endregion

# region 3.1.b

f2 = lambda x, y: numpy.power(x - 1, 2) + 100 * numpy.power(y - numpy.power(x, 2), 2)


def steepest_descent_fixrate(f, x0, y0, epsilon=1e-5, alpha=1e-1):
    xmin = x0
    ymin = y0
    xlist = [x0]
    ylist = [y0]
    value = f(xmin, ymin)
    value_new = value + 2 * epsilon
    while abs(value_new - value) >= epsilon:
        p = -gradient(f, xmin, ymin)
        g = lambda alpha: f(xmin + alpha * p[0], ymin + alpha * p[1])
        xmin = xmin + alpha * p[0]
        ymin = ymin + alpha * p[1]
        value = value_new
        value_new = f(xmin, ymin)
        xlist.append(xmin)
        ylist.append(ymin)
    return (xlist, ylist, xmin, ymin)


def conjugate_gradient_fixrate(f, x0, y0, epsilon=1e-5, alpha=1e-1):
    xmin = x0
    ymin = y0
    xlist = [x0]
    ylist = [y0]
    value = f(xmin, ymin)
    value_new = value + 2 * epsilon
    r_old = -gradient(f, xmin, ymin)
    p_old = r_old
    while abs(value_new - value) >= epsilon:
        xmin = xmin + alpha * p_old[0]
        ymin = ymin + alpha * p_old[1]
        r = -gradient(f, xmin, ymin)
        beta = beta = (r @ r) / (r_old @ r_old)
        p = r + beta * p_old
        r_old = r
        p_old = p
        value = value_new
        value_new = f(xmin, ymin)
        xlist.append(xmin)
        ylist.append(ymin)
    return (xlist, ylist, xmin, ymin)


def draw_contour_f2():
    lattice = 2**10
    x = numpy.linspace(-0.5, 2, lattice)
    y = numpy.linspace(-0.5, 2, lattice)
    X, Y = numpy.meshgrid(x, y)
    plot.contourf(X, Y, f2(X, Y), 8, alpha=0.8, cmap=plot.cm.hot)
    C = plot.contour(X, Y, f2(X, Y), 8, alpha=0.8, colors='black')
    plot.clabel(C, inline=True, fontsize=10)


draw_contour_f2()
for rate in [0.0001, 0.002]:
    f2_steepest_xlist, f2_steepest_ylist, f2_steepest_xmin, f2_steepest_ymin = steepest_descent_fixrate(f2, 0, 0, 1e-4, rate)
    plot.plot(f2_steepest_xlist, f2_steepest_ylist, label=f'steepest_descent_path (alpha={rate})')
plot.legend()
plot.show()

draw_contour_f2()
rate = 0.006
f2_steepest_xlist, f2_steepest_ylist, f2_steepest_xmin, f2_steepest_ymin = steepest_descent_fixrate(f2, 0, 0, 1e-4, rate)
plot.plot(f2_steepest_xlist, f2_steepest_ylist, label=f'steepest_descent_path (alpha={rate})')
plot.legend()
plot.show()


rate = 0.00001
f2_conj_xlist, f2_conj_ylist, f2_conj_xmin, f2_conj_ymin = conjugate_gradient_fixrate(f2, 0, 0, 1e-4, rate)
plot.plot(f2_conj_xlist, f2_conj_ylist, label=f'conjugate_gradient_path (alpha={rate})')
plot.legend()
plot.show()

rate = 0.0001
f2_conj_xlist, f2_conj_ylist, f2_conj_xmin, f2_conj_ymin = conjugate_gradient_fixrate(f2, 0, 0, 1e-4, rate)
plot.plot(f2_conj_xlist, f2_conj_ylist, label=f'conjugate_gradient_path (alpha={rate})')
plot.legend()
plot.show()

# endregion

# region 3.1.c

def record_path(point):
    path.append(point.copy())


f2_vec = lambda x: numpy.power(x[0] - 1, 2) + 100 * numpy.power(x[1] - numpy.power(x[0], 2), 2)
f2_start = numpy.array([0.0, 0.0])
path = []
df2 = lambda x: gradient(f2, x[0], x[1])
f2_result = scipy.optimize.minimize(f2_vec, f2_start, method='Newton-CG', jac=df2, callback=record_path, options={'disp': True})
print(f2_result.x)
draw_contour_f2()
path = numpy.array(path)
plot.plot(path[:, 0], path[:, 1], label=f'scipy_path')
plot.legend()
plot.show()

# endregion