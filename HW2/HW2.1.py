import time
import numpy

repeat_times = 10000


# region Forward
def forward_x1(x1: float, x2: float):
    x1_dot = 1
    x2_dot = 0
    a: float = numpy.power(x1, 2) + numpy.power(x2, 2)
    a_dot: float = 2 * x1 * x1_dot + 2 * x2 * x2_dot
    b: float = numpy.power(numpy.sin(x2), 2)
    b_dot: float = 2 * numpy.sin(x2) * numpy.cos(x2) * x2_dot
    c: float = numpy.cos(x1 * x2)
    c_dot: float = -numpy.sin(x1 * x2) * x2 * x1_dot - numpy.sin(x1 * x2) * x1 * x2_dot
    f1 = a * b * c
    f1_dot = a * b * c_dot + a * b_dot * c + a_dot * b * c
    f2 = numpy.exp(a) - 4 * b
    f2_dot = numpy.exp(a) * a_dot - 4 * b_dot
    f3 = 2 * a + b + c
    f3_dot = 2 * a_dot + b_dot + c_dot
    return [float(f1_dot), float(f2_dot), float(f3_dot)]


def forward_x2(x1: float, x2: float):
    x1_dot = 0
    x2_dot = 1
    a: float = numpy.power(x1, 2) + numpy.power(x2, 2)
    a_dot: float = 2 * x1 * x1_dot + 2 * x2 * x2_dot
    b: float = numpy.power(numpy.sin(x2), 2)
    b_dot: float = 2 * numpy.sin(x2) * numpy.cos(x2) * x2_dot
    c: float = numpy.cos(x1 * x2)
    c_dot: float = -numpy.sin(x1 * x2) * x2 * x1_dot - numpy.sin(x1 * x2) * x1 * x2_dot
    f1 = a * b * c
    f1_dot = a * b * c_dot + a * b_dot * c + a_dot * b * c
    f2 = numpy.exp(a) - 4 * b
    f2_dot = numpy.exp(a) * a_dot - 4 * b_dot
    f3 = 2 * a + b + c
    f3_dot = 2 * a_dot + b_dot + c_dot
    return [float(f1_dot), float(f2_dot), float(f3_dot)]


time_forward1 = time.time_ns()
nabla_x1 = None
nabla_x2 = None
for i in range(repeat_times):
    nabla_x1 = forward_x1(5, 10)
    nabla_x2 = forward_x2(5, 10)
print(f"∂_x1 f1, ∂_x1 f2, ∂_x1 f3 = {nabla_x1}\n∂_x2 f1, ∂_x2 f2, ∂_x2 f3 = {nabla_x2}")
time_forward2 = time.time_ns()
time_forward_s = (time_forward2 - time_forward1) / 1000000000.0
print(time_forward_s)
# endregion


# region Backward
def backward_f1(x1: float, x2: float):
    f1_bar = 1
    f2_bar = 0
    f3_bar = 0
    a: float = numpy.power(x1, 2) + numpy.power(x2, 2)
    b: float = numpy.power(numpy.sin(x2), 2)
    c: float = numpy.cos(x1 * x2)
    f1 = a * b * c
    f2 = numpy.exp(a) - 4 * b
    f3 = 2 * a + b + c
    a_bar = b * c * f1_bar + numpy.exp(a) * f2_bar + 2 * f3_bar
    b_bar = a * c * f1_bar - 4 * f2_bar + f3_bar
    c_bar = a * b * f1_bar + f3_bar
    x1_bar = 2 * x1 * a_bar - numpy.sin(x1 * x2) * x2 * c_bar
    x2_bar = 2 * x2 * a_bar + 2 * numpy.sin(x2) * numpy.cos(x2) * b_bar - numpy.sin(x1 * x2) * x1 * c_bar
    return [float(x1_bar), float(x2_bar)]


def backward_f2(x1: float, x2: float):
    f1_bar = 0
    f2_bar = 1
    f3_bar = 0
    a: float = numpy.power(x1, 2) + numpy.power(x2, 2)
    b: float = numpy.power(numpy.sin(x2), 2)
    c: float = numpy.cos(x1 * x2)
    f1 = a * b * c
    f2 = numpy.exp(a) - 4 * b
    f3 = 2 * a + b + c
    a_bar = b * c * f1_bar + numpy.exp(a) * f2_bar + 2 * f3_bar
    b_bar = a * c * f1_bar - 4 * f2_bar + f3_bar
    c_bar = a * b * f1_bar + f3_bar
    x1_bar = 2 * x1 * a_bar - numpy.sin(x1 * x2) * x2 * c_bar
    x2_bar = 2 * x2 * a_bar + 2 * numpy.sin(x2) * numpy.cos(x2) * b_bar - numpy.sin(x1 * x2) * x1 * c_bar
    return [float(x1_bar), float(x2_bar)]


def backward_f3(x1: float, x2: float):
    f1_bar = 0
    f2_bar = 0
    f3_bar = 1
    a: float = numpy.power(x1, 2) + numpy.power(x2, 2)
    b: float = numpy.power(numpy.sin(x2), 2)
    c: float = numpy.cos(x1 * x2)
    f1 = a * b * c
    f2 = numpy.exp(a) - 4 * b
    f3 = 2 * a + b + c
    a_bar = b * c * f1_bar + numpy.exp(a) * f2_bar + 2 * f3_bar
    b_bar = a * c * f1_bar - 4 * f2_bar + f3_bar
    c_bar = a * b * f1_bar + f3_bar
    x1_bar = 2 * x1 * a_bar - numpy.sin(x1 * x2) * x2 * c_bar
    x2_bar = 2 * x2 * a_bar + 2 * numpy.sin(x2) * numpy.cos(x2) * b_bar - numpy.sin(x1 * x2) * x1 * c_bar
    return [float(x1_bar), float(x2_bar)]


time_backward1 = time.time_ns()
nabla_f1 = None
nabla_f2 = None
nabla_f3 = None
for i in range(repeat_times):
    nabla_f1 = backward_f1(5, 10)
    nabla_f2 = backward_f2(5, 10)
    nabla_f3 = backward_f3(5, 10)
print(f"∂_x1 f1, ∂_x2 f1 = {nabla_f1}\n∂_x1 f2, ∂_x2 f2 = {nabla_f2}\n∂_x1 f3, ∂_x2 f3 = {nabla_f3}")
time_backward2 = time.time_ns()
time_backward_s = (time_backward2 - time_backward1) / 1000000000.0
print(time_backward_s)
# endregion


# region Forward
def forward_x1(x1: float, x2: float, x3: float, x4: float):
    x1_dot = 1
    x2_dot = 0
    x3_dot = 0
    x4_dot = 0
    a: float = numpy.log(x1 * x2)
    a_dot: float = 1 / x1 * x1_dot + 1 / x2 * x2_dot
    b: float = numpy.sin(x2 / x3)
    b_dot: float = numpy.cos(x2 / x3) / x3 * x2_dot - numpy.cos(x2 / x3) * x2 / numpy.power(x3, 2) * x3_dot
    c: float = numpy.power(x1, 3) * numpy.power(x3, 2) * x4
    c_dot: float = 3 * numpy.power(x1, 2) * numpy.power(x3, 2) * x4 * x1_dot + 2 * numpy.power(x1, 3) * x3 * x4 * x3_dot + numpy.power(x1, 3) * numpy.power(x3, 2) * x4_dot
    f = a + b + c
    f_dot = a_dot + b_dot + c_dot
    return float(f_dot)


def forward_x2(x1: float, x2: float, x3: float, x4: float):
    x1_dot = 0
    x2_dot = 1
    x3_dot = 0
    x4_dot = 0
    a: float = numpy.log(x1 * x2)
    a_dot: float = 1 / x1 * x1_dot + 1 / x2 * x2_dot
    b: float = numpy.sin(x2 / x3)
    b_dot: float = numpy.cos(x2 / x3) / x3 * x2_dot - numpy.cos(x2 / x3) * x2 / numpy.power(x3, 2) * x3_dot
    c: float = numpy.power(x1, 3) * numpy.power(x3, 2) * x4
    c_dot: float = 3 * numpy.power(x1, 2) * numpy.power(x3, 2) * x4 * x1_dot + 2 * numpy.power(x1, 3) * x3 * x4 * x3_dot + numpy.power(x1, 3) * numpy.power(x3, 2) * x4_dot
    f = a + b + c
    f_dot = a_dot + b_dot + c_dot
    return float(f_dot)


def forward_x3(x1: float, x2: float, x3: float, x4: float):
    x1_dot = 0
    x2_dot = 0
    x3_dot = 1
    x4_dot = 0
    a: float = numpy.log(x1 * x2)
    a_dot: float = 1 / x1 * x1_dot + 1 / x2 * x2_dot
    b: float = numpy.sin(x2 / x3)
    b_dot: float = numpy.cos(x2 / x3) / x3 * x2_dot - numpy.cos(x2 / x3) * x2 / numpy.power(x3, 2) * x3_dot
    c: float = numpy.power(x1, 3) * numpy.power(x3, 2) * x4
    c_dot: float = 3 * numpy.power(x1, 2) * numpy.power(x3, 2) * x4 * x1_dot + 2 * numpy.power(x1, 3) * x3 * x4 * x3_dot + numpy.power(x1, 3) * numpy.power(x3, 2) * x4_dot
    f = a + b + c
    f_dot = a_dot + b_dot + c_dot
    return float(f_dot)


def forward_x4(x1: float, x2: float, x3: float, x4: float):
    x1_dot = 0
    x2_dot = 0
    x3_dot = 0
    x4_dot = 1
    a: float = numpy.log(x1 * x2)
    a_dot: float = 1 / x1 * x1_dot + 1 / x2 * x2_dot
    b: float = numpy.sin(x2 / x3)
    b_dot: float = numpy.cos(x2 / x3) / x3 * x2_dot - numpy.cos(x2 / x3) * x2 / numpy.power(x3, 2) * x3_dot
    c: float = numpy.power(x1, 3) * numpy.power(x3, 2) * x4
    c_dot: float = 3 * numpy.power(x1, 2) * numpy.power(x3, 2) * x4 * x1_dot + 2 * numpy.power(x1, 3) * x3 * x4 * x3_dot + numpy.power(x1, 3) * numpy.power(x3, 2) * x4_dot
    f = a + b + c
    f_dot = a_dot + b_dot + c_dot
    return float(f_dot)


time_forward1 = time.time_ns()
nabla_x1 = None
nabla_x2 = None
nabla_x3 = None
nabla_x4 = None
for i in range(repeat_times):
    nabla_x1 = forward_x1(2, 4, 6, 9)
    nabla_x2 = forward_x2(2, 4, 6, 9)
    nabla_x3 = forward_x3(2, 4, 6, 9)
    nabla_x4 = forward_x4(2, 4, 6, 9)
print(f"∂_x1 f = {nabla_x1}\n∂_x2 f = {nabla_x2}\n∂_x3 f = {nabla_x3}\n∂_x4 f = {nabla_x4}")
time_forward2 = time.time_ns()
time_forward_s = (time_forward2 - time_forward1) / 1000000000.0
print(time_forward_s)
# endregion


# region Backward
def backward_f(x1: float, x2: float, x3: float, x4: float):
    f_bar = 1
    a: float = numpy.log(x1 * x2)
    b: float = numpy.sin(x2 / x3)
    c: float = numpy.power(x1, 3) * numpy.power(x3, 2) * x4
    f = a + b + c
    a_bar = f_bar
    b_bar = f_bar
    c_bar = f_bar
    x1_bar = 1 / x1 * a_bar + 3 * numpy.power(x1, 2) * numpy.power(x3, 2) * x4 * c_bar
    x2_bar = 1 / x2 * a_bar + numpy.cos(x2 / x3) / x3 * b_bar
    x3_bar = -numpy.cos(x2 / x3) / numpy.power(x3, 2) * b_bar + 2 * numpy.power(x1, 3) * x3 * x4 * c_bar
    x4_bar = numpy.power(x1, 3) * numpy.power(x3, 2) * c_bar
    return [float(x1_bar), float(x2_bar), float(x3_bar), float(x4_bar)]


time_backward1 = time.time_ns()
nabla_f = None
for i in range(repeat_times):
    nabla_f = backward_f(2, 4, 6, 9)
print(f"∂_x1 f, ∂_x2 f, ∂_x3 f, ∂_x4 f = {nabla_f}")
time_backward2 = time.time_ns()
time_backward_s = (time_backward2 - time_backward1) / 1000000000.0
print(time_backward_s)
# endregion
