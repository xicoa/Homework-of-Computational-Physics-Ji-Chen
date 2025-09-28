import numpy
import matplotlib.pyplot as plot

# region 2.2.1
add = 100
xmin = 0 - add
xmax = 1 + add
N = 2**20
xs = numpy.linspace(xmin, xmax, N, endpoint=False)

psi = lambda x, n: numpy.sqrt(2) * numpy.sin(n * numpy.pi * x) * numpy.greater_equal(xs, 0) * numpy.less_equal(xs, 1)
psis: numpy.ndarray = 1 / numpy.sqrt(3) * (psi(xs, 1) + psi(xs, 3) + psi(xs, 5))

phis = numpy.fft.fft(psis)
phis_abs = numpy.absolute(phis)
phis_abs_order = numpy.fft.fftshift(phis_abs)
fs = numpy.fft.fftfreq(N, (xmax - xmin) / N)
ks = fs * 2 * numpy.pi
ks_order = numpy.fft.fftshift(ks)

plot.xlim(-50, 50)
plot.xlabel("k")
plot.ylabel("|Phi(k)|")
plot.plot(ks_order, phis_abs_order)
plot.show()


def Time_develop(phis: numpy.ndarray, ks: numpy.ndarray, t: numpy.complexfloating):
    phits = phis * numpy.exp(-1j * t * numpy.power(ks, 2) / 2)
    return phits


dt = 0.2
plot.xlim(-14, 15)
plot.xlabel("x")
plot.ylabel("|Psi(x)|")
for i in range(4):
    phits = Time_develop(phis, ks, i * dt)
    psits = numpy.fft.ifft(phits)
    psits_abs = numpy.abs(psits)
    plot.plot(xs, psits_abs, label=f't={i}Δt')
plot.legend()
plot.show()
# endregion


# region 2.2.2
M = 2**16
N = 2 * M - 1
tmin = -6
tmax = 6 - 0.8 * tmin

S0 = lambda t: 0.9 * numpy.exp(-numpy.power(t - 1, 2) / 0.2) + 0.5 * numpy.exp(-numpy.power(t - 3, 2) / 0.1) + 0.3 * numpy.exp(-numpy.power(t - 6, 2) / 0.4)
R = lambda t: numpy.sqrt(2 / numpy.pi) * numpy.exp(-numpy.power(t, 2) / 0.5)

ts = numpy.linspace(tmin, tmax, M, endpoint=False)
ts_new = numpy.linspace(2 * tmin, 2 * tmax, N, endpoint=False)
S0ts = S0(ts)
Rts = R(ts)
S0fs = numpy.fft.fft(S0ts, n=N)
Rfs = numpy.fft.fft(Rts, n=N)
Sts = numpy.fft.ifft(S0fs * Rfs) * (ts[1] - ts[0])
Sts_real = numpy.real(Sts)
plot.xlim(tmin, tmax)
plot.xlabel("t")
plot.ylabel("signal")
plot.plot(ts, S0ts, label="S_0(t)")
plot.plot(ts_new, Sts_real, label="S(t)")
plot.legend()
plot.show()
# endregion
