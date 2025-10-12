import numpy
import matplotlib.pyplot as plot

# region 3.2.1


def construct_A(N, k):
    index = lambda i, j: i * N + j
    A = numpy.zeros([numpy.power(N, 2), numpy.power(N, 2)])
    for i in range(numpy.power(N, 2)):
        A[i, i] = 4 * k
    for i in range(N):
        for j in range(N):
            if i > 0:
                A[index(i - 1, j), index(i, j)] = -k
                A[index(i, j), index(i - 1, j)] = -k
            if j > 0:
                A[index(i, j - 1), index(i, j)] = -k
                A[index(i, j), index(i, j - 1)] = -k
            if i < N - 1:
                A[index(i + 1, j), index(i, j)] = -k
                A[index(i, j), index(i + 1, j)] = -k
            if j < N - 1:
                A[index(i, j + 1), index(i, j)] = -k
                A[index(i, j), index(i, j + 1)] = -k
    return A


A = construct_A(5, 1)
max = numpy.abs(A).max()
plot.matshow(A, cmap='seismic', vmin=-max, vmax=max)
plot.title("matrix A")
plot.colorbar()
plot.show()

# endregion

# region 3.2.2


def LU_decompose(A: numpy.ndarray):
    dim = A.shape[0]
    P = numpy.zeros([dim - 1, dim, dim])
    G = numpy.zeros([dim - 1, dim, dim])
    for i in range(dim - 1):
        max = abs(A[0, i])
        index = i
        for j in range(i + 1, dim):
            if abs(A[j, i]) > max:
                index = j
                max = abs(A[j, i])
        P[i] = numpy.eye(dim)
        if index != i:
            P[i, index, index] = 0
            P[i, i, i] = 0
            P[i, index, i] = 1
            P[i, i, index] = 1
        A = P[i] @ A
        for j in range(i + 1, dim):
            G[i, j, i] = -A[j, i] / A[i, i]
        A = (numpy.eye(dim) + G[i]) @ A
    U = A
    L = numpy.eye(dim)
    for i in range(dim - 1):
        L = (numpy.eye(dim) - G[dim - i - 2]) @ L
        L = P[dim - i - 2] @ L
    P_tot = numpy.eye(dim)
    for i in range(dim - 1):
        L = P[i] @ L
        P_tot = P[i] @ P_tot

    return (L, U, P_tot)


def forward_sol(L: numpy.ndarray, b: numpy.ndarray):
    dim = b.size
    solution = numpy.zeros(dim)
    solution[0] = b[0] / L[0, 0]
    for i in range(1, dim):
        rhs = b[i]
        for j in range(0, i):
            rhs -= L[i, j] * solution[j]
        solution[i] = rhs / L[i, i]
    return solution


def backward_sol(U: numpy.ndarray, b: numpy.ndarray):
    dim = b.size
    solution = numpy.zeros(dim)
    solution[dim - 1] = b[dim - 1] / U[dim - 1, dim - 1]
    for i in range(dim - 2, -1, -1):
        rhs = b[i]
        for j in range(i + 1, dim):
            rhs -= U[i, j] * solution[j]
        solution[i] = rhs / U[i, i]
    return solution


N = 10
A = construct_A(N, 1)
b = numpy.zeros(N * N)
b[44] = b[45] = b[54] = b[55] = 1

L, U, P = LU_decompose(A)
solution = backward_sol(U, forward_sol(L, P @ b))
x = numpy.linspace(0, N, N)
y = numpy.linspace(0, N, N)
X, Y = numpy.meshgrid(x, y)

fig = plot.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, solution.reshape([N, N]), cmap='viridis')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("u(x,y)")
ax.view_init(azim=45)
fig.colorbar(surf)
fig.subplots_adjust(left=0.2)
plot.show()

# endregion

# region 3.2.3

plot.spy(numpy.abs(L) > 1e-9)
plot.title("matrix L")
plot.show()

plot.spy(numpy.abs(U) > 1e-9)
plot.title("matrix U")
plot.show()

# endregion

# region 3.2.4


def cholesky_decompose(A: numpy.ndarray):
    dim = A.shape[0]
    H = numpy.zeros(A.shape)
    H[0, 0] = numpy.sqrt(A[0, 0])
    for i in range(1, dim):
        for j in range(0, i):
            s = A[i, j]
            for k in range(0, j):
                s -= H[i, k] * H[j, k]
            H[i, j] = s / H[j, j]
        H[i, i] = numpy.sqrt(A[i, i] - numpy.power(H[i, 0:i], 2).sum())
    return H


L = cholesky_decompose(A)
y = L.T @ solution
p = numpy.arange(0, N * N, 1)

plot.plot(p, solution, label="u(p)")
plot.plot(p, y, label="y(p)")
plot.xlabel("p")
plot.legend()
plot.show()

print(f"y^T·y/2 = {y.T @ y / 2}")
print(f"u^T·A·u/2 = {solution.T @ A @ solution / 2}")

# endregion