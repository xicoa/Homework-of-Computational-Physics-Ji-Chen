import numpy
import matplotlib.pyplot as plt


# 臭名昭著的 RANDU 算法
def randu_random(seed: int, num: int):
    a: int = 65539
    m: int = pow(2, 31)
    randoms_int = numpy.empty([num], dtype=numpy.int64)
    randoms_int[0] = a * seed % m
    for i in range(1, num):
        randoms_int[i] = a * randoms_int[i - 1] % m
    randoms = numpy.empty([num], dtype=float)
    randoms = randoms_int / m
    return randoms


generate_number = 100000
randoms = randu_random(114514, generate_number)

plt.figure()
plt.scatter(randoms[:-1], randoms[1:], s=0.01)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

figure = plt.figure()
plot3D = figure.add_subplot(111, projection="3d")
plot3D.scatter3D(randoms[:-2], randoms[1:-1], randoms[2:], s=0.01)
plot3D.view_init(elev=30, azim=60)
plot3D.set_xlabel("X")
plot3D.set_ylabel("Y")
plot3D.set_zlabel("Z")
plt.show()


# glibc 的优化 LCG 算法
def glibc_random(seed: int, num: int):
    a: int = 1103515245
    c: int = 12345
    m: int = pow(2, 31)
    randoms_int = numpy.empty([num], numpy.int64)
    randoms_int[0] = (a * seed + c) % m
    for i in range(1, num):
        randoms_int[i] = (a * randoms_int[i - 1] + c) % m
    randoms = numpy.empty([num], dtype=float)
    randoms = randoms_int / m
    return randoms


glibc_randoms = glibc_random(114514, generate_number)
figure = plt.figure()
plot3D = figure.add_subplot(111, projection="3d")
plot3D.scatter3D(glibc_randoms[:-2], glibc_randoms[1:-1], glibc_randoms[2:], s=0.01)
plot3D.view_init(elev=30, azim=60)
plot3D.set_xlabel("X")
plot3D.set_ylabel("Y")
plot3D.set_zlabel("Z")
plt.show()

# 梅森旋转算法
numpy_randoms = numpy.random.rand(generate_number)
figure = plt.figure()
plot3D = figure.add_subplot(111, projection="3d")
plot3D.scatter3D(numpy_randoms[:-2], numpy_randoms[1:-1], numpy_randoms[2:], s=0.01)
plot3D.view_init(elev=30, azim=60)
plot3D.set_xlabel("X")
plot3D.set_ylabel("Y")
plot3D.set_zlabel("Z")
plt.show()
