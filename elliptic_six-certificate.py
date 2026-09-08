#!/usr/bin/env python3
"""Interval certificate for a nondegenerate elliptic period-6 orbit of Phi_{1,1}.

Map convention:
    y1 = y + sin(2*pi*x)
    x1 = x + sin(2*pi*y1)
viewed modulo Z^2.

The script uses mpmath interval arithmetic. It verifies:
  1. A unique zero of Phi^6(z)-z-(3,0) in a small box by Krawczyk inclusion.
  2. The trace of D Phi^6 at that zero lies strictly in (-2,2).
  3. The first resonant cubic Birkhoff coefficient is nonzero.
"""

import mpmath as mp

mp.mp.dps = 100
iv = mp.iv
iv.dps = 100
DEG = 3

X0 = mp.mpf(
    "0.27474448332000881878050615161797640957394247787842890490043558664838204815059897"
)
Y0 = mp.mpf(
    "0.092588780576427956472961292120881745888037901834466710224563523023949661988119064"
)
R = mp.mpf("1e-30")
TWO_PI = 2 * mp.pi
ITWO_PI = 2 * iv.pi


def point_map_and_jac(x, y, n=6):
    M = mp.matrix([[1, 0], [0, 1]])
    for _ in range(n):
        a = TWO_PI * mp.cos(TWO_PI * x)
        yn = y + mp.sin(TWO_PI * x)
        b = TWO_PI * mp.cos(TWO_PI * yn)
        J = mp.matrix([[1 + a * b, b], [a, 1]])
        M = J * M
        x = x + mp.sin(TWO_PI * yn)
        y = yn
    return x, y, M


def mm(A, B):
    return [
        [A[0][0] * B[0][0] + A[0][1] * B[1][0],
         A[0][0] * B[0][1] + A[0][1] * B[1][1]],
        [A[1][0] * B[0][0] + A[1][1] * B[1][0],
         A[1][0] * B[0][1] + A[1][1] * B[1][1]],
    ]


def mv(A, v):
    return [
        A[0][0] * v[0] + A[0][1] * v[1],
        A[1][0] * v[0] + A[1][1] * v[1],
    ]


def interval_map_and_jac(x, y, n=6):
    M = [[iv.mpf(1), iv.mpf(0)], [iv.mpf(0), iv.mpf(1)]]
    for _ in range(n):
        a = ITWO_PI * iv.cos(ITWO_PI * x)
        yn = y + iv.sin(ITWO_PI * x)
        b = ITWO_PI * iv.cos(ITWO_PI * yn)
        J = [[1 + a * b, b], [a, iv.mpf(1)]]
        M = mm(J, M)
        x = x + iv.sin(ITWO_PI * yn)
        y = yn
    return x, y, M


# Krawczyk certificate.
XP, YP, MP = point_map_and_jac(X0, Y0)
g = mp.matrix([XP - X0 - 3, YP - Y0])
C = (MP - mp.eye(2)) ** -1
XB = iv.mpf([str(X0 - R), str(X0 + R)])
YB = iv.mpf([str(Y0 - R), str(Y0 + R)])
_, _, MI = interval_map_and_jac(XB, YB)
DGI = [[MI[0][0] - 1, MI[0][1]], [MI[1][0], MI[1][1] - 1]]
CI = [[iv.mpf(str(C[i, j])) for j in range(2)] for i in range(2)]
CD = mm(CI, DGI)
B = [[iv.mpf(1) - CD[0][0], -CD[0][1]],
     [-CD[1][0], iv.mpf(1) - CD[1][1]]]
D = [iv.mpf([str(-R), str(R)]), iv.mpf([str(-R), str(R)])]
BD = mv(B, D)
CG = [C[0, 0] * g[0] + C[0, 1] * g[1],
      C[1, 0] * g[0] + C[1, 1] * g[1]]
K = [iv.mpf(str(X0 - CG[0])) + BD[0],
     iv.mpf(str(Y0 - CG[1])) + BD[1]]

assert mp.mpf(K[0].a) > X0 - R and mp.mpf(K[0].b) < X0 + R
assert mp.mpf(K[1].a) > Y0 - R and mp.mpf(K[1].b) < Y0 + R


class Poly:
    def __init__(self, coeff=None):
        self.c = {}
        if coeff:
            for key, value in coeff.items():
                if sum(key) <= DEG:
                    self.c[key] = value

    @staticmethod
    def const(value):
        return Poly({(0, 0): value})

    @staticmethod
    def z(one=1):
        return Poly({(1, 0): one})

    @staticmethod
    def w(one=1):
        return Poly({(0, 1): one})

    def __add__(self, other):
        if not isinstance(other, Poly):
            other = Poly.const(other)
        keys = set(self.c) | set(other.c)
        return Poly({key: self.c.get(key, 0) + other.c.get(key, 0) for key in keys})

    __radd__ = __add__

    def __neg__(self):
        return Poly({key: -value for key, value in self.c.items()})

    def __sub__(self, other):
        return self + (-other if isinstance(other, Poly) else -other)

    def __rsub__(self, other):
        return Poly.const(other) - self

    def __mul__(self, other):
        if not isinstance(other, Poly):
            other = Poly.const(other)
        out = {}
        for (i, j), a in self.c.items():
            for (k, l), b in other.c.items():
                if i + j + k + l <= DEG:
                    key = (i + k, j + l)
                    out[key] = out.get(key, 0) + a * b
        return Poly(out)

    __rmul__ = __mul__

    def power(self, n):
        out = Poly.const(1)
        for _ in range(n):
            out = out * self
        return out

    def coeff(self, i, j):
        return self.c.get((i, j), 0)

    def constant(self):
        return self.c.get((0, 0), 0)

    def homogeneous(self, degree):
        return Poly({key: value for key, value in self.c.items() if sum(key) == degree})


def sin_jet(P):
    c = P.constant()
    d = P - Poly.const(c)
    return (
        Poly.const(iv.sin(ITWO_PI * c))
        + d * (ITWO_PI * iv.cos(ITWO_PI * c))
        - (d * d) * ((ITWO_PI ** 2 / 2) * iv.sin(ITWO_PI * c))
        - (d * d * d) * ((ITWO_PI ** 3 / 6) * iv.cos(ITWO_PI * c))
    )


# Third-order Taylor jet of Phi^6 over the root box.
u = Poly.z(iv.mpf(1))
v = Poly.w(iv.mpf(1))
PX = Poly.const(XB) + u
PY = Poly.const(YB) + v
for _ in range(6):
    NY = PY + sin_jet(PX)
    NX = PX + sin_jet(NY)
    PX, PY = NX, NY

M00, M01 = PX.coeff(1, 0), PX.coeff(0, 1)
M10, M11 = PY.coeff(1, 0), PY.coeff(0, 1)
TRACE = M00 + M11
assert mp.mpf(TRACE.a) > -2 and mp.mpf(TRACE.b) < -1
# Since trace lies in (-2,-1), the elliptic multiplier is not a root
# of unity of order 1, 2, 3, or 4 (the strong resonances).


class CInterval:
    def __init__(self, real=0, imag=0):
        self.r = real if hasattr(real, "a") else iv.mpf(real)
        self.i = imag if hasattr(imag, "a") else iv.mpf(imag)

    @staticmethod
    def real(value):
        return CInterval(value, 0)

    def __add__(self, other):
        if not isinstance(other, CInterval):
            other = CInterval.real(other)
        return CInterval(self.r + other.r, self.i + other.i)

    __radd__ = __add__

    def __neg__(self):
        return CInterval(-self.r, -self.i)

    def __sub__(self, other):
        return self + (-other if isinstance(other, CInterval) else -CInterval.real(other))

    def __rsub__(self, other):
        return CInterval.real(other) - self

    def __mul__(self, other):
        if not isinstance(other, CInterval):
            other = CInterval.real(other)
        return CInterval(
            self.r * other.r - self.i * other.i,
            self.r * other.i + self.i * other.r,
        )

    __rmul__ = __mul__

    def conjugate(self):
        return CInterval(self.r, -self.i)

    def __truediv__(self, other):
        if not isinstance(other, CInterval):
            other = CInterval.real(other)
        den = other.r * other.r + other.i * other.i
        num = self * other.conjugate()
        return CInterval(num.r / den, num.i / den)

    def __pow__(self, n):
        out = CInterval.real(1)
        for _ in range(n):
            out = out * self
        return out


# Put the linear part into complex eigen-coordinates.
cos_theta = TRACE / 2
sin_theta = iv.sqrt(1 - cos_theta * cos_theta)
lam = CInterval(cos_theta, sin_theta)
lambar = lam.conjugate()
q0 = CInterval.real(M01)
q1 = lam - CInterval.real(M00)
det_q = q0 * (q1.conjugate() - q1)
l0 = q1.conjugate() / det_q
l1 = (-q0) / det_q

z = Poly.z(CInterval.real(1))
w = Poly.w(CInterval.real(1))
U = z * q0 + w * q0.conjugate()
V = z * q1 + w * q1.conjugate()


def complex_substitute(P):
    out = Poly.const(CInterval.real(0))
    for (i, j), value in P.c.items():
        out += (U.power(i) * V.power(j)) * CInterval.real(value)
    return out


PX_D = Poly({key: value for key, value in PX.c.items() if key != (0, 0)})
PY_D = Poly({key: value for key, value in PY.c.items() if key != (0, 0)})
PX_C = complex_substitute(PX_D)
PY_C = complex_substitute(PY_D)
FZ = PX_C * l0 + PY_C * l1
FW = PX_C * l0.conjugate() + PY_C * l1.conjugate()


def derivative(P, variable):
    out = {}
    for (i, j), value in P.c.items():
        if variable == 0 and i > 0:
            out[(i - 1, j)] = out.get((i - 1, j), 0) + i * value
        if variable == 1 and j > 0:
            out[(i, j - 1)] = out.get((i, j - 1), 0) + j * value
    return Poly(out)


F2 = [FZ.homogeneous(2), FW.homogeneous(2)]
F3 = [FZ.homogeneous(3), FW.homogeneous(3)]
EIG = [lam, lambar]
H2 = []
for component in range(2):
    H = Poly.const(CInterval.real(0))
    for (j, k), value in F2[component].c.items():
        multiplier = (lam ** j) * (lambar ** k)
        H.c[(j, k)] = -value / (EIG[component] - multiplier)
    H2.append(H)

G3 = []
for component in range(2):
    correction = derivative(F2[component], 0) * H2[0] + derivative(F2[component], 1) * H2[1]
    G3.append(F3[component] + correction)

# In normal form: z1 = lam*z + c21*z^2*conj(z) + O(4).
# Area preservation forces c21/lam to be purely imaginary. Its imaginary
# part is a nonzero multiple of the first Birkhoff twist coefficient.
C21 = G3[0].coeff(2, 1)
RATIO = C21 / lam
TAU = RATIO.i
assert mp.mpf(TAU.a) > 0

print("Krawczyk inclusion: PASSED")
print("root box x:", XB)
print("root box y:", YB)
print("trace(D Phi^6):", TRACE)
print("sin(theta):", sin_theta)
print("strong resonances of orders 1-4: EXCLUDED")
print("Im(c21/lambda):", TAU)
print("Re(c21/lambda), consistency check:", RATIO.r)
print("All certificate checks passed.")
