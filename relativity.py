import numpy as np
from scipy import constants


def speed(p, m):
    return 1 / np.sqrt(1 + m * m / (p * p))


def beta_gamma(p, m):
    v = speed(p, m)
    return lorentz_factor(v) * v


def beta(bg):
    return np.sqrt(1 / (1 / (bg * bg) + 1))


def gamma(bg):
    return bg / beta(bg)


def t_diff(s, p, m1, m2):
    return s * (1 / speed(p, m1) - 1 / speed(p, m2)) / constants.c * 1e9


def e_kin(p, m):
    return np.sqrt(p**2 + m**2) - m


def p2e(p, m):
    return e_kin(p, m)


def e2p(e, m):
    return np.sqrt((e + m) * (e + m) - m * m)


def lorentz_factor(v):
    return 1 / np.sqrt(1 - v * v)


def momentum(m, v):
    return m * v * lorentz_factor(v)


def decay_ratio(p, m, d, tau):
    return np.exp(-d * m / (tau * 1e-9 * p * constants.c))


def decay_momentum(m, m1, m2=0):
    return np.sqrt((m**2 + m1**2 + m2**2)**2 - 4 * m**2 * m1**2) / (2 * m)


def decay_energy(m, m1, m2=0):
    return (m**2 + m1**2 - m2**2) / (2 * m)


def decay_angle(theta, p, m, m1, m2=0):
    p1 = decay_momentum(m, m1, m2)
    v = speed(p, m)
    return np.arctan(p1 * np.sin(theta) / (lorentz_factor(v) * (p1 * np.cos(theta) + v * decay_energy(m, m1, m2))))
