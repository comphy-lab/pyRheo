import numpy as np
import math
from scipy.special import gamma

# Pade for Mittag-Leffler m=5, n=4 according to I. O. Sarumi, K. M. Furati and A. Q. M. Khaliq, Journal of Scientific Computing, 2020, 82, 1–27.

# Pade for Mittag-Leffler m=5, n=4
def system_54ab(alpha, beta):
    A = np.array([
        [1, 0, 0, -gamma(beta - alpha) / gamma(beta), 0, 0, 0],
        [0, 1, 0, gamma(beta - alpha) / gamma(beta + alpha), -gamma(beta - alpha) / gamma(beta), 0, 0],
        [0, 0, 1, -gamma(beta - alpha) / gamma(beta + 2 * alpha), gamma(beta - alpha) / gamma(beta + alpha), -gamma(beta - alpha) / gamma(beta), 0],
        [0, 0, 0, gamma(beta - alpha) / gamma(beta + 3 * alpha), -gamma(beta - alpha) / gamma(beta + 2 * alpha), gamma(beta - alpha) / gamma(beta + alpha), -gamma(beta - alpha) / gamma(beta)],
        [1, 0, 0, 0, -1, gamma(beta - alpha) / gamma(beta - 2 * alpha), -gamma(beta - alpha) / gamma(beta - 3 * alpha)],
        [0, 1, 0, 0, 0, -1, gamma(beta - alpha) / gamma(beta - 2 * alpha)],
        [0, 0, 1, 0, 0, 0, -1]
    ])
    
    b = np.array([0, 0, 0, -1, -gamma(beta - alpha) / gamma(beta - 4 * alpha), gamma(beta - alpha) / gamma(beta - 3 * alpha), -gamma(beta - alpha) / gamma(beta - 2 * alpha)])
    
    return np.linalg.solve(A, b)


def system_54aa(alpha):
    A = np.array([
        [1, 0, gamma(-alpha) / gamma(alpha), 0, 0, 0],
        [0, 1, -gamma(-alpha) / gamma(2 * alpha), gamma(-alpha) / gamma(alpha), 0, 0],
        [0, 0, gamma(-alpha) / gamma(3 * alpha), -gamma(-alpha) / gamma(2 * alpha), gamma(-alpha) / gamma(alpha), 0],
        [0, 0, 0, -1, -gamma(-alpha) / gamma(-2 * alpha), 0],
        [1, 0, 0, 0, -1, gamma(-alpha) / gamma(-2 * alpha)],
        [0, 1, 0, 0, 0, -1]
    ])
    
    b = np.array([0, 0, -1, 0, 0, -gamma(-alpha) / gamma(-2 * alpha)])
    
    return np.linalg.solve(A, b)


def R_alpha_beta_5_4(x, alpha, beta):
    if beta > alpha:
        coefficients = system_54ab(alpha, beta)
        p = coefficients[:3]
        q = coefficients[3:]
        R_val = (1 / gamma(beta - alpha)) * (p[0] + p[1] * (-x) + p[2] * (-x)**2 + (-x)**3) / (q[0] + q[1] * (-x) + q[2] * (-x)**2 + q[3] * (-x)**3 + x**4)
    else:
        coefficients = system_54aa(alpha)
        p_hat = coefficients[:2]
        q_hat = coefficients[2:]
        R_val = (-1 / gamma(-alpha)) * (p_hat[0] + p_hat[1] * (-x) + (-x)**2) / (q_hat[0] + q_hat[1] * (-x) + q_hat[2] * (-x)**2 + q_hat[3] * (-x)**3 + (-x)**4)
    
    return R_val

