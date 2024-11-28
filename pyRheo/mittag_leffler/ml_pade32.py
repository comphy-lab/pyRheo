import numpy as np
import math
from scipy.special import gamma

# Pade for Mittag-Leffler m=3, n=2 according to C. Zeng and Y. Q. Chen, Fractional Calculus and Applied Analysis, 2015, 18, 1492–1506

def coeff_ab(alpha, beta):
    c_ab = 1 / (gamma(beta + alpha) * gamma(beta - alpha) - gamma(beta)**2)
    
    p1 = c_ab * (gamma(beta) * gamma(beta + alpha) - (gamma(beta + alpha) * gamma(beta - alpha)**2) / gamma(beta - 2*alpha))
    
    q0 = c_ab * ((gamma(beta)**2 * gamma(beta + alpha)) / gamma(beta - alpha) -
                 (gamma(beta) * gamma(beta + alpha) * gamma(beta - alpha)) / gamma(beta - 2*alpha))
    
    q1 = c_ab * (gamma(beta) * gamma(beta + alpha) -
                 (gamma(beta)**2 * gamma(beta - alpha)) / gamma(beta - 2*alpha))
    
    return p1, q0, q1

def R_alpha_beta_3_2(x, alpha, beta):
    if beta > alpha:
        p1, q0, q1 = coeff_ab(alpha, beta)
        numerator = p1 + (-x)
        denominator = q0 + q1 * (-x) + (-x)**2
        R_val = (1 / gamma(beta - alpha)) * (numerator / denominator)
    elif beta == alpha:
        a = alpha
        numerator = a
        denominator = gamma(1 + a) + (2 * gamma(1 - a)**2 / gamma(1 - 2*a)) * (-x) + gamma(1 - a) * (-x)**2
        R_val = numerator / denominator
    else:
        raise ValueError("beta should be greater than or equal to alpha")
    
    return R_val
