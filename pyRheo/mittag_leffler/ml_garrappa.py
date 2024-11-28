import numpy as np
import scipy.special as spec

# Mittag-Leffler function Garrappa algorithm

def E_alpha_beta(z, alpha, beta=1, gamma=1):
    """
    Evaluation of the Mittag-Leffler (ML) function with 1, 2 or 3 parameters
    by means of the OPC algorithm.

    Parameters
    ----------
    z : array_like
        The argument(s) of the function.
    alpha : float
        The first parameter must be a real and positive scalar.
    beta : float, optional
        The second parameter, default is 1.
    gamma : float, optional
        The third parameter, default is 1.

    Returns
    -------
    E : ndarray
        Values of the Mittag-Leffler function.

    References
    ----------
    R. Garrappa, Numerical evaluation of two and three parameter
    Mittag-Leffler functions, SIAM Journal of Numerical Analysis, 2015,
    53(3), 1350-1369
    """
    
    # Check parameter constraints
    if alpha <= 0 or gamma <= 0 or not np.isreal(alpha) or not np.isreal(beta) or not np.isreal(gamma):
        raise ValueError("Parameters ALPHA and GAMMA must be real and positive. The parameter BETA must be real.")
        
    if gamma != 1:
        if alpha >= 1:
            raise ValueError("With the three-parameter Mittag-Leffler function, ALPHA must satisfy 0 < ALPHA < 1.")
        if np.any(np.abs(np.angle(z[np.abs(z) > np.finfo(float).eps])) <= alpha * np.pi):
            raise ValueError("With the three-parameter Mittag-Leffler function, this code works only when |Arg(z)|>alpha*pi.")

    log_epsilon = np.log(10**-15)
    
    E = np.zeros_like(z, dtype=complex)
    
    for idx, zk in np.ndenumerate(z):
        if np.abs(zk) < 1.0e-15:
            E[idx] = 1 / spec.gamma(beta)
        else:
            E[idx] = LTInversion(1, zk, alpha, beta, gamma, log_epsilon)

    return E if np.iscomplexobj(E) else E.real


def LTInversion(t, lam, alpha, beta, gamma, log_epsilon):
    theta = np.angle(lam)
    kmin = int(np.ceil(-alpha/2 - theta/2/np.pi))
    kmax = int(np.floor(alpha/2 - theta/2/np.pi))
    k_vett = np.arange(kmin, kmax + 1)
    s_star = np.abs(lam)**(1/alpha) * np.exp(1j * (theta + 2 * k_vett * np.pi) / alpha)
    phi_s_star = (s_star.real + np.abs(s_star)) / 2
    index_s_star = np.argsort(phi_s_star)
    phi_s_star = phi_s_star[index_s_star]
    s_star = s_star[index_s_star]
    s_star = np.concatenate(([0], s_star[phi_s_star > 1.0e-15]))
    phi_s_star = np.concatenate(([0], phi_s_star[phi_s_star > 1.0e-15]))
    J1 = len(s_star)
    J = J1 - 1
    p = np.concatenate(([max(0, -2 * (alpha * gamma - beta + 1))], np.ones(J) * gamma))
    q = np.concatenate((np.ones(J) * gamma, [np.inf]))
    phi_s_star = np.concatenate((phi_s_star, [np.inf]))
    admissible_regions = np.where(
        (phi_s_star[:-1] < (log_epsilon - np.log(np.finfo(float).eps)) / t) &
        (phi_s_star[:-1] < phi_s_star[1:])
    )[0]

    if len(admissible_regions) == 0:
        raise ValueError("No admissible regions found.")
    
    JJ1 = admissible_regions[-1]
    mu_vett = np.full(JJ1 + 1, np.inf)
    N_vett = np.full(JJ1 + 1, np.inf)
    h_vett = np.full(JJ1 + 1, np.inf)
    
    while True:
        for j1 in admissible_regions:
            if j1 < J1:
                muj, hj, Nj = OptimalParam_RB(t, phi_s_star[j1], phi_s_star[j1 + 1], p[j1], q[j1], log_epsilon)
            else:
                muj, hj, Nj = OptimalParam_RU(t, phi_s_star[j1], p[j1], log_epsilon)
            mu_vett[j1], h_vett[j1], N_vett[j1] = muj, hj, Nj

        if np.min(N_vett) > 200:
            log_epsilon += np.log(10)
        else:
            break
    
    N = np.min(N_vett)
    iN = np.argmin(N_vett)
    mu = mu_vett[iN]
    h = h_vett[iN]
    
    k = np.arange(-N, N + 1)
    u = h * k
    z = mu * (1j * u + 1)**2
    zd = -2 * mu * u + 2 * mu * 1j
    zexp = np.exp(z * t)
    F = z**(alpha * gamma - beta) / (z**alpha - lam)**gamma * zd
    S = zexp * F
    Integral = h * np.sum(S) / (2 * np.pi * 1j)
    
    ss_star = s_star[iN + 1:]
    Residues = np.sum(1 / alpha * ss_star**(1 - beta) * np.exp(t * ss_star))
    
    E = Integral + Residues
    return E.real if np.isreal(lam) else E

def OptimalParam_RB(t, phi_s_star_j, phi_s_star_j1, pj, qj, log_epsilon):
    log_eps = -36.043653389117154
    fac = 1.01
    conservative_error_analysis = False
    f_max = np.exp(log_epsilon - log_eps)
    sq_phi_star_j = np.sqrt(phi_s_star_j)
    threshold = 2 * np.sqrt((log_epsilon - log_eps) / t)
    sq_phi_star_j1 = min(np.sqrt(phi_s_star_j1), threshold - sq_phi_star_j)
    adm_region = False
    
    if pj < 1.0e-14 and qj < 1.0e-14:
        sq_phibar_star_j = sq_phi_star_j
        sq_phibar_star_j1 = sq_phi_star_j1
        adm_region = True
    if pj < 1.0e-14 and qj >= 1.0e-14:
        sq_phibar_star_j = sq_phi_star_j
        if sq_phi_star_j > 0:
            f_min = fac * (sq_phi_star_j / (sq_phi_star_j1 - sq_phi_star_j)) ** qj
        else:
            f_min = fac
        if f_min < f_max:
            f_bar = f_min + f_min / f_max * (f_max - f_min)
            fq = f_bar ** (-1 / qj)
            sq_phibar_star_j1 = (2 * sq_phi_star_j1 - fq * sq_phi_star_j) / (2 + fq)
            adm_region = True
    if pj >= 1.0e-14 and qj < 1.0e-14:
        sq_phibar_star_j1 = sq_phi_star_j1
        f_min = fac * (sq_phi_star_j1 / (sq_phi_star_j1 - sq_phi_star_j)) ** pj
        if f_min < f_max:
            f_bar = f_min + f_min / f_max * (f_max - f_min)
            fp = f_bar ** (-1 / pj)
            sq_phibar_star_j = (2 * sq_phi_star_j + fp * sq_phi_star_j1) / (2 - fp)
            adm_region = True
    if pj >= 1.0e-14 and qj >= 1.0e-14:
        f_min = fac * (sq_phi_star_j + sq_phi_star_j1) / (sq_phi_star_j1 - sq_phi_star_j) ** max(pj, qj)
        if f_min < f_max:
            f_min = max(f_min, 1.5)
            f_bar = f_min + f_min / f_max * (f_max - f_min)
            fp = f_bar ** (-1 / pj)
            fq = f_bar ** (-1 / qj)
            if not conservative_error_analysis:
                w = -phi_s_star_j1 * t / log_epsilon
            else:
                w = -2 * phi_s_star_j1 * t / (log_epsilon - phi_s_star_j1 * t)
            den = 2 + w - (1 + w) * fp + fq
            sq_phibar_star_j = ((2 + w + fq) * sq_phi_star_j + fp * sq_phi_star_j1) / den
            sq_phibar_star_j1 = (-(1 + w) * fq * sq_phi_star_j + (2 + w - (1 + w) * fp) * sq_phi_star_j1) / den
            adm_region = True
    
    if adm_region:
        log_epsilon -= np.log(f_bar)
        if not conservative_error_analysis:
            w = -sq_phibar_star_j1**2 * t / log_epsilon
        else:
            w = -2 * sq_phibar_star_j1**2 * t / (log_epsilon - sq_phibar_star_j1**2 * t)
        muj = (((1 + w) * sq_phibar_star_j + sq_phibar_star_j1) / (2 + w))**2
        hj = -2 * np.pi / log_epsilon * (sq_phibar_star_j1 - sq_phibar_star_j) / ((1 + w) * sq_phibar_star_j + sq_phibar_star_j1)
        Nj = int(np.ceil(np.sqrt(1 - log_epsilon / t / muj) / hj))
    else:
        muj, hj, Nj = 0, 0, np.inf
    
    return muj, hj, Nj

def OptimalParam_RU(t, phi_s_star_j, pj, log_epsilon):
    sq_phi_s_star_j = np.sqrt(phi_s_star_j)
    phibar_star_j = phi_s_star_j * 1.01 if phi_s_star_j > 0 else 0.01
    sq_phibar_star_j = np.sqrt(phibar_star_j)
    f_min, f_max, f_tar = 1, 10, 5
    stop = False
    
    while not stop:
        phi_t = phibar_star_j * t
        log_eps_phi_t = log_epsilon / phi_t
        Nj = int(np.ceil(phi_t / np.pi * (1 - 3 * log_eps_phi_t / 2 + np.sqrt(1 - 2 * log_eps_phi_t))))
        A = np.pi * Nj / phi_t
        sq_muj = sq_phibar_star_j * np.abs(4 - A) / np.abs(7 - np.sqrt(1 + 12 * A))
        fbar = ((sq_phibar_star_j - sq_phi_s_star_j) / sq_muj)**(-pj)
        stop = (pj < 1.0e-14) or (f_min < fbar < f_max)
        if not stop:
            sq_phibar_star_j = f_tar**(-1 / pj) * sq_muj + sq_phi_s_star_j
            phibar_star_j = sq_phibar_star_j**2
    
    muj = sq_muj**2
    hj = (-3 * A - 2 + 2 * np.sqrt(1 + 12 * A)) / (4 - A) / Nj
    log_eps = np.log(np.finfo(float).eps)
    threshold = (log_epsilon - log_eps) / t
    
    if muj > threshold:
        if np.abs(pj) < 1.0e-14:
            Q = 0
        else:
            Q = f_tar**(-1 / pj) * np.sqrt(muj)
        phibar_star_j = (Q + np.sqrt(phi_s_star_j))**2
        if phibar_star_j < threshold:
            w = np.sqrt(log_eps / (log_eps - log_epsilon))
            u = np.sqrt(-phibar_star_j * t / log_eps)
            muj, hj, Nj = threshold, np.sqrt(log_eps / (log_eps - log_epsilon)) / Nj, np.inf
    
    return muj, hj, Nj
