import numpy as np
import math

# Error function
def createRandomError(n, std):
    return np.random.normal(loc=1, scale=std, size=n)

# Maxwell Model
def MaxwellModel(G_s, eta_s, omega, errorInserted=0):
    """
    Compute the Maxwell model response

    Parameters
    ----------
    G_s : float
        Shear modulus (Pa). Subindex "s" implies that the element is connected in serie.
    eta_s : float
        Viscosity (Pa s). Subindex "s" implies that the element is connected in serie.
    omega : numpy
        Array of angular frequency values (rad/s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """ 
    tau_c = eta_s / G_s
    omega_tau_c = omega * tau_c
    omega_tau_c_squared = omega_tau_c ** 2

    # Compute the storage and loss modulus
    G_prime = G_s * (omega_tau_c_squared / (1 + omega_tau_c_squared))
    G_double_prime = G_s * (omega_tau_c / (1 + omega_tau_c_squared))

    # Apply error if specified
    if errorInserted != 0:
        G_prime *= (1 + errorInserted * np.random.randn(*omega.shape))
        G_double_prime *= (1 + errorInserted * np.random.randn(*omega.shape))

    return np.concatenate([G_prime, G_double_prime])

# Spring Pot Model
def SpringPot(V, alpha, omega, errorInserted=0):
    """
    Compute the SpringPot model response

    Parameters
    ----------
    V: float.
       Quasi-modulus (Pa*s^alpha)
    alpha: float 
        Parameter between [0, 1] (dimensionless).
    omega : numpy
        Array of angular frequency values (rad/s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """ 
    # Compute the storage and loss modulus
    G_prime = V * np.power(omega, alpha) * np.cos(np.pi * alpha / 2)
    G_double_prime = V * np.power(omega, alpha) * np.sin(np.pi * alpha / 2)
    
    # Apply error if specified
    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)
        
    return np.concatenate([G_prime, G_double_prime])

# Fractional Maxwell Gel Model
def FractionalMaxwellGel(G_s, V, alpha, omega, errorInserted=0):
    """
    Compute the Fractional Maxwell Gel model response

    Parameters
    ----------
    V : float
        Quasi-modulus (Pa*s^alpha).
    G_s : float
        Shear modulus (Pa). Subindex "s" implies that the element is connected in serie.
    alpha : float
        Fractional order parameter between 0 and 1 (dimensionless).
    omega : numpy
        Array of angular frequency values (rad/s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """ 
    tau_c = (V / G_s) ** (1 / alpha)
    G_c = V * (tau_c ** (-alpha))
    omega_tau_pow_a = np.power(np.multiply(omega, tau_c), alpha)
    omega_tau_pow_2a = np.power(np.multiply(omega, tau_c), 2 * alpha)
    cos_pi_alpha_d_2 = np.cos(np.pi * alpha / 2)
    sin_pi_alpha_d_2 = np.sin(np.pi * alpha / 2)
    
    # Compute the storage and loss modulus
    numerator_gp = omega_tau_pow_2a + np.multiply(omega_tau_pow_a, cos_pi_alpha_d_2)
    denumerator_gp = 1 + omega_tau_pow_2a + 2 * omega_tau_pow_a * cos_pi_alpha_d_2
    G_prime = G_c * numerator_gp / denumerator_gp

    numerator_gpp = omega_tau_pow_a * sin_pi_alpha_d_2
    denumerator_gpp = 1 + omega_tau_pow_2a + 2 * omega_tau_pow_a * cos_pi_alpha_d_2
    G_double_prime = G_c * numerator_gpp / denumerator_gpp
    
    # Apply error if specified
    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)

    return np.concatenate([G_prime, G_double_prime])

# Fractional Maxwell Liquid Model
def FractionalMaxwellLiquid(eta_s, G, beta, omega, errorInserted=0):
    """
    Compute the Fractional Maxwell Liquid model response

    Parameters
    ----------
    G : float
        Quasi-modulus (Pa*s^beta).
    eta_s: float.
        viscosity (Pa*s). Subindex "s" implies that the element is connected in serie.
    beta : float
        Fractional order parameter between 0 and 1 (dimensionless).
    omega : numpy
        Array of angular frequency values (rad/s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """ 
    tau_c = (eta_s / G) ** (1 / (1 - beta))
    G_c = eta_s * (tau_c ** (-1))
    omega_tau_c = omega * tau_c
    omega_tau_c_beta = omega_tau_c ** (2 - beta)
    omega_tau_c_1_minus_beta = omega_tau_c ** (1 - beta)
    omega_tau_c_2_1_minus_beta = omega_tau_c ** (2 * (1 - beta))

    cos_beta_half = np.cos(np.pi * beta / 2)
    cos_1_minus_beta_half = np.cos(np.pi * (1 - beta) / 2)
    sin_beta_half = np.sin(np.pi * beta / 2)

    numerator_gp = omega_tau_c_beta * cos_beta_half
    numerator_gpp = omega_tau_c + omega_tau_c_beta * sin_beta_half

    denominator = 1 + omega_tau_c_2_1_minus_beta + 2 * omega_tau_c_1_minus_beta * cos_1_minus_beta_half

    # Compute the storage and loss modulus
    G_prime = G_c * (numerator_gp / denominator)
    G_double_prime = G_c * (numerator_gpp / denominator)

    # Apply error if specified
    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)

    return np.concatenate([G_prime, G_double_prime])

# Fractional Maxwell Model
def FractionalMaxwellModel(G, V, alpha, beta, omega, errorInserted=0):
    """
    Compute the Fractional Maxwell model response

    Parameters
    ----------
    G: float.
        Quasi-modulus (Pa*s^beta)
    V: float.
        Quasi-modulus (Pa*s^alpha)
    alpha: float 
        Parameter between [0, 1] (dimensionless).
    beta: float 
        Parameter between [0, 1] (dimensionless).
    omega : numpy
        Array of angular frequency values (rad/s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """ 
    tau_c = (V / G) ** (1 / (alpha - beta))
    G_c = V * (tau_c ** (-alpha))
    omega_tau_c = omega * tau_c
    omega_tau_c_alpha = omega_tau_c ** alpha
    omega_tau_c_2alpha_minus_beta = omega_tau_c ** (2 * alpha - beta)
    omega_tau_c_alpha_minus_beta = omega_tau_c ** (alpha - beta)
    omega_tau_c_2_alpha_minus_beta = omega_tau_c ** (2 * (alpha - beta))

    cos_alpha_half = np.cos(np.pi * alpha / 2)
    cos_beta_half = np.cos(np.pi * beta / 2)
    cos_alpha_minus_beta_half = np.cos(np.pi * (alpha - beta) / 2)
    sin_alpha_half = np.sin(np.pi * alpha / 2)
    sin_beta_half = np.sin(np.pi * beta / 2)

    numerator_G_prime = omega_tau_c_alpha * cos_alpha_half + omega_tau_c_2alpha_minus_beta * cos_beta_half
    numerator_G_double_prime = omega_tau_c_alpha * sin_alpha_half + omega_tau_c_2alpha_minus_beta * sin_beta_half

    denominator = 1 + omega_tau_c_2_alpha_minus_beta + 2 * omega_tau_c_alpha_minus_beta * cos_alpha_minus_beta_half

    # Compute the storage and loss modulus
    G_prime = G_c * (numerator_G_prime / denominator)
    G_double_prime = G_c * (numerator_G_double_prime / denominator)

    # Apply error if specified    
    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)

    return np.concatenate([G_prime, G_double_prime])

# Fractional Kelvin-VoigtS Model
def FractionalKelvinVoigtS(G_p, V, alpha, omega, errorInserted=0):
    """
    Compute the Fractional Kelvin-VoigtS model response

    Parameters
    ----------
    G_p: float.
        shear modulus (Pa). Subindex "p" implies that the element is connected in parallel.
    V: float
        Quasi-modulus (Pa*s^alpha)
    alpha: float 
        Parameter between [0, 1] (dimensionless).
    omega : numpy
        Array of angular frequency values (rad/s).
    errorInserted: float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """ 
    omega_alpha = omega ** alpha  # Computing omega^alpha

    cos_alpha_half = np.cos(np.pi * alpha / 2)  # Computing cos(pi * alpha / 2)
    sin_alpha_half = np.sin(np.pi * alpha / 2)  # Computing sin(pi * alpha / 2)

    G_prime = V * omega_alpha * cos_alpha_half + G_p  # Computing storage modulus
    G_double_prime = V * omega_alpha * sin_alpha_half  # Computing loss modulus

    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)  # Applying error to storage modulus
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)  # Applying error to loss modulus

    return np.concatenate([G_prime, G_double_prime])
        
# Fractional Kelvin-Voigt model (springpot-dashpot) 
def FractionalKelvinVoigtD(G, eta_p, beta, omega, errorInserted=0):
    """
    Compute the Fractional Kelvin-VoigtD model response

    Parameters
    ----------
    G: float
        Quasi-modulus (Pa*s^beta)
    eta_p: float
        Viscosity (Pa*s)
    beta: float 
        Parameter between [0, 1] (dimensionless).
    omega : numpy
        Array of angular frequency values (rad/s).
    errorInserted: float, optional
        Optional error to insert into the model (default is 0).


    Returns  
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """     
    omega_beta = omega ** beta

    cos_beta_half = np.cos(np.pi * beta / 2)
    sin_beta_half = np.sin(np.pi * beta / 2)

    # Compute the storage and loss modulus
    G_prime = G * omega_beta * cos_beta_half
    G_double_prime = eta * omega + G * omega_beta * sin_beta_half

    # Apply error if specified    
    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)
        
    return np.concatenate([G_prime, G_double_prime])        
        
# Fractional Kelvin-Voigt model (springpot-springpot)        
def FractionalKelvinVoigtModel(G, V, alpha, beta, omega, errorInserted=0):
    """
    Compute the Fractional Kelvin-Voigt model
    
    Parameters
    ----------
    G: float.
        Quasi-modulus (Pa*s^beta)
    V: float.
        Quasi-modulus (Pa*s^alpha)
    alpha: float 
        Parameter between [0, 1] (dimensionless).
    beta: float 
        Parameter between [0, 1] (dimensionless).
    omega : numpy
        Array of angular frequency values (rad/s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns  
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """    
    omega_alpha = omega ** alpha
    omega_beta = omega ** beta

    cos_alpha_half = np.cos(np.pi * alpha / 2)
    cos_beta_half = np.cos(np.pi * beta / 2)
    sin_alpha_half = np.sin(np.pi * alpha / 2)
    sin_beta_half = np.sin(np.pi * beta / 2)

    G_prime = V * omega_alpha * cos_alpha_half + G * omega_beta * cos_beta_half
    G_double_prime = V * omega_alpha * sin_alpha_half + G * omega_beta * sin_beta_half
    
    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)

    return np.concatenate([G_prime, G_double_prime])        
        
        
# Zener model
def ZenerModel(G_p, G_s, eta_s, omega, errorInserted=0):
    """
    Compute the Zener model response for given shear moduli, viscosity, 
    and omega array with optional error insertion.

    Parameters
    ----------
    G_s: float.
        shear modulus (Pa). Subindex "s" implies that the element is connected in serie.
    G_p: float.
        shear modulus (Pa). Subindex "p" implies that the element is connected in parallel.
    eta_s: float.
        viscosity (Pa*s). Subindex "s" implies that the element is connected in serie.
    omega : numpy
        Array of angular frequency values (rad/s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns  
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """    
    term1 = np.multiply(np.power(np.divide(omega * eta_s, G_s), 2), G_s)
    term2 = 1 + np.power(np.divide(omega * eta_s, G_s), 2)
    term3 = np.multiply(np.power(np.divide(omega * eta_s, G_s), 1), G_s)
    term4 = 1 + np.power(np.divide(omega * eta_s, G_s), 2)

    G_prime = G_p + np.divide(term1, term2)
    G_double_prime = np.divide(term3, term4)
    
    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)

    return np.concatenate([G_prime, G_double_prime])        
        
# Fractional Zener model (springpot-spring --- spring)      
def FractionalZenerSolidS(G_p, G_s, V, alpha, omega, errorInserted=0):
    """
    Compute the relaxation modulus for the Fractional Zener Solid-S model for given quasi-moduli, 
    fractional order, and time array.
    
    Parameters
    ----------
    G_p: float
        Quasi-modulus related to the viscous component (Pa*s^alpha)
    G_s: float
        Quasi-modulus related to the solid component (Pa)
    V: float
        Viscosity related parameter (Pa*s)
    alpha: float
        Fractional parameter between [0, 1] (dimensionless)
    omega : numpy
        Array of angular frequency values (rad/s).
    errorInserted: float, optional
        Optional error to insert into the model (default is 0).
    
    Returns  
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """    
    # Compute intermediate terms
    V_omega_alpha = V * omega**alpha
    cos_pi_alpha = np.cos(np.pi * alpha / 2)
    sin_pi_alpha = np.sin(np.pi * alpha / 2)
    
    # Numerators for G' and G''
    G_prime_numerator = ( (G_s**2 * V_omega_alpha * cos_pi_alpha) + (V_omega_alpha**2 * G_s) )
    G_double_prime_numerator = ( (G_s**2 * V_omega_alpha * sin_pi_alpha) )
    
    # Denominator for both G' and G''
    denominator = ( (V_omega_alpha**2) + (G_s**2) + (2 * V_omega_alpha * G_s * cos_pi_alpha) )
    
    # Storage modulus G' and Loss modulus G''
    G_prime = G_p + G_prime_numerator / denominator
    G_double_prime = G_double_prime_numerator / denominator

    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)
    
    return np.concatenate([G_prime, G_double_prime]) 

# Fractional Zener model (springpot-dashpot --- spring)  
def FractionalZenerLiquidS(G_p, G, eta_s, beta, omega, errorInserted=0):
    """
    Compute the storage modulus (G') and loss modulus (G'') for the fractional Zener liquid model.
    
    Parameters
    ----------
    G_p : float
        Elastic modulus at zero frequency.
    G : float
        Modulus related to the spring and dashpot interaction.
    eta_s : float
        Viscosity.
    beta : float
        Fractional parameter of the model.
    omega : numpy.ndarray
        Angular frequency array (rad/s).
    errorInserted: float, optional
        Optional error to insert into the model (default is 0).
    
    Returns  
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """    
    # Compute intermediate terms
    omega_beta = omega**beta
    eta_omega = eta_s * omega
    cos_pi_beta = np.cos(np.pi * beta / 2)
    sin_pi_beta = np.sin(np.pi * beta / 2)
    cos_pi_1_minus_beta = np.cos(np.pi * (1 - beta) / 2)
    
    # Numerators for G' and G''
    G_prime_numerator = (eta_omega**2) * G * omega_beta * cos_pi_beta
    G_double_prime_numerator = (G * omega_beta)**2 * eta_omega + (eta_omega**2) * G * omega_beta * sin_pi_beta
    
    # Denominator for both G' and G''
    denominator = (eta_omega**2) + (G * omega_beta)**2 + 2 * eta_omega * G * omega_beta * cos_pi_1_minus_beta
    
    # Storage modulus G' and Loss modulus G''
    G_prime = G_p + G_prime_numerator / denominator
    G_double_prime = G_double_prime_numerator / denominator
    
    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)
    
    return np.concatenate([G_prime, G_double_prime]) 
        

# Fractional Zener model (springpot-dashpot --- dashpot)     
def FractionalZenerLiquidD(eta_p, eta_s, G, beta, omega, errorInserted=0):
    """
    Compute the storage modulus (G') and loss modulus (G'') for the Fractional Zener Liquid-D model.
    
    Parameters
    ----------
    eta_p: float
        Viscous parameter related to the pure viscous component (Pa*s).
    eta_s: float
        Viscosity parameter (Pa*s).
    G: float
        Modulus parameter related to the solid component (Pa).
    beta: float
        Fractional parameter between [0, 1] (dimensionless).
    omega: numpy.ndarray
        Array of angular frequency values (rad/s).
    errorInserted: float, optional
        Optional error to insert into the model (default is 0).
    
    Returns  
    -------
    G_prime : numpy.ndarray
        Storage modulus at each omega point.
    G_double_prime : numpy.ndarray
        Loss modulus at each omega point.
    """  
    # Compute intermediate terms
    eta_s_omega = eta_s * omega
    G_omega_beta = G * omega**beta
    cos_pi_beta = np.cos(np.pi * beta / 2)
    sin_pi_beta = np.sin(np.pi * beta / 2)
    
    # Numerators for G' and G''
    G_prime_numerator = eta_s**2 * G * omega**(beta + 2) * cos_pi_beta
    G_double_prime_numerator = (
        G**2 * eta_s * omega**(2 * beta + 1)
        + eta_s**2 * G * omega**(beta + 2) * sin_pi_beta
    )
    
    # Denominator for both G' and G''
    denominator = (
        (eta_s_omega)**2 
        + (G_omega_beta)**2 
        + 2 * eta_s * G * omega**(beta + 1) * sin_pi_beta
    )
    
    # Storage modulus G' and Loss modulus G''
    G_prime = G_prime_numerator / denominator
    G_double_prime = eta_p * omega + G_double_prime_numerator / denominator

    if errorInserted != 0:
        G_prime *= createRandomError(omega.shape[0], errorInserted)
        G_double_prime *= createRandomError(omega.shape[0], errorInserted)
    
    return np.concatenate([G_prime, G_double_prime]) 

















     
        
        
        
        
        
        
        
    
    
