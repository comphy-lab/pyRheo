# For the training we use the functions with the relaxation/retardation time as parameter. This helps to avoid numerical instabilities.

# Import required packages
import numpy as np
import math
from scipy.special import gamma
import mpmath

# Import Mittag-Leffler implementations from their .py files
from pyRheo.mittag_leffler.ml_pade32 import R_alpha_beta_3_2
from pyRheo.mittag_leffler.ml_pade54 import R_alpha_beta_5_4
from pyRheo.mittag_leffler.ml_pade63 import R_alpha_beta_6_3
from pyRheo.mittag_leffler.ml_pade72 import R_alpha_beta_7_2
from pyRheo.mittag_leffler.ml_garrappa import E_alpha_beta


# Error function
def createRandomError(n, std):
    return np.random.normal(loc=(1), scale=(std), size=(n,))

# Creep models


# Maxwell model
def MaxwellModel(G_s, eta_s, t, errorInserted=0):
    """
    Compute the Maxwell model response

    Parameters
    ----------
    G_s : float
        Shear modulus (Pa). Subindex "s" implies that the element is connected in serie.
    eta_s : float
        Viscosity (Pa s). Subindex "s" implies that the element is connected in serie.
    t : numpy
        Array of time values (s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    ndarray
        The creep compliance response at each time point in `t`.
    """ 
    tau_c = eta_s / G_s
    result = np.add(np.divide(t, eta_s), 1/G_s)
    
    if errorInserted != 0:
        error = createRandomError(t.shape[0], errorInserted)
        result = np.multiply(result,error)
        
    return result
    
# SpringPot model
def SpringPot(V, alpha, t, errorInserted=0):
    """
    Compute the SpringPot model response

    Parameters
    ----------
    V: float.
       Quasi-modulus (Pa*s^alpha)
    alpha: float 
        Parameter between [0, 1] (dimensionless).
    t : numpy
        Array of time values (s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    ndarray
        The creep compliance response at each time point in `t`.
    """ 
    numerator = np.power(t, alpha)
    denumerator = V*gamma(1+alpha)

    result = np.divide(numerator, denumerator)    
    if errorInserted != 0:
        error = createRandomError(t.shape[0], errorInserted)
        result = np.multiply(result,error)
        
    return result

# Fractional Maxwell Gel model (springpot-spring)
def FractionalMaxwellGel(V, G_s, alpha, t, errorInserted=0):
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
    t : numpy.ndarray
        Array of time values (s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    numpy.ndarray
        The creep compliance response at each time point in `t`.
    """
    
    gamma_1_plus_alpha = gamma(1 + alpha)
    term1 = np.divide(np.power(t, alpha), V * gamma_1_plus_alpha)
    term2 = 1 / G_s
    result = np.add(term1, term2)
    
    if errorInserted != 0:
        error = createRandomError(t.shape[0], errorInserted)
        result = np.multiply(result, error)
        
    return result


# Fractional Maxwell Liquid model (springpot-dashpot)
def FractionalMaxwellLiquid(G, eta_s, beta, t, errorInserted=0):
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
    t : numpy.ndarray
        Array of time values (s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    numpy.ndarray
        The creep compliance response at each time point in `t`.
    """
    
    gamma_1_plus_beta = gamma(1 + beta)
    term1 = np.divide(t, eta_s)
    term2 = np.divide(np.power(t, beta), G * gamma_1_plus_beta)
    result = np.add(term1, term2)
    
    if errorInserted != 0:
        error = createRandomError(t.shape[0], errorInserted)
        result = np.multiply(result, error)
        
    return result

# Fractional Maxwell model (springpot-springpot)
def FractionalMaxwellModel(G, V, alpha, beta, t, errorInserted = 0):
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
    t : numpy
        Array of time values (s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).

    Returns
    -------
    ndarray
        The creep compliance response at each time point in `t`.
    """
    gamma_1_plus_alpha = gamma(1+alpha)
    gamma_1_plus_beta = gamma(1+beta)
    term1 = np.divide(np.power(t, alpha), V*gamma_1_plus_alpha)
    term2 = np.divide(np.power(t, beta), G*gamma_1_plus_beta)
    result = np.add(term1, term2)
    
    if errorInserted != 0:
        error = createRandomError(t.shape[0], errorInserted)
        result = np.multiply(result, error)
        
    return result

# Fractional Kelvin-Voigt model (spring-springpot)
def FractionalKelvinVoigtS(V, tau_c, alpha, t, errorInserted=0, mittag_leffler_type="Pade32"):
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
    t: numpy.ndarray
        Array of time values (s).
    errorInserted: float, optional
        Optional error to insert into the model (default is 0).
    mittag_leffler_type : str, optional
        Type of function to use ("Pade" or "Garrappa")

    Returns
    -------
    ndarray
        The creep compliance at each time point in `t`.
    """
    #tau_c = (V / G_p)**(1 / alpha) 
    a = alpha 
    b = 1 + alpha 
    z = -np.power(np.divide(t, tau_c), alpha)

    if mittag_leffler_type == "Pade32":
        response_func = R_alpha_beta_3_2
    elif mittag_leffler_type == "Pade54":
        response_func = R_alpha_beta_5_4
    elif mittag_leffler_type == "Pade63":
        response_func = R_alpha_beta_6_3
    elif mittag_leffler_type == "Pade72":
        response_func = R_alpha_beta_7_2
    elif mittag_leffler_type == "Garrappa":
        response_func = E_alpha_beta
    else:
        raise ValueError("mittag_leffler_type must be either 'Pade' or 'Garrappa'")

    result = np.multiply(np.divide(np.power(t, alpha), V), response_func(z, a, b))  

    if errorInserted != 0:
        error = createRandomError(t.shape[0], errorInserted)  
        result = np.multiply(result, error)  

    return result  
    
# Fractional Kelvin-Voigt model (springpot-dashpot) 
def FractionalKelvinVoigtD(tau_c, eta_p, beta, t, errorInserted=0, mittag_leffler_type="Pade32"):
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
    t: numpy.ndarray
        Array of time values (s).
    errorInserted: float, optional
        Optional error to insert into the model (default is 0).
    mittag_leffler_type : str, optional
        Type of function to use ("Pade" or "Garrappa")

    Returns
    -------
    ndarray
        The creep compliance at each time point in `t`.
    """
    #tau_c = (eta_p / G)**(1 / (1 - beta))  
    a = 1 - beta  
    b = 1 + 1 
    z = -np.power(np.divide(t, tau_c), 1 - beta)

    if mittag_leffler_type == "Pade32":
        response_func = R_alpha_beta_3_2
    elif mittag_leffler_type == "Pade54":
        response_func = R_alpha_beta_5_4
    elif mittag_leffler_type == "Pade63":
        response_func = R_alpha_beta_6_3
    elif mittag_leffler_type == "Pade72":
        response_func = R_alpha_beta_7_2
    elif mittag_leffler_type == "Garrappa":
        response_func = E_alpha_beta
    else:
        raise ValueError("mittag_leffler_type must be either 'Pade' or 'Garrappa'")

    result = np.multiply(np.divide(t, eta_p), response_func(z, a, b)) 

    if errorInserted != 0:
        error = createRandomError(t.shape[0], errorInserted)  
        result = np.multiply(result, error)  

    return result  # Returning the result    
    
# Fractional Kelvin-Voigt model (springpot-springpot)
def FractionalKelvinVoigtModel(tau_c, V, alpha, beta, t, errorInserted=0, mittag_leffler_type="Pade32"):
    """
    Compute the Fractional Kelvin-Voigt model response

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
    t : numpy
        Array of time values (s).
    errorInserted : float, optional
        Optional error to insert into the model (default is 0).
    mittag_leffler_type : str, optional
        Type of function to use ("Pade" or "Garrappa")
    
    Returns
    -------
    ndarray
        The creep compliance response at each time point in `t`.
    """
    #tau_c = (V / G)**(1 / (alpha - beta))
    #G_c = V * (tau_c**(-alpha))
    a = alpha - beta
    b = 1 + alpha
    z = -np.power(np.divide(t, tau_c), alpha - beta)
    
    if mittag_leffler_type == "Pade32":
        response_func = R_alpha_beta_3_2
    elif mittag_leffler_type == "Pade54":
        response_func = R_alpha_beta_5_4
    elif mittag_leffler_type == "Pade63":
        response_func = R_alpha_beta_6_3
    elif mittag_leffler_type == "Pade72":
        response_func = R_alpha_beta_7_2
    elif mittag_leffler_type == "Garrappa":
        response_func = E_alpha_beta
    else:
        raise ValueError("mittag_leffler_type must be either 'Pade' or 'Garrappa'")
        
    result = np.multiply(np.divide(np.power(t, alpha), V), response_func(z, a, b))
    
    if errorInserted != 0:
        error = createRandomError(t.shape[0], errorInserted)
        result = np.multiply(result, error)
        
    return result 
