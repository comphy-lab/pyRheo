import numpy as np
import math

def HerschelBulkley(sigma_y, k, n, gamma_dot, error=0):
    result = (sigma_y / gamma_dot) + (k * (gamma_dot**(n-1)))
    if error != 0:
        error = createRandomError(t.shape[0], error)
        result = np.multiply(result,error)
    return result

def Bingham(sigma_y, eta_p, gamma_dot, error=0):
    result = (sigma_y / gamma_dot) + eta_p
    if error != 0:
        error = createRandomError(t.shape[0], error)
        result = np.multiply(result,error)
    return result

def PowerLaw(k, n, gamma_dot, error=0):
    result = (k * (gamma_dot**(n-1)))
    if error != 0:
        error = createRandomError(t.shape[0], error)
        result = np.multiply(result,error)
    return result

def CarreauYasuda(eta_inf, eta_zero, k, a, n, gamma_dot, error=0):
    result = eta_inf + (eta_zero - eta_inf) * (1 + (k * gamma_dot)**(a))**((n-1)/a)
    if error != 0:
        error = createRandomError(t.shape[0], error)
        result = np.multiply(result,error)
    return result

def Cross(eta_inf, eta_zero, k, a, n, gamma_dot, error=0):
    result = eta_inf + ( (eta_zero - eta_inf) / (1 + (k * gamma_dot)**n) )
    if error != 0:
        error = createRandomError(t.shape[0], error)
        result = np.multiply(result,error)
    return result
    
def Casson(sigma_y, eta_p, gamma_dot, error=0):
    result = ( ((sigma_y**0.5) / (gamma_dot**0.5)) + (eta_p**0.5) )**2
    if error != 0:
        error = createRandomError(t.shape[0], error)
        result = np.multiply(result,error)
    return result
