import numpy as np

def leapfrog(theta, grad_loss, r, eps):
    r_tilde = r + (eps/2)*grad_loss(theta)
    theta_tilde = theta + eps*r_tilde
    r_tilde+= (eps/2)*grad_loss(theta_tilde)
    return theta_tilde, r_tilde