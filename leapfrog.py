import numpy as np


def leapfrog(theta, r, grad_loss, eps):
    # Single update step
    r_tilde = r + (eps/2)*grad_loss(theta)
    theta_tilde = theta + eps*r_tilde
    r_tilde += (eps/2)*grad_loss(theta_tilde)
    return theta_tilde, r_tilde
