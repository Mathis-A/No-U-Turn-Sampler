import numpy as np


def leapfrog(theta, r, grad_logp, eps):
    # Single update step
    r_tilde = r + (eps/2)*grad_logp(theta)
    theta_tilde = theta + eps*r_tilde
    r_tilde += (eps/2)*grad_logp(theta_tilde)
    return theta_tilde, r_tilde


def FindReasonableEpsilon(theta, logp, grad_logp, rng):
    eps = 1
    r = rng.multivariate_normal(np.zeros_like(theta),
                                np.eye(len(theta)))
    theta_prime, r_prime = leapfrog(theta, r, grad_logp, eps)
    a = 1 if ((logp(theta_prime) - r_prime @ r_prime / 2) /
              (logp(theta) - r @ r / 2)) > 0.5 else -1
    while ((logp(theta_prime) - r_prime @ r_prime / 2) /
           (logp(theta) - r @ r / 2))**a > 2**(-a):
        eps *= 1.4**a
        theta_prime, r_prime = leapfrog(theta, r_prime, grad_logp, eps)
    return eps
