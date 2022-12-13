import numpy as np
from utils import leapfrog


class HamiltonianMonteCarlo:

    def __init__(self, eps, L, logp, grad_logp, num_iterations, seed=None):
        self.rng = np.random.RandomState(seed=seed)
        self.eps = eps
        self.L = L
        self.logp = logp
        self.grad_logp = grad_logp
        self.num_iterations = num_iterations
        self.reject = 0

    def run(self, theta0):
        theta = [theta0]
        r = []  # Not currently used but could be for visualization.

        for _ in range(self.num_iterations):
            # Set r_0 and theta_m
            r_0 = self.rng.multivariate_normal(np.zeros_like(theta0),
                                               np.eye(len(theta0)))
            r.append(r_0)
            theta.append(theta[-1].copy())

            # Set theta_tilde and r_tilde
            r_tilde = r_0.copy()
            theta_tilde = theta[-1].copy()

            # Leapfrog steps to generate a new state
            for _ in range(self.L):
                theta_tilde, r_tilde = leapfrog(theta_tilde, r_tilde,
                                                self.grad_logp,
                                                self.eps)
                # Each step could be extracted for visualisation purposes

            # Acceptance probability
            alpha = min(1, np.exp(self.logp(theta_tilde) -
                                  r_tilde @ r_tilde / 2) /
                        np.exp(self.logp(theta[-1]) - r_0 @ r_0 / 2))

            if self.rng.random() < alpha:
                # Accept
                theta[-1] = theta_tilde
                r[-1] = r_tilde
            else:
                self.reject += 1
            # else: theta[-1] and r[-1] are already set.
        return theta


if __name__ == '__main__':
    pass
