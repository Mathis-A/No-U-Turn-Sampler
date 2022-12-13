import numpy as np
from leapfrog import leapfrog


class HamiltonianMonteCarlo:

    def __init__(self, eps, L, loss, grad_loss, num_iterations, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.eps = eps
        self.L = L
        self.loss = loss
        self.grad_loss = grad_loss
        self.num_iterations = num_iterations

    def run(self, theta0):
        theta = [theta0]
        r = [self.rng.multivariate_normal(np.zeros_like(theta0),
                                          np.eye(len(theta0)))]
        for m in range(self.M):
            theta.append(theta[-1].copy())
            r.append(r[-1].copy())

            theta_tilde = theta[-1].copy()
            r_tilde = r[-1].copy()
            for i in range(len(self.L)):
                theta_tilde = leapfrog(theta_tilde, self.grad_loss, r_tilde,
                                       self.eps)
            alpha = min(1, np.exp(self.loss(theta_tilde) -
                                  r_tilde @ r_tilde / 2) /
                        np.exp(self.loss(theta[-1]) -
                               r[-1] @ r[-1] / 2))
            if self.rng.random() < alpha:
                theta[-1] = theta_tilde
                r[-1] = r_tilde
        return theta
