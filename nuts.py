import numpy as np
from leapfrog import leapfrog


class NaiveNUTS:
    DELTAMAX = 1000

    def __init__(self, eps, loss, grad_loss, num_iterations, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.eps = eps
        self.loss = loss
        self.grad_loss = grad_loss
        self.num_iterations = num_iterations

    def run(self, theta0):
        theta = [theta0]
        r = [self.rng.multivariate_normal(
            np.zeros_like(theta0), np.eye(len(theta0)))]
        u = self.rng
        for m in range(self.M):
            theta_minus = theta[-1].copy()
            theta_plus = theta[-1].copy()
            r_minus = r[-1].copy()
            r_plus = r[-1].copy()
            j = 0
            C = [(theta[-1].copy(), r[-1].copy())]
            s = 1
            while s == 1:
                v = self.rng.choice([-1, 1])
                if v == -1:
                    theta_minus, r_minus, _, _, C_prime, s_prime = \
                        self.buildTree(theta_minus, r_minus, u, v, j)
                else:
                    _, _, theta_plus, r_plus, C_prime, s_prime = \
                        self.buildTree(theta_plus, r_plus, u, v, j)
                if s_prime == 1:
                    C.extend(C_prime)
                if ((theta_plus - theta_minus) @ r_minus < 0) or \
                   ((theta_plus - theta_minus) @ r_plus < 0) or \
                   (s_prime == 0):
                    s = 0
                j += 1
            new_theta, new_r = C[self.rng.randint(len(C))]
            theta.append(new_theta)
            r.append(new_r)
        return theta

    def buildTree(self, theta, r, u, v, j):
        if j == 0:
            theta_prime, r_prime = leapfrog(theta, self.grad_loss, r,
                                            v*self.eps)
            alpha = np.exp(self.loss(theta_prime) - r_prime @ r_prime / 2)
            C_prime = [(theta_prime, r_prime)] if u <= alpha else []
            s_prime = 1 if u < np.exp(NaiveNUTS.DELTAMAX +
                                      self.loss(theta_prime) -
                                      r_prime @ r_prime / 2) \
                else 0
            return theta_prime, r_prime, theta_prime, r_prime, C_prime, s_prime
        else:
            theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime = \
                self.buildTree(theta, r, u, v, j-1)
            if v == -1:
                theta_minus, r_minus, _, _, C_pp, s_pp = \
                    self.buildTree(theta_minus, r_minus, u, v, j-1)
            else:
                _, _, theta_plus, r_plus, C_pp, s_pp = \
                    self.buildTree(theta_plus, r_plus, u, v, j-1)
            if ((theta_plus - theta_minus) @ r_minus < 0) or \
               ((theta_plus - theta_minus) @ r_plus < 0) or \
               (s_pp == 0):
                s_prime = 0
            C_prime.extend(C_pp)
            return theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime
