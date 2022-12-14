import numpy as np
from utils import *


class NaiveNUTS:
    DELTAMAX = 1000.

    def __init__(self, eps, logp, grad_logp, num_iterations, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.eps = eps
        self.logp = logp
        self.grad_logp = grad_logp
        self.num_iterations = num_iterations

    def run(self, theta0):
        theta = [theta0]
        for m in range(self.num_iterations):
            r_0 = self.rng.multivariate_normal(np.zeros_like(theta0),
                                               np.eye(len(theta0)))
            # Setup slice variable
            energy = self.logp(theta[-1]) - r_0 @ r_0 / 2.
            u = self.rng.uniform(0, np.exp(energy))

            # Initialization
            theta_minus = theta[-1].copy()
            theta_plus = theta[-1].copy()
            r_minus = r_0
            r_plus = r_0
            j = 0
            C = [(theta[-1].copy(), r_0)]
            s = 1

            while s == 1:
                # Choose direction
                v = self.rng.choice([-1., 1.])
                # Doubling step in correct direction
                if v == -1.:
                    theta_minus, r_minus, _, _, C_prime, s_prime = \
                        self.buildTree(theta_minus, r_minus, u, v, j)
                else:
                    _, _, theta_plus, r_plus, C_prime, s_prime = \
                        self.buildTree(theta_plus, r_plus, u, v, j)
                """
                if np.isnan((theta_plus - theta_minus) @ r_minus):
                    print(m, j)
                    print(theta_plus, theta_minus)
                """
                # Check for stopping criterion
                if ((theta_plus - theta_minus) @ r_minus < 0.) or \
                   ((theta_plus - theta_minus) @ r_plus < 0.) or \
                   (s_prime == 0):
                    s = 0
                else:
                    # Add the new states to C if this is not the last doubling
                    C.extend(C_prime)
                j += 1
            chosen_state = C[self.rng.randint(len(C))]
            theta.append(chosen_state[0])
            # r is not used
        return theta

    def buildTree(self, theta, r, u, v, j):
        # Base case:
        if j == 0:
            # one leapfrog step in the right direction
            theta_prime, r_prime = leapfrog(theta, r,
                                            self.grad_logp,
                                            v*self.eps)
            alpha = np.exp(self.logp(theta_prime) - r_prime @ r_prime / 2.)
            C_prime = [(theta_prime, r_prime)] if u <= alpha else []

            s_prime = 1 if u < np.exp(NaiveNUTS.DELTAMAX +
                                      self.logp(theta_prime) -
                                      r_prime @ r_prime / 2.) \
                else 0

            return theta_prime, r_prime, theta_prime, r_prime, C_prime, s_prime
        else:
            # Construct j-1 steps (recursively)
            theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime = \
                self.buildTree(theta, r, u, v, j-1)
            # create j-th step by creating another tree of size j-1
            # on correct side
            if v == -1:
                theta_minus, r_minus, _, _, C_pp, s_pp = \
                    self.buildTree(theta_minus, r_minus, u, v, j-1)
            else:
                _, _, theta_plus, r_plus, C_pp, s_pp = \
                    self.buildTree(theta_plus, r_plus, u, v, j-1)

            # test for u-turn
            if ((theta_plus - theta_minus) @ r_minus < 0) or \
               ((theta_plus - theta_minus) @ r_plus < 0) or \
               (s_pp == 0):
                s_prime = 0
            C_prime.extend(C_pp)
            return theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime


class EfficientNUTS:
    DELTAMAX = 1000.

    def __init__(self, eps, logp, grad_logp, num_iterations, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.eps = eps
        self.logp = logp
        self.grad_logp = grad_logp
        self.num_iterations = num_iterations

    def run(self, theta0):
        theta = [theta0]
        for m in range(self.num_iterations):
            r_0 = self.rng.multivariate_normal(np.zeros_like(theta0),
                                               np.eye(len(theta0)))
            # Setup slice variable
            energy = self.logp(theta[-1]) - r_0 @ r_0 / 2.
            u = self.rng.uniform(0, np.exp(energy))

            # Initialization
            theta_minus = theta[-1].copy()
            theta_plus = theta[-1].copy()
            r_minus = r_0
            r_plus = r_0
            j = 0
            theta.append(theta[-1].copy())
            n = 1
            s = 1

            while s == 1:
                # Choose direction
                v = self.rng.choice([-1., 1.])
                # Doubling step in correct direction
                if v == -1.:
                    theta_minus, r_minus, _, _, theta_prime, \
                        n_prime, s_prime = \
                        self.buildTree(theta_minus, r_minus, u, v, j)
                else:
                    _, _, theta_plus, r_plus, theta_prime, \
                        n_prime, s_prime = \
                        self.buildTree(theta_plus, r_plus, u, v, j)
                """
                if np.isnan((theta_plus - theta_minus) @ r_minus):
                    print(m, j)
                    print(theta_plus, theta_minus)
                """
                # If not stopping criterion: sample with new subtree
                if s_prime == 1:
                    alpha = min(1, n_prime / n)
                    if self.rng.random() < alpha:
                        theta[-1] = theta_prime
                # update variables
                n += n_prime
                if ((theta_plus - theta_minus) @ r_minus < 0.) or \
                   ((theta_plus - theta_minus) @ r_plus < 0.) or \
                   (s_prime == 0):
                    s = 0
                j += 1
        return theta

    def buildTree(self, theta, r, u, v, j):
        # Base case:
        if j == 0:
            # one leapfrog step in the right direction
            theta_prime, r_prime = leapfrog(theta, r,
                                            self.grad_logp,
                                            v*self.eps)
            n_prime = 1 if u <= np.exp(self.logp(theta_prime) -
                                       r_prime @ r_prime / 2) else 0
            s_prime = 1 if u < np.exp(NaiveNUTS.DELTAMAX +
                                      self.logp(theta_prime) -
                                      r_prime @ r_prime / 2.) else 0

            return theta_prime, r_prime, theta_prime, r_prime,\
                theta_prime, n_prime, s_prime
        else:
            # Construct j-1 steps (recursively)
            theta_minus, r_minus, theta_plus, r_plus,\
                theta_prime, n_prime, s_prime = \
                self.buildTree(theta, r, u, v, j-1)
            # create j-th step by creating another tree of size j-1
            # on correct side
            if v == -1:
                theta_minus, r_minus, _, _, theta_pp, n_pp, s_pp = \
                    self.buildTree(theta_minus, r_minus, u, v, j-1)
            else:
                _, _, theta_plus, r_plus, theta_pp, n_pp, s_pp = \
                    self.buildTree(theta_plus, r_plus, u, v, j-1)

            # test for u-turn
            if n_prime + n_pp > 0:
                alpha = n_pp / (n_prime + n_pp)
            else:
                alpha = 0
            if self.rng.random() < alpha:
                theta_prime = theta_pp

            if ((theta_plus - theta_minus) @ r_minus < 0) or \
               ((theta_plus - theta_minus) @ r_plus < 0) or \
               (s_pp == 0):
                s_prime = 0
            n_prime += n_pp
            return theta_minus, r_minus, theta_plus, r_plus,\
                theta_prime, n_prime, s_prime


class AutoNUTS:
    DELTAMAX = 1000.

    def __init__(self, logp, grad_logp, num_iterations, num_adapt,
                 delta=0.65, seed=None):
        self.rng = np.random.RandomState(seed=seed)
        self.logp = logp
        self.grad_logp = grad_logp
        self.num_iterations = num_iterations
        self.num_adapt = num_adapt
        self.delta = delta
        self.gamma = 0.05
        self.t_0 = 0.75
        self.kappa = 0.75

        self.eps = []

    def run(self, theta0):
        theta = [theta0]

        # Initialize dual averaging
        self.eps = [FindReasonableEpsilon(theta0, logp=self.logp,
                                          grad_logp=self.grad_logp,
                                          rng=self.rng)]
        mu = 10*self.eps[-1]
        logeps_bar = 0
        H = 0

        for m in range(self.num_iterations):
            r_0 = self.rng.multivariate_normal(np.zeros_like(theta0),
                                               np.eye(len(theta0)))
            # Setup slice variable
            energy = self.logp(theta[-1]) - r_0 @ r_0 / 2.
            u = self.rng.uniform(0, np.exp(energy))

            # Initialization
            theta_minus = theta[-1].copy()
            theta_plus = theta[-1].copy()
            r_minus = r_0
            r_plus = r_0
            j = 0
            theta.append(theta[-1].copy())
            n = 1
            s = 1

            while s == 1:
                # Choose direction
                v = self.rng.choice([-1., 1.])
                # Doubling step in correct direction
                if v == -1.:
                    theta_minus, r_minus, _, _, theta_prime, \
                        n_prime, s_prime, alpha, n_alpha = \
                        self.buildTree(theta_minus, r_minus, u, v, j,
                                       theta[-1], r_0)
                else:
                    _, _, theta_plus, r_plus, theta_prime, \
                        n_prime, s_prime, alpha, n_alpha = \
                        self.buildTree(theta_plus, r_plus, u, v, j,
                                       theta[-1], r_0)
                """
                if np.isnan((theta_plus - theta_minus) @ r_minus):
                    print(m, j)
                    print(theta_plus, theta_minus)
                """
                # If not stopping criterion: sample with new subtree
                if s_prime == 1:
                    accept = min(1, n_prime / n)
                    if self.rng.random() < accept:
                        theta[-1] = theta_prime
                # update variables
                n += n_prime
                if ((theta_plus - theta_minus) @ r_minus < 0.) or \
                   ((theta_plus - theta_minus) @ r_plus < 0.) or \
                   (s_prime == 0):
                    s = 0
                j += 1

            # Update epsilon
            if m < self.num_adapt:
                H = (1-(1/(m+1 + self.t_0))) * H + \
                    (self.delta - alpha / n_alpha)/(m+1 + self.t_0)
                logeps = mu - H * np.sqrt(m)/self.gamma
                self.eps.append(np.exp(logeps))
                logeps_bar = (m+1)**(-self.kappa) * logeps + \
                    (1-(m+1)**(-self.kappa)) * logeps_bar
            else:
                self.eps.append(np.exp(logeps_bar))
        return theta

    def buildTree(self, theta, r, u, v, j, theta_0, r_0):
        # Base case:
        if j == 0:
            # one leapfrog step in the right direction
            theta_prime, r_prime = leapfrog(theta, r,
                                            self.grad_logp,
                                            v*self.eps[-1])
            n_prime = 1 if u <= np.exp(self.logp(theta_prime) -
                                       r_prime @ r_prime / 2) else 0
            s_prime = 1 if u < np.exp(NaiveNUTS.DELTAMAX +
                                      self.logp(theta_prime) -
                                      r_prime @ r_prime / 2.) else 0

            alpha = min(1, np.exp(
                self.logp(theta_prime) - r_prime@r_prime/2
                - self.logp(theta_0) + r_0 @ r_0 / 2
            ))
            n_alpha = 1
            return theta_prime, r_prime, theta_prime, r_prime,\
                theta_prime, n_prime, s_prime, alpha, n_alpha
        else:
            # Construct j-1 steps (recursively)
            theta_minus, r_minus, theta_plus, r_plus, theta_prime,\
                n_prime, s_prime, alpha_prime, n_alpha_prime = \
                self.buildTree(theta, r, u, v, j-1, theta_0, r_0)
            # create j-th step by creating another tree of size j-1
            # on correct side
            if v == -1:
                theta_minus, r_minus, _, _, theta_pp, n_pp, s_pp,\
                    alpha_pp, n_alpha_pp = \
                    self.buildTree(theta_minus, r_minus, u, v, j-1,
                                   theta_0, r_0)
            else:
                _, _, theta_plus, r_plus, theta_pp, n_pp, s_pp,\
                    alpha_pp, n_alpha_pp = \
                    self.buildTree(theta_plus, r_plus, u, v, j-1,
                                   theta_0, r_0)

            # sample appropriately
            if n_prime + n_pp > 0:
                accept = n_pp / (n_prime + n_pp)
            else:
                accept = 0
            if self.rng.random() < accept:
                theta_prime = theta_pp

            alpha_prime += alpha_pp
            n_alpha_prime += n_alpha_pp

            # test for u-turn
            if ((theta_plus - theta_minus) @ r_minus < 0) or \
               ((theta_plus - theta_minus) @ r_plus < 0) or \
               (s_pp == 0):
                s_prime = 0
            n_prime += n_pp

            return theta_minus, r_minus, theta_plus, r_plus,\
                theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
