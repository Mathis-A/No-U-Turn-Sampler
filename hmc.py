import numpy as np
from utils import leapfrog, FindReasonableEpsilon


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


class AutoHMC:

    def __init__(self, lambd, logp, grad_logp, num_iterations,
                 num_adapt, delta=0.65, seed=None):
        self.rng = np.random.RandomState(seed=seed)
        self.lambd = lambd
        self.logp = logp
        self.grad_logp = grad_logp
        self.num_iterations = num_iterations
        self.num_adapt = num_adapt
        self.delta = delta
        self.gamma = 0.05
        self.t_0 = 0.75
        self.kappa = 0.75
        self.reject = 0

    def run(self, theta0):
        theta = [theta0]
        r = []  # Not currently used but could be for visualization.
        # Initialize dual averaging
        eps = [FindReasonableEpsilon(theta0, logp=self.logp,
                                     grad_logp=self.grad_logp, rng=self.rng)]
        mu = 10*eps[-1]
        logeps_bar = 0
        H = 0
        for m in range(self.num_iterations):
            # Set r_0 and theta_m
            r_0 = self.rng.multivariate_normal(np.zeros_like(theta0),
                                               np.eye(len(theta0)))
            r.append(r_0)
            theta.append(theta[-1].copy())

            # Set theta_tilde and r_tilde
            r_tilde = r_0.copy()
            theta_tilde = theta[-1].copy()

            # Set number of leapfrog steps
            L_m = max(1, int(self.lambd/eps[-1]))
            # Leapfrog steps to generate a new state
            for _ in range(L_m):
                theta_tilde, r_tilde = leapfrog(theta_tilde, r_tilde,
                                                self.grad_logp,
                                                eps[-1])
                print(theta_tilde, r_tilde)
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

            # Update epsilon
            if m < self.num_adapt:
                H = (1-(1/(m+1 + self.t_0))) * H + \
                    (self.delta - alpha)/(m+1 + self.t_0)
                logeps = mu - H * np.sqrt(m)/self.gamma
                eps.append(np.exp(logeps))
                logeps_bar = (m+1)**(-self.kappa) * logeps + \
                    (1-(m+1)**(-self.kappa)) * logeps_bar
            else:
                eps.append(np.exp(logeps_bar))
        return theta, eps


if __name__ == '__main__':
    pass
