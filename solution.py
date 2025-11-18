"""Solution."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, DotProduct
from scipy.stats import norm  # Add this import for P(safe)


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""

        self.prior_mean_v = 4.0

        self.beta_constant = 3.0
        self.beta_log = 0.5

        self.expander_constant = 30.0
        self.expander_log = 5.0

        self.lambda_pen = 12.0
        #set the kernel hyperparameters
        kernel_f_params = {
            "constant": {
                "constant_value": 1.0,
                "constant_value_bounds": (1e-3, 1e3)
            },
            "matern": {
                "length_scale": 1.0,
                "nu": 2.5,
                "length_scale_bounds": (1e-3, 4.0)
            },
            "white": {
                "noise_level": 0.0001**2,
                "noise_level_bounds": "fixed"
            }
        }
        kernel_v_params = {
            "constant": {
                "constant_value": 1.0,
                "constant_value_bounds": (1e-2, 100.0)
            },
            "matern": {
                "length_scale": 1.0,
                "nu": 2.5,
                "length_scale_bounds": (1e-3, 4.0)
            },
            "white": {
                "noise_level": 0.0001**2,
                "noise_level_bounds": "fixed"
            }
        }

        #kernel for objective f (logP) match sigma_f: 0.15
        kernel_f = ConstantKernel(**kernel_f_params["constant"])\
            * Matern(**kernel_f_params["matern"]) \
            + WhiteKernel(**kernel_f_params["white"])
        #kernel for constraint v (SA) match sigma_v: 0.0001
        kernel_v = ConstantKernel(**kernel_v_params["constant"]) \
         * Matern(**kernel_v_params["matern"]) \
         + WhiteKernel(**kernel_v_params["white"])

        self.gp_f = GaussianProcessRegressor(
         kernel               = kernel_f,
         alpha                = 0.0,
         n_restarts_optimizer = 10,
         normalize_y          = True,
         )
        self.gp_v = GaussianProcessRegressor(
         kernel               = kernel_v,
         alpha                = 0.0,
         n_restarts_optimizer = 10,
         normalize_y          = False,
         )


        self.X = []
        self.Y_f = []
        self.Y_v = []

        #track best safe point
        self.best_safe_x = None
        self.best_safe_f = -np.inf

        self._mean_set = False

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        x_opt = self.optimize_acquisition_function()

        return np.atleast_2d(x_opt)

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt


    def acquisition_function(self, x: np.ndarray):
        x = np.atleast_2d(x)

        mu_f, sigma_f = self.gp_f.predict(x, return_std=True)
        mu_v, sigma_v = self.gp_v.predict(x, return_std=True)

        sigma_f = sigma_f.reshape(-1, 1)
        sigma_v = sigma_v.reshape(-1, 1)

        n = max(len(self.X), 1)
        beta = self.beta_constant + self.beta_log * np.log(n + 1)

        
        U_f = mu_f + beta * sigma_f
        L_v = mu_v - beta * sigma_v
        U_v = mu_v + beta * sigma_v

        safe = (L_v < SAFETY_THRESHOLD).ravel()

        # Lagrangian penalty term: λ * max(v - κ, 0)

        penalty = self.lambda_pen * np.maximum(U_v - SAFETY_THRESHOLD, 0).ravel()  # Penalize if upper v > κ (risky)

        current_max = self.best_safe_f if self.best_safe_f > -np.inf else mu_f.max()

        maximizer = np.maximum(U_f.ravel() - current_max, 0) * safe

        expander = sigma_v.ravel() * safe
        expander *= self.expander_constant + self.expander_log * np.log(n + 1)

        # Apply Lagrangian: penalize the entire AF for risky points
        af = (maximizer + expander) - penalty * safe  # Only penalize possibly safe points

        # Soft P(safe) multiplier for extra safety (from Gelbart paper)
        p_safe = norm.cdf((SAFETY_THRESHOLD - mu_v.ravel()) / sigma_v.ravel())
        af *= p_safe  # Downweight points with low P(v < κ)

        return af

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float or np.ndarray
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        self.X.append(float(x))
        self.Y_f.append(f)
        self.Y_v.append(v)

        X_arr = np.array(self.X).reshape(-1, 1)

        # Fit both GPs
        self.gp_f.fit(X_arr, self.Y_f)
        self.gp_v.fit(X_arr, self.Y_v)

        # CRITICAL: Set prior mean = 4.0 ONLY ONCE
        if not self._mean_set and len(self.X) == 1:  # after first point
            self.gp_v.kernel_.k1.constant_value = self.prior_mean_v
            self.gp_v.kernel_.k1.constant_value_bounds = "fixed"
            # Refit once with correct mean
            self.gp_v.fit(X_arr, self.Y_v)
            self._mean_set = True

        # Update best known safe point
        if v < SAFETY_THRESHOLD and f > self.best_safe_f:
            self.best_safe_f = f
            self.best_safe_x = float(x)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        # Return the best SAFE point we have actually observed
        if self.best_safe_x is not None:
            return np.atleast_2d(self.best_safe_x)
        else:
            # Fallback: return center
            return np.array([[5.0]])

    def plot(self, plot_recommendation: bool = True):
        try:
            self._plot(plot_recommendation)
        except Exception as e:
            print(f"Error plotting: {e}")

    def _plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        if len(self.X) == 0:
            print("No data points to plot yet.")
            return
        
        # Create a fine grid of x values across the domain
        x_grid = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 200).reshape(-1, 1)
        
        # Predict posterior for objective f
        mu_f, sigma_f = self.gp_f.predict(x_grid, return_std=True)
        mu_f = mu_f.ravel()
        sigma_f = sigma_f.ravel()
        
        # Predict posterior for constraint v
        mu_v, sigma_v = self.gp_v.predict(x_grid, return_std=True)
        mu_v = mu_v.ravel()
        sigma_v = sigma_v.ravel()
        
        # Get recommended point if requested (compute once)
        x_rec = None
        mu_f_rec = None
        mu_v_rec = None
        if plot_recommendation:
            x_rec = self.next_recommendation()
            mu_f_rec, _ = self.gp_f.predict(x_rec, return_std=True)
            mu_v_rec, _ = self.gp_v.predict(x_rec, return_std=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot objective posterior
        ax1.plot(x_grid, mu_f, 'b-', label='Mean f(x)', linewidth=2)
        ax1.fill_between(x_grid.ravel(), mu_f - 2*sigma_f, mu_f + 2*sigma_f, 
                        alpha=0.3, color='blue', label='95% CI')
        
        # Plot observed objective data points
        if len(self.X) > 0:
            X_arr = np.array(self.X).reshape(-1, 1)
            ax1.scatter(X_arr, self.Y_f, c='red', s=50, zorder=5, 
                       label='Observed f(x)', edgecolors='black', linewidths=1)
        
        # Plot recommended point if requested
        if plot_recommendation and x_rec is not None:
            ax1.scatter(x_rec, mu_f_rec, c='green', s=100, marker='*', 
                       zorder=6, label='Recommended point', edgecolors='black', linewidths=1)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x) (logP)')
        ax1.set_title('Objective Posterior (f)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(DOMAIN[0, 0], DOMAIN[0, 1])
        
        # Plot constraint posterior
        ax2.plot(x_grid, mu_v, 'r-', label='Mean v(x)', linewidth=2)
        ax2.fill_between(x_grid.ravel(), mu_v - 2*sigma_v, mu_v + 2*sigma_v, 
                        alpha=0.3, color='red', label='95% CI')
        
        # Plot safety threshold
        ax2.axhline(y=SAFETY_THRESHOLD, color='orange', linestyle='--', 
                   linewidth=2, label=f'Safety threshold ({SAFETY_THRESHOLD})')
        
        # Highlight safe region
        safe_mask = (mu_v - 2*sigma_v) < SAFETY_THRESHOLD
        ax2.fill_between(x_grid.ravel(), mu_v - 2*sigma_v, SAFETY_THRESHOLD, 
                        where=safe_mask, alpha=0.2, color='green', label='Safe region (95% CI)')
        
        # Plot observed constraint data points
        if len(self.X) > 0:
            X_arr = np.array(self.X).reshape(-1, 1)
            ax2.scatter(X_arr, self.Y_v, c='blue', s=50, zorder=5, 
                       label='Observed v(x)', edgecolors='black', linewidths=1)
        
        # Plot recommended point if requested
        if plot_recommendation and x_rec is not None:
            ax2.scatter(x_rec, mu_v_rec, c='green', s=100, marker='*', 
                       zorder=6, label='Recommended point', edgecolors='black', linewidths=1)
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('v(x) (SA)')
        ax2.set_title('Constraint Posterior (v)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(DOMAIN[0, 0], DOMAIN[0, 1])
        
        plt.tight_layout()
        plt.savefig('bo_plot.png')
        plt.close()


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)
    agent.plot()

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    unsafe_evals = len([v for v in agent.Y_v if v > SAFETY_THRESHOLD])

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals {unsafe_evals}\n')


if __name__ == "__main__":
    main()
