import numpy as np


# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return the bivariate Gaussian PDF f_XY(x,y).

    Use the formula:

    f_XY(x,y) =
    1 / (2*pi*sigma_x*sigma_y*sqrt(1-rho^2))
    *
    exp( -Q / (2*(1-rho^2)) )
    """
    coef = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))

    q = (
        ((x - mu_x) ** 2) / sigma_x**2
        - 2 * rho * ((x - mu_x) * (y - mu_y)) / (sigma_x * sigma_y)
        + ((y - mu_y) ** 2) / sigma_y**2
    )

    return coef * np.exp(-q / (2 * (1 - rho**2)))


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    """
    Return marginal Gaussian PDF of X.
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(
        -((x - mu_x) ** 2) / (2 * sigma_x**2)
    )


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    """
    Return marginal Gaussian PDF of Y.
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma_y)) * np.exp(
        -((y - mu_y) ** 2) / (2 * sigma_y**2)
    )


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return covariance matrix:

    [[sigma_x^2, rho*sigma_x*sigma_y],
     [rho*sigma_x*sigma_y, sigma_y^2]]
    """
    return np.array([[sigma_x**2, rho * sigma_x * sigma_y], [rho * sigma_x * sigma_y, sigma_y**2]])


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):
    """
    Numerically approximate integral of joint Gaussian PDF
    over the rectangle:

    [mu_x - 4*sigma_x, mu_x + 4*sigma_x]
    x
    [mu_y - 4*sigma_y, mu_y + 4*sigma_y]

    Use a rectangular grid or trapezoidal numerical integration.
    """
    x_vals = np.linspace(mu_x - 4*sigma_x, mu_x + 4*sigma_x, n)
    y_vals = np.linspace(mu_y - 4*sigma_y, mu_y + 4*sigma_y, n)

    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    total = 0

    for x in x_vals:
        for y in y_vals:
            total += joint_gaussian_pdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho)

    return total * dx * dy


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    """
    Generate n samples from a jointly Gaussian distribution.

    Return two arrays:
    x_samples, y_samples

    Hint:
    Use np.random.multivariate_normal.
    """
    np.random.seed(seed)

    mean = [mu_x, mu_y]
    cov = covariance_matrix(sigma_x, sigma_y, rho)

    samples = np.random.multivariate_normal(mean, cov, n)

    return samples[:, 0], samples[:, 1]


def sample_means(x_samples, y_samples):
    """
    Return sample means of X and Y.
    """
    return np.mean(x_samples), np.mean(y_samples)


def sample_covariance_matrix(x_samples, y_samples):
    """
    Return 2 by 2 sample covariance matrix.

    Use denominator n-1.
    """
    return np.cov(x_samples, y_samples)


def sample_correlation(x_samples, y_samples):
    """
    Return sample correlation coefficient.
    """
    return np.corrcoef(x_samples, y_samples)[0, 1]


def gaussian_independence_check(rho):
    """
    For jointly Gaussian variables:
    return True if rho is zero, otherwise False.
    """
    return rho == 0


def zero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0 and check that
    sample covariance is approximately zero.
    Return True or False.
    """
    x, y = generate_joint_gaussian_samples(rho=0, n=n)

    cm = sample_covariance_matrix(x, y)

    return bool(abs(cm[0, 1]) < 0.05)

def nonzero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0.6 and check that
    sample covariance is close to rho*sigma_x*sigma_y.
    Return True or False.
    """
    sigma_x = 2
    sigma_y = 3
    rho = 0.6

    x, y = generate_joint_gaussian_samples(
        n=n,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        rho=rho
    )

    cm = sample_covariance_matrix(x, y)

    expected = rho * sigma_x * sigma_y

    return bool(abs(cm[0, 1] - expected) < 0.2)
