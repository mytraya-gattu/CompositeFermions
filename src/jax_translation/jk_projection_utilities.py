import numpy as np
from scipy.special import gammaln
from calculate_j_y_eigenstates import calculate_and_save_j_y_eigenstates
from math import pi, sqrt

def custom_logbinomial(q1, q2):
    """
    Compute the logarithm of the binomial coefficient.
    """
    if q1 - q2 >= 0:
        return gammaln(q1 + 1) - gammaln(q2 + 1) - gammaln(q1 - q2 + 1)
    else:
        return -np.inf


def projection_coeff(L, Qstar, Q1, m):
    """
    Compute the projection coefficient.
    """
    return np.exp(
        custom_logbinomial(2 * Q1, L - Qstar)
        + custom_logbinomial(L - Qstar, m - Qstar)
        + 0.5 * custom_logbinomial(2 * L, L + Qstar)
        - custom_logbinomial(2 * Q1 + L + Qstar + 1, L - Qstar)
        - 0.5 * custom_logbinomial(2 * L, L + m)
    )


def generate_fourier_matrices(Qstar, N, L, Lz_list):
    """
    Generate Fourier matrices for the given parameters.

    Args:
        Qstar: Integer or rational angular momentum offset.
        N: Number of particles.
        L: Total angular momentum.
        Lz_list: List of Lz values.

    Returns:
        A 3D NumPy array representing the Fourier matrices.
    """
    # Ensure valid angular momentum
    assert (2 * L).is_integer() and L >= abs(Qstar), "Invalid angular momentum."

    # Calculate Wigner-d Fourier coefficients
    wigner_d_fourier_coefficients = calculate_and_save_j_y_eigenstates(L)

    Q1 = (N - 1) // 2

    # Initialize the Fourier matrix
    fourier_matrix = np.zeros(
        (len(Lz_list), int(1 + L - Qstar), int(2 * L + 1)), dtype=np.complex128
    )

    for Lzprime in np.arange(Qstar, L + 1, step=1):
        coeff = (
            projection_coeff(L, Qstar, Q1, Lzprime)
            * (-1) ** int(Lzprime - Qstar)
        )

        for iter, Lz in enumerate(Lz_list):
            for μ in np.arange(-L, L + 1, step=1):
                fourier_matrix[
                    iter, int(Lzprime + 1 - Qstar), int(μ + L + 1)
                ] = wigner_d_fourier_coefficients[(μ, Lzprime, Lz)]

        # Scale the Fourier matrix by the coefficient
        fourier_matrix[:, int(Lzprime + 1 - Qstar), :] *= coeff

    return fourier_matrix * sqrt((2 * L + 1) / (4.0 * pi))