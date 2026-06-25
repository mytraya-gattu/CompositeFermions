import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

@partial(jit, static_argnums=(1,))
def get_symmetric_polynomials_single(roots, b, reg_coeffs):
    """
    Calculate the first `b` elementary symmetric polynomials for a single set of roots,
    regularized by ∏ᵢreg_coeffs[i].

    Args:
        roots: Array of elements for which the elementary symmetric polynomials are to be computed.
        b: The maximum order of elementary symmetric polynomials to compute.
        reg_coeffs: Array of regularization coefficients.

    Returns:
        A JAX array containing the computed elementary symmetric polynomials.
    """
    dest = jnp.zeros(b + 1, dtype=roots.dtype)
    dest = dest.at[0].set(1)  # Set the first element to 1

    if b == 0:
        return dest
    elif b == 1:
        dest = dest.at[1].set(jnp.sum(roots) * reg_coeffs[0])
        return dest

    for i, r in enumerate(roots):
        for j in range(min(i + 1, b), 0, -1):
            dest = dest.at[j].add(r * dest[j - 1] * reg_coeffs[j - 1])

    return dest

# Vectorized version of the function
@partial(jit, static_argnums=(1,))
def get_symmetric_polynomials(roots_matrix, b, reg_coeffs):
    """
    Vectorized version of get_symmetric_polynomials_single to handle a batch of roots.

    Args:
        roots_matrix: A matrix of shape (B, N), where B is the batch size and N is the number of roots.
        b: The maximum order of elementary symmetric polynomials to compute.
        reg_coeffs: Array of regularization coefficients.

    Returns:
        A JAX array of shape (B, b + 1) containing the computed symmetric polynomials for each batch.
    """
    return vmap(lambda roots: get_symmetric_polynomials_single(roots, b, reg_coeffs))(roots_matrix)