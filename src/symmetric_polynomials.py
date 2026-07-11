import jax
import jax.numpy as jnp

def get_symmetric_polynomials(roots, b: int, reg_coeffs):
    """
    Calculate the first `b` elementary symmetric polynomials for given roots,
    regularized by ∏ᵢreg_coeffs[i].
    
    Args:
        roots: Array of elements for symmetric polynomial computation
        b: Maximum order of elementary symmetric polynomials to compute
        reg_coeffs: Array of regularization coefficients
        
    Returns:
        Array containing computed elementary symmetric polynomials
    """
    
    dest = jnp.zeros(b + 1, roots.shape[0], dtype=roots.dtype)
    
    dest = dest.at[0, :].set(1.0)  # First element is always 1
    
    if b == 0:
        return dest
    elif b == 1:
        dest = dest.at[1, :].set(jnp.sum(roots) * reg_coeffs[0])
        return dest
    
    return dest

print(get_symmetric_polynomials(jnp.array([1.0, 2.0, 3.0]), 2, jnp.array([1, 1, 1])))