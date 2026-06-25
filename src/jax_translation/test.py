import jax.numpy as jnp
from symmetric_polynomials import get_symmetric_polynomials
from calculate_j_y_eigenstates import calculate_and_save_j_y_eigenstates
from fractions import Fraction

def test_symmetric_batch():
    # Test inputs
    roots_matrix = jnp.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=jnp.float32)  # Shape (2, 4)
    b = 3
    reg_coeffs = jnp.array([1, 1, 1], dtype=jnp.float32)

    # Run the function
    result = get_symmetric_polynomials(roots_matrix, b, reg_coeffs)

    # Print the result
    print("Computed symmetric polynomials for each batch:")
    print(result)

# Uncomment to run the test
# test_symmetric_batch()

# Define angular momentum
def test_calculate_and_save_j_y_eigenstates():
    # Define the angular momentum
    l = float(Fraction(9, 2))

    # Calculate and save eigenstates
    eigenstates = calculate_and_save_j_y_eigenstates(l)
    
    # print(eigenstates[(l, l, l)])
    # print(eigenstates.keys())
    # Print a sample result
    print(eigenstates)
    return

test_calculate_and_save_j_y_eigenstates()
# Calculate and save eigenstates
# eigenstates = calculate_and_save_j_y_eigenstates(l)

# Print a sample result
# print(eigenstates)