import os
import pickle
import numpy as np
from fractions import Fraction

def calculate_j_y_eigenstates(l):
    """
    Calculate the eigenstates of the J_y operator in the J_z basis for a given angular momentum `l`.

    Args:
        l: Angular momentum as a float (e.g., 0.5, 1.0, 1.5).

    Returns:
        A dictionary mapping (μ, m1, m2) to the corresponding matrix element.
    """
    # Ensure `l` is valid
    if not (2 * l).is_integer():
        raise ValueError("Invalid angular momentum.")

    # Dimension of the J_z basis
    dim_j = int(2 * l + 1)

    # Initialize the J_y matrix in the J_z basis
    j_y_in_j_z_basis = np.zeros((dim_j, dim_j), dtype=np.complex128)

    # Fill the J_y matrix
    for m in np.arange(-l, l, step=1.0):
        miter = int(m + l)
        j_y_in_j_z_basis[miter + 1, miter] = np.sqrt(l * (l + 1) - m * (m + 1)) / (2.0j)

    for m in np.arange(-l + 1, l + 1, step=1.0):
        miter = int(m + l)
        j_y_in_j_z_basis[miter - 1, miter] = -np.sqrt(l * (l + 1) - m * (m - 1)) / (2.0j)

    # Compute eigenvalues and eigenvectors
    _, jy_eigvecs = np.linalg.eigh(j_y_in_j_z_basis)

    # Create the dictionary of results
    ans_dict = {}
    for μ in np.arange(-l, l + 1, step=1.0):
        for m1 in np.arange(-l, l + 1, step=1.0):
            for m2 in np.arange(-l, l + 1, step=1.0):
                # Convert keys to Fraction to ensure exact arithmetic
                ans_dict[(Fraction(μ), Fraction(m1), Fraction(m2))] = (
                    jy_eigvecs[int(m1 + l), int(μ + l)]
                    * np.conj(jy_eigvecs[int(m2 + l), int(μ + l)])
                )
    return ans_dict


def save_eigenstates_to_file(l, ans_dict, directory="eigenstates"):
    """
    Save the eigenstates dictionary to a file.

    Args:
        l: Angular momentum as a float.
        ans_dict: The dictionary of eigenstates to save.
        directory: Directory where the file will be saved.
    """
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(
        directory, f"j_y_eigenstates_at_angular_momentum_{int(2 * l)}.pkl"
    )
    with open(filename, "wb") as f:
        pickle.dump(ans_dict, f)


def load_eigenstates_from_file(l, directory="eigenstates"):
    """
    Load the eigenstates dictionary from a file.

    Args:
        l: Angular momentum as a float.
        directory: Directory where the file is saved.

    Returns:
        The loaded dictionary of eigenstates, or None if the file does not exist.
    """
    filename = os.path.join(
        directory, f"j_y_eigenstates_at_angular_momentum_{int(2 * l)}.pkl"
    )
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None


def calculate_and_save_j_y_eigenstates(l):
    """
    Calculate and save the eigenstates of the J_y operator.

    Args:
        l: Angular momentum as a float.

    Returns:
        The dictionary of eigenstates.
    """
    # Try to load from file
    ans_dict = load_eigenstates_from_file(l)
    if ans_dict is not None:
        return ans_dict

    # Calculate eigenstates
    ans_dict = calculate_j_y_eigenstates(l)

    # Save to file
    save_eigenstates_to_file(l, ans_dict)

    return ans_dict