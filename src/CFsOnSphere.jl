module CFsOnSphere

using LinearAlgebra
using Random
using JLD2
using Combinatorics
using Serialization
using SpecialFunctions: loggamma

LinearAlgebra.BLAS.set_num_threads(1)

# Low-level building blocks.
include("calculate_j_y_eigenstates.jl")
include("spinor_coordinates.jl")
include("monopole_harmonics.jl")
include("symmetric_polynomials.jl")
include("jk_projection_utilities.jl")
include("legendre_polynomials.jl")

# Wavefunction types and their updates.
include("projected_wavefunction.jl")      # Ψproj, Ψparton
include("unprojected_wavefunction.jl")    # Ψunproj, ΨoneLL

# Linear algebra and Monte Carlo on top of the wavefunctions.
include("slater_inverse.jl")              # Sherman-Morrison (Ψunproj) + extended Slater
include("lambda_levels.jl")               # l_m_list builders
include("monte_carlo.jl")                 # proposal, ARM, gibbs_thermalization!, density

export
    # Wavefunction types
    Ψproj, Ψparton, Ψunproj, ΨoneLL,
    update_wavefunction!,
    # Orbitals / projection / polynomials
    calculate_ll, get_symmetric_polynomials!,
    # Slater inverse (Ψunproj) + quasihole/quasiparticle helper
    initialize_inverse!, slater_det_ratio, update_inverse!, build_extended_slater!,
    # Λ-level convenience builders
    cf_ground_state_lm, cf_quasihole_lm, cf_quasiparticle_lm,
    # Monte Carlo
    rand_θ_ϕ_gen, proposal, gibbs_thermalization!, update_density!, construct_det_ratios,
    # Misc
    legendre_polynomials!,
    # Re-exported utilities
    save, load, logdet, lu, inv

end
