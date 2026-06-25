module CFsOnSphere

include("projected_wavefunction.jl")
using .SpinPolarizedProjectedWavefunction

include("monte_carlo_utilities.jl")
using .MonteCarloOnSphere

include("legendre_polynomials.jl")
using .LegendrePolynomials

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

using JLD2
using Random

export Ψproj, ΨoneLL, Ψparton, update_wavefunction!, gibbs_thermalization!, rand_θ_ϕ_gen, proposal, legendre_polynomials!, save, load, logdet, lu, inv, update_density!, construct_det_ratios

end
