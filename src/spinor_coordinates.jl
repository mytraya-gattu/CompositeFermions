# Spinor coordinates on the sphere, shared by all wavefunction types.

"""
    u_v_generator(θ, ϕ)

Convert spherical coordinates `(θ, ϕ)` to the spinor (CP¹) coordinates

    u = cos(θ/2) exp(+i ϕ/2),   v = sin(θ/2) exp(-i ϕ/2).

Works for scalar inputs (returns a `Tuple{ComplexF64,ComplexF64}`) and for vector
inputs (returns a tuple of `Vector{ComplexF64}`), since the body broadcasts.
"""
function u_v_generator(θ, ϕ)
    return cos.(θ ./ 2) .* exp.(0.5im .* ϕ), sin.(θ ./ 2) .* exp.(-0.5im .* ϕ)
end
