# Jain-Kamilla projection coefficients and the Fourier/Wigner matrices that, contracted
# with the elementary symmetric polynomials, give the projected single-particle orbitals.

"""
    custom_logbinomial(q1, q2) -> Float64

Logarithm of the binomial coefficient `C(q1, q2)` via `loggamma`, returning `-Inf` when
`q1 - q2 < 0` (so the corresponding term drops out of [`projection_coeff`](@ref)).
"""
function custom_logbinomial(q1, q2)
    if q1 - q2 >= 0
        return loggamma(q1 + 1) - loggamma(q2 + 1) - loggamma(q1 - q2 + 1)
    else
        return -Inf
    end
end

"""
    projection_coeff(L, Qstar, Q1, m) -> Float64

The Jain-Kamilla projection coefficient `N^L_{m,Q*,Q1}` (see the manuscript),

    N^L_{m,Q*,Q1} = C(2Q1, L-Q*) C(L-Q*, m-Q*) / C(2Q1+L+Q*+1, L-Q*)
                    · √[ (2L+1)/4π · C(2L, L+Q*) / C(2L, L+m) ],

with `Q1 = jk_type·(N-1)/2`. Here `√(2L+1)/4π` is supplied separately in
[`generate_fourier_matrices`](@ref).
"""
function projection_coeff(L, Qstar, Q1, m)

    return exp(custom_logbinomial(2 * Q1, L - Qstar) + custom_logbinomial(L - Qstar, m - Qstar) + 0.50 * custom_logbinomial(2 * L, L + Qstar) - custom_logbinomial(2 * Q1 + L + Qstar + 1, L - Qstar) - 0.50 * custom_logbinomial(2 * L, L + m))

end

"""
    generate_fourier_matrices(Qstar, N, L, Lz_list; jk_type::Int = 1)

Build the Fourier-coefficient matrix for the Jain-Kamilla projection at total angular
momentum `L`, effective monopole strength `Qstar`, and the requested `Lz_list`.

`jk_type` selects the power of the Jastrow factor `Jᵢ = ∏_{j≠i}(uᵢvⱼ - uⱼvᵢ)` that is
LLL-projected together with the monopole harmonic: the projected orbital is
`[P_LLL Y_{Q*,L,M} Jᵢ^{jk_type}] / Jᵢ^{jk_type}`. Correspondingly the JK angular momentum
is `Q1 = jk_type·(N-1)/2` (so `2Q1 = jk_type·(N-1)` is the number of multiplicity-weighted
Jastrow roots), which enters the binomials of [`projection_coeff`](@ref). `jk_type = 1`
reproduces the standard JK projection.
"""
function generate_fourier_matrices(Qstar, N, L, Lz_list; jk_type::Int = 1)

    @assert denominator(2 * L) == 1 && L ≥ abs(Qstar) "Invalid angular momentum."
    wigner_d_fourier_coeffecients = calculate_j_y_eigenstates(L)

    Q1 = jk_type * (N - 1) // 2

    fourier_matrix = zeros(ComplexF64, length(Lz_list), numerator(1 + L - Qstar), numerator(2 * L + 1))

    for Lzprime in Qstar:1:L

        coeff = projection_coeff(L, Qstar, Q1, Lzprime) * (-1)^(round(Int64, Lzprime - Qstar))

        for (iter, Lz) in enumerate(Lz_list)

            for μ in -L:1:L

                fourier_matrix[iter, round(Int64, Lzprime + 1 - Qstar), round(Int64, μ + L + 1)] = wigner_d_fourier_coeffecients[(μ, Lzprime, Lz)]

            end

        end

        fourier_matrix[:, numerator(Lzprime + 1 - Qstar), :] .*= coeff

    end

    return fourier_matrix .* sqrt((2 * L + 1) / (4.0 * π))

end
