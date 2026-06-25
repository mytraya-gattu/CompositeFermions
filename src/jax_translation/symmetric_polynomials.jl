module SymmetricPolynomials
export get_symmetric_polynomials!
"""
    get_symmetric_polynomials!(dest, roots, b::Int, reg_coeffs)
Calculate the first `b` elementary symmetric polynomials for the given roots regularized by ∏ᵢreg_coeffs[i].

# Arguments
- `dest`: Destination array for the computed elementary symmetric polynomials
- `roots`: Vector of elements for which the elementary symmetric polynomials are to be computed
- `b::Int`: The maximum order of elementary symmetric polynomials to compute.
- `reg_coeffs`: Array of regularization coefficients.

# Returns
- `nothing`: In-place computation
"""
function get_symmetric_polynomials!(dest, roots, b, reg_coeffs)

    dest[1] = one(eltype(dest))

    if b == 0
        return
    elseif b == 1
        dest[2] = sum(roots) * reg_coeffs[1] ## okay.
        return
    end

    dest[2:end] .= zero(eltype(dest))

    for i in eachindex(roots)
        r = roots[i]
        for j in min(i, b):-1:1

            @inbounds dest[j + 1] += r * dest[j] * reg_coeffs[j]

        end

    end

    return

end
end
