module SymmetricPolynomials
export get_symmetric_polynomials!
function get_symmetric_polynomials(roots, b) ## For reference only.

    dest = zeros(ComplexF64, b + 1)
    dest[1] = one(ComplexF64)

    if b == 0
        return dest
    elseif b == 1
        dest[2] = sum(roots)
        return dest
    end

    for i in eachindex(roots)

        for j in min(i, b):-1:1

            @inbounds dest[j + 1] += roots[i] * dest[j]

        end

    end

    return dest
end

"""
    get_symmetric_polynomials!(dest, roots, b::Int)
Calculate the first `b` elementary symmetric polynomials for the given roots.

# Arguments
- `dest`: Destination array for the computed elementary symmetric polynomials
- `roots`: Vector of elements for which the elementary symmetric polynomials are to be computed
- `b::Int`: The maximum order of elementary symmetric polynomials to compute.

# Returns
- `nothing`: In-place computation
    """
function get_symmetric_polynomials!(dest, roots, b)

    dest[1] = one(eltype(dest))

    if b == 0
        return
    elseif b == 1
        dest[2] = sum(roots)
        return
    end

    dest[2:end] .= zero(eltype(dest))

    for i in eachindex(roots)

        r = roots[i]

        for j in min(i, b):-1:1

            @inbounds dest[j + 1] += r * dest[j]

        end

    end

    return

end
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
