module SymmetricPolynomials
export get_symmetric_polynomials!, update_symmetric_polynomials!
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
"""
    update_symmetric_polynomials!(dest, r_old, r_new, b, reg_coeffs)

Incrementally update the regularized elementary symmetric polynomials when a single root
changes from `r_old` to `r_new`. This is O(b) instead of the O(N*b) full recomputation.

Uses the fact that elementary symmetric polynomials are symmetric in the roots:
1. **Remove** `r_old` via forward recurrence (k = 1 → b):
   `f_k ← f_k - r_old * reg_coeffs[k] * f_{k-1}`
2. **Add** `r_new` via backward recurrence (k = b → 1):
   `f_k ← f_k + r_new * reg_coeffs[k] * f_{k-1}`

# Arguments
- `dest`: Array of current regularized elementary symmetric polynomials (modified in-place)
- `r_old`: The old root value being removed
- `r_new`: The new root value being added
- `b::Int`: Maximum order of elementary symmetric polynomials
- `reg_coeffs`: Regularization coefficients

# Returns
- `nothing`: In-place computation
"""
function update_symmetric_polynomials!(dest, r_old, r_new, b, reg_coeffs)

    if b == 0
        return
    end

    # Step 1: Remove old root (forward recurrence, k = 1 → b)
    for k in 1:b
        @inbounds dest[k + 1] -= r_old * reg_coeffs[k] * dest[k]
    end

    # Step 2: Add new root (backward recurrence, k = b → 1)
    for k in b:-1:1
        @inbounds dest[k + 1] += r_new * reg_coeffs[k] * dest[k]
    end

    return

end
end
