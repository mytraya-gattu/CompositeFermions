# Elementary symmetric polynomials (ESP) used by the Jain-Kamilla projection.
#
# The JK projection binds a single vortex pair into each orbital — `[P_LLL Y_{Q*,l,m} Jᵢ] / Jᵢ`
# with `Jᵢ = ∏_{j≠i}(uᵢvⱼ - uⱼvᵢ)` — so the ESP are over the `N-1` distinct ratio roots, each
# with multiplicity one, and `Q₁ = (N-1)/2`. (Higher Jastrow powers `ν = n/(2pn+1)` are
# obtained from the global Jastrow factor `∏(uᵢvⱼ-uⱼvᵢ)^p`, not by multiplying roots.)

"""
    get_symmetric_polynomials(roots, b)

Allocating reference implementation (not exported): returns a length-`b+1` vector whose
`k+1`-th entry is the elementary symmetric polynomial of degree `k` in `roots`. Used to
cross-check the in-place versions.
"""
function get_symmetric_polynomials(roots, b)

    dest = zeros(ComplexF64, b + 1)
    dest[1] = one(ComplexF64)

    if b == 0
        return dest
    elseif b == 1
        dest[2] = sum(roots)
        return dest
    end

    for i in eachindex(roots)
        r = roots[i]
        for j in min(i, b):-1:1
            @inbounds dest[j+1] += r * dest[j]
        end
    end

    return dest
end

"""
    get_symmetric_polynomials!(dest, roots, b::Int)

In-place: fill `dest[1:b+1]` with the elementary symmetric polynomials of degrees `0…b` in
`roots`.

# Arguments
- `dest`: destination array of length ≥ `b+1`.
- `roots`: the roots.
- `b::Int`: maximum ESP degree to compute.
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
            @inbounds dest[j+1] += r * dest[j]
        end
    end

    return
end

"""
    get_symmetric_polynomials!(dest, roots, b::Int, reg_coeffs)

In-place, regularized variant: `dest[k+1]` holds the degree-`k` elementary symmetric
polynomial multiplied by `∏_{i=1}^{k} reg_coeffs[i]`.

For the JK projection `reg_coeffs[i] = i / ((N-1) - i + 1)`, so the product equals
`1 / C(N-1, k) = 1 / C(2Q₁, k)` with `Q₁ = (N-1)/2`, matching the normalization `ẽ` in the
projection coefficient `N^l_{m',Q*,Q1}`.
"""
function get_symmetric_polynomials!(dest, roots, b, reg_coeffs)

    dest[1] = one(eltype(dest))

    if b == 0
        return
    elseif b == 1
        dest[2] = sum(roots) * reg_coeffs[1]
        return
    end

    dest[2:end] .= zero(eltype(dest))

    for i in eachindex(roots)
        r = roots[i]
        for j in min(i, b):-1:1
            @inbounds dest[j+1] += r * dest[j] * reg_coeffs[j]
        end
    end

    return
end
