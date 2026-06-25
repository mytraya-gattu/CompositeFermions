# Elementary symmetric polynomials (ESP) used by the Jain-Kamilla projection.
#
# Each root may be taken with an integer multiplicity `mult` ≥ 1. This corresponds to the
# JK projection of Y_{Q*,l,m} Jᵢ^{mult} (jk_type = mult): the N-1 ratio roots of Jᵢ each
# appear `mult` times. `mult == 1` and `mult == 2` use hand-optimised recurrences (the
# common ν = n/(2n±1) cases); `mult ≥ 3` uses the general regularized binomial convolution
#
#   ẽ_d ← Σ_{k=0}^{min(d,mult)} C(mult,k) · rᵏ · (∏_{j=d-k+1}^{d} reg[j]) · ẽ_{d-k},
#
# iterating the degree `d` from high to low so the lower-degree entries read on the
# right-hand side still hold their previous-root values. With mult=1,2 this reduces exactly
# to the optimised branches (used as a correctness check in the tests).

"""
    get_symmetric_polynomials(roots, b; mult::Int = 1)

Allocating reference implementation (not exported): returns a length-`b+1` vector whose
`k+1`-th entry is the elementary symmetric polynomial of degree `k` in `roots`, each root
taken with multiplicity `mult`. Used to cross-check the in-place versions.
"""
function get_symmetric_polynomials(roots, b; mult::Int = 1)

    dest = zeros(ComplexF64, b + 1)
    dest[1] = one(ComplexF64)

    if b == 0
        return dest
    elseif b == 1
        dest[2] = mult * sum(roots)
        return dest
    end

    if mult == 1
        for i in eachindex(roots)
            r = roots[i]
            for j in min(i, b):-1:1
                @inbounds dest[j+1] += r * dest[j]
            end
        end
    elseif mult == 2
        for i in eachindex(roots)
            r  = roots[i]
            tr = 2 * r
            r2 = r * r
            upper = min(2i, b)
            for k in upper:-1:2
                @inbounds dest[k+1] += tr * dest[k] + r2 * dest[k-1]
            end
            @inbounds dest[2] += tr * dest[1]
        end
    else
        for i in eachindex(roots)
            r = roots[i]
            upper = min(i * mult, b)
            for d in upper:-1:1
                acc = dest[d+1]
                rk = one(r)
                for k in 1:min(d, mult)
                    rk *= r
                    @inbounds acc += binomial(mult, k) * rk * dest[d-k+1]
                end
                @inbounds dest[d+1] = acc
            end
        end
    end

    return dest
end

"""
    get_symmetric_polynomials!(dest, roots, b::Int; mult::Int = 1)

In-place: fill `dest[1:b+1]` with the elementary symmetric polynomials of degrees `0…b` in
`roots`, each root taken with multiplicity `mult` (≥ 1).

# Arguments
- `dest`: destination array of length ≥ `b+1`.
- `roots`: the (distinct) roots.
- `b::Int`: maximum ESP degree to compute.

# Keywords
- `mult::Int = 1`: multiplicity of every root. `1` and `2` use optimised recurrences;
  `≥ 3` uses the general binomial convolution.
"""
function get_symmetric_polynomials!(dest, roots, b; mult::Int = 1)

    dest[1] = one(eltype(dest))

    if b == 0
        return
    elseif b == 1
        dest[2] = mult * sum(roots)
        return
    end

    dest[2:end] .= zero(eltype(dest))

    if mult == 1
        for i in eachindex(roots)
            r = roots[i]
            for j in min(i, b):-1:1
                @inbounds dest[j+1] += r * dest[j]
            end
        end
    elseif mult == 2
        for i in eachindex(roots)
            r  = roots[i]
            tr = 2 * r
            r2 = r * r
            upper = min(2i, b)
            for k in upper:-1:2
                @inbounds dest[k+1] += tr * dest[k] + r2 * dest[k-1]
            end
            @inbounds dest[2] += tr * dest[1]
        end
    else
        for i in eachindex(roots)
            r = roots[i]
            upper = min(i * mult, b)
            for d in upper:-1:1
                acc = dest[d+1]
                rk = one(r)
                for k in 1:min(d, mult)
                    rk *= r
                    @inbounds acc += binomial(mult, k) * rk * dest[d-k+1]
                end
                @inbounds dest[d+1] = acc
            end
        end
    end

    return
end

"""
    get_symmetric_polynomials!(dest, roots, b::Int, reg_coeffs; mult::Int = 1)

In-place, regularized variant: `dest[k+1]` holds the degree-`k` elementary symmetric
polynomial (each root with multiplicity `mult`) multiplied by `∏_{i=1}^{k} reg_coeffs[i]`.

For the JK projection `reg_coeffs[i] = i / (mult·(N-1) - i + 1)`, so the product equals
`1 / C(mult·(N-1), k) = 1 / C(2Q₁, k)` with `Q₁ = mult·(N-1)/2`, matching the
normalization `ẽ` in the projection coefficient `N^l_{m',Q*,Q1}`.
"""
function get_symmetric_polynomials!(dest, roots, b, reg_coeffs; mult::Int = 1)

    dest[1] = one(eltype(dest))

    if b == 0
        return
    elseif b == 1
        dest[2] = mult * sum(roots) * reg_coeffs[1]
        return
    end

    dest[2:end] .= zero(eltype(dest))

    if mult == 1
        for i in eachindex(roots)
            r = roots[i]
            for j in min(i, b):-1:1
                @inbounds dest[j+1] += r * dest[j] * reg_coeffs[j]
            end
        end
    elseif mult == 2
        for i in eachindex(roots)
            r  = roots[i]
            tr = 2 * r
            r2 = r * r
            upper = min(2i, b)
            for k in upper:-1:2
                @inbounds dest[k+1] += tr * reg_coeffs[k] * dest[k] +
                                       r2 * reg_coeffs[k] * reg_coeffs[k-1] * dest[k-1]
            end
            @inbounds dest[2] += tr * reg_coeffs[1] * dest[1]
        end
    else
        for i in eachindex(roots)
            r = roots[i]
            upper = min(i * mult, b)
            for d in upper:-1:1
                acc = dest[d+1]
                rk = one(r)
                regprod = one(eltype(reg_coeffs))
                for k in 1:min(d, mult)
                    rk *= r
                    regprod *= reg_coeffs[d-k+1]
                    @inbounds acc += binomial(mult, k) * rk * regprod * dest[d-k+1]
                end
                @inbounds dest[d+1] = acc
            end
        end
    end

    return
end
