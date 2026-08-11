# Elementary symmetric polynomials (ESP) used by the Jain-Kamilla projection.
#
# The projection binds `q` vortex pairs into each orbital — `[P_LLL Y_{Q*,l,m} Jᵢ^q] / Jᵢ^q`
# with `Jᵢ = ∏_{j≠i}(uᵢvⱼ - uⱼvᵢ)` — so the ESP run over the `N-1` distinct ratio roots each
# taken with multiplicity `q`, i.e. `q(N-1)` roots in all, and `Q₁ = q(N-1)/2`.
#
# `q = 1` (the default) is the standard Jain-Kamilla ansatz: the remaining vortices come from
# the global Jastrow factor `∏(uᵢvⱼ-uⱼvᵢ)^p` outside the projection.
#
# `q = p` binds *every* vortex into the projected orbital. At `ν = 1/m` (so `p = m-1`) this is
# not a variational ansatz but the **exact** LLL projection, provided exactly one composite
# fermion sits outside the lowest Λ level — see `docs/src/theory.md` and the note in
# `single_exciton_nu_1_m/book/book.tex`.

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
    jk_reg_coeffs(system_size, b, q = 1) -> Vector{Float64}

The ESP regularization coefficients used by `Ψproj`/`Ψparton`:
`reg_coeffs[i] = i/(q(N-1) - i + 1)` with `N = system_size`, so the running product is
`∏_{i=1}^{k} reg_coeffs[i] = 1/C(q(N-1), k) = 1/C(2Q₁, k)` with `Q₁ = q(N-1)/2` — the
normalization `ẽ` consumed by [`projection_coeff`](@ref).

For `q = 1` this is the familiar `i/(N-i)`.
"""
jk_reg_coeffs(system_size::Int64, b::Int64, q::Int64 = 1) = [i / (q * (system_size - 1) - i + 1) for i in 1:b]

"""
    get_symmetric_polynomials!(dest, roots, b::Int, reg_coeffs)

In-place, regularized variant: `dest[k+1]` holds the degree-`k` elementary symmetric
polynomial multiplied by `∏_{i=1}^{k} reg_coeffs[i]`.

For the JK projection `reg_coeffs[i] = i / (q(N-1) - i + 1)`, so the product equals
`1 / C(q(N-1), k) = 1 / C(2Q₁, k)` with `Q₁ = q(N-1)/2`, matching the normalization `ẽ` in
the projection coefficient `N^l_{m',Q*,Q1}`.
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

"""
    get_symmetric_polynomials!(dest, roots, b::Int, reg_coeffs, mult::Int)

In-place, regularized, with each root taken with multiplicity `mult`: the ESP are those of
the multiset in which every element of `roots` appears `mult` times, i.e. of
`mult * length(roots)` roots in all.

This is the `q > 1` projection, `[P_LLL Y_{Q*,l,m} Jᵢ^q] / Jᵢ^q`. Roots are visited one copy
at a time so the `min(n, b)` bound (only `e_0 … e_n` can be nonzero after `n` roots) stays
valid; `reg_coeffs` must be built with the matching `q`, see [`jk_reg_coeffs`](@ref).
"""
function get_symmetric_polynomials!(dest, roots, b, reg_coeffs, mult::Int)

    mult == 1 && return get_symmetric_polynomials!(dest, roots, b, reg_coeffs)

    dest[1] = one(eltype(dest))

    if b == 0
        return
    end

    dest[2:end] .= zero(eltype(dest))

    n = 0
    for i in eachindex(roots)
        r = roots[i]
        for _ in 1:mult
            n += 1
            for j in min(n, b):-1:1
                @inbounds dest[j+1] += r * dest[j] * reg_coeffs[j]
            end
        end
    end

    return
end
