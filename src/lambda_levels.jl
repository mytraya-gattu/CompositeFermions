# Convenience builders for the (Qstar, l_m_list) of the standard composite-fermion states,
# factoring out the boilerplate repeated across samplers.

"""
    cf_ground_state_lm(N, n, p) -> (Qstar, l_m_list)

Effective monopole strength and occupied `(L, Lz)` orbitals for the CF ground state of `N`
electrons in `n` filled Λ-levels at filling `ν = n/(2np+1)`:
`Qstar = (N//n - n)//2`, with all `Lz` filled in Λ-levels `|Qstar| … |Qstar|+|n|-1`.
"""
function cf_ground_state_lm(N::Int, n::Int, p::Int)
    Qstar = (N // n - n) // 2
    l_m_list = NTuple{2,Rational{Int64}}[(L, Lz) for L in abs(Qstar):1:(abs(Qstar)+abs(n)-1) for Lz in -L:1:L]
    return Qstar, l_m_list
end

"""
    cf_quasihole_lm(N, n, p) -> (Qstar, l_m_list)

As [`cf_ground_state_lm`](@ref) but with one quasihole: the top orbital `(Lqh, Lqh)` of the
highest occupied Λ-level (`Lqh = |Qstar|+|n|-1`) is removed.
"""
function cf_quasihole_lm(N::Int, n::Int, p::Int)
    Qstar = (N // n - n) // 2
    Lqh = abs(Qstar) + abs(n) - 1
    l_m_list = NTuple{2,Rational{Int64}}[(L, Lz) for L in abs(Qstar):1:(abs(Qstar)+abs(n)-1) for Lz in -L:1:L if !(L == Lqh && Lz == Lqh)]
    return Qstar, l_m_list
end

"""
    cf_quasiparticle_lm(N, n, p) -> (Qstar, l_m_list)

As [`cf_ground_state_lm`](@ref) but with one quasiparticle: the orbital `(Lqp, Lqp)` in the
next Λ-level (`Lqp = |Qstar|+|n|`) is added.
"""
function cf_quasiparticle_lm(N::Int, n::Int, p::Int)
    Qstar = (N // n - n) // 2
    Lqp = abs(Qstar) + abs(n)
    l_m_list = NTuple{2,Rational{Int64}}[(L, Lz) for L in abs(Qstar):1:(abs(Qstar)+abs(n)-1) for Lz in -L:1:L]
    push!(l_m_list, (Lqp, Lqp))
    return Qstar, l_m_list
end
