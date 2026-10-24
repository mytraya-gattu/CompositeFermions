module OneBodyWeight

# Real spherical-harmonic scalar field on the unit sphere, in the cf-psiformer
# coefficient layout (fields.py `_harmonic_tables`): flat index l^2 + l + m
# (0-based) holds the cos(m phi) coefficient for m >= 0 and the sin(|m| phi)
# coefficient for m < 0; each m != 0 term carries a weight sqrt(2). The
# Legendre functions are fully normalized with the Condon-Shortley phase, so
# the m = 0 term is c_l0 Y_l0 and the m > 0 pair is sqrt(2)[c Re Y_lm + s Im Y_lm].
#
# Used for the Aharonov-Casher dress: the ideal-band many-body weight is
# |F|^2 exp(2 sum_i chi(Omega_i)) with chi stored in this layout.

export RealHarmonicField, field_value

struct RealHarmonicField
    lmax::Int
    c::Matrix{Float64}     # [l+1, m+1]: weight_m * cos-type coefficient
    s::Matrix{Float64}     # [l+1, m+1]: weight_m * sin-type coefficient (m >= 1)
    a::Matrix{Float64}     # recurrence coefficients, [l+1, m+1], used for l >= m+2
    b::Matrix{Float64}
    pmm::Vector{Float64}   # Pbar_mm / sin^m(theta), Condon-Shortley phase included
    fac::Vector{Float64}   # sqrt(2m+3): Pbar_{m+1,m} = fac[m+1] x Pbar_mm
end

function RealHarmonicField(coeffs::AbstractVector{<:Real})
    n = length(coeffs)
    lmax = isqrt(n) - 1
    (lmax + 1)^2 == n || throw(ArgumentError("coefficient length $n is not (lmax+1)^2"))
    c = zeros(lmax + 1, lmax + 1)
    s = zeros(lmax + 1, lmax + 1)
    a = zeros(lmax + 1, lmax + 1)
    b = zeros(lmax + 1, lmax + 1)
    for l in 0:lmax, m in 0:l
        w = m == 0 ? 1.0 : sqrt(2.0)
        c[l + 1, m + 1] = w * coeffs[l * l + l + m + 1]
        m > 0 && (s[l + 1, m + 1] = w * coeffs[l * l + l - m + 1])
        if l >= m + 2
            a[l + 1, m + 1] = sqrt((4.0 * l^2 - 1.0) / (l^2 - m^2))
            b[l + 1, m + 1] = sqrt((2.0 * l + 1.0) * ((l - 1.0)^2 - m^2) /
                                   ((2.0 * l - 3.0) * (l^2 - m^2)))
        end
    end
    pmm = zeros(lmax + 1)
    pmm[1] = 1.0 / sqrt(4.0 * pi)
    for m in 1:lmax
        pmm[m + 1] = -sqrt((2.0 * m + 1.0) / (2.0 * m)) * pmm[m]
    end
    fac = [sqrt(2.0 * m + 3.0) for m in 0:lmax]
    return RealHarmonicField(lmax, c, s, a, b, pmm, fac)
end

"""
    field_value(F, θ, ϕ) -> Float64

Value of the real harmonic field at (θ, ϕ). Cost O(lmax^2), no allocation.
"""
function field_value(F::RealHarmonicField, θ::Float64, ϕ::Float64)
    st, x = sincos(θ)
    e1 = cis(ϕ)
    em = one(ComplexF64)
    stm = 1.0
    total = 0.0
    lmax = F.lmax
    @inbounds for m in 0:lmax
        p2 = F.pmm[m + 1] * stm
        cs = F.c[m + 1, m + 1] * p2
        ss = F.s[m + 1, m + 1] * p2
        if m < lmax
            p1 = F.fac[m + 1] * x * p2
            cs += F.c[m + 2, m + 1] * p1
            ss += F.s[m + 2, m + 1] * p1
            for l in (m + 2):lmax
                p = F.a[l + 1, m + 1] * x * p1 - F.b[l + 1, m + 1] * p2
                cs += F.c[l + 1, m + 1] * p
                ss += F.s[l + 1, m + 1] * p
                p2, p1 = p1, p
            end
        end
        total += real(em) * cs + imag(em) * ss
        em *= e1
        stm *= st
    end
    return total
end

end # module OneBodyWeight
