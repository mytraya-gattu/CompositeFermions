# Legendre polynomials by the standard recurrence.

"""
    legendre_polynomials!(res, x, kmax::Int64)

Compute the Legendre polynomials up to degree `kmax` at point `x`, storing the results in
the pre-allocated array `res` (length `kmax+1`); `res[k]` holds `P_{k-1}(x)`.

Uses the recurrence `Pₙ₊₁(x) = ((2n+1) x Pₙ(x) - n Pₙ₋₁(x)) / (n+1)`. Modifies `res`
in-place.
"""
function legendre_polynomials!(res, x, kmax::Int64)

    @assert length(res) == kmax+1 "Destination array has wrong dimensions."

    res[1] = 1.0

    if kmax == 0
        return
    end

    res[2] = x

    for k in 3:kmax+1
        n = k-2
        @inbounds res[k] = ((2.0*n+1.0) * x * res[k-1] - n * res[k-2]) / (n+1.0)
    end

    return

end
