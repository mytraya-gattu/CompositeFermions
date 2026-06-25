module LegendrePolynomials
export legendre_polynomials!
"""
    legendre_polynomials!(res, x, kmax::Int64)

Compute the Legendre polynomials up to degree `kmax` at point `x`, storing the results in pre-allocated array `res`.

The function implements the recurrence relation for Legendre polynomials:
Pₙ₊₁(x) = ((2n+1)xPₙ(x) - nPₙ₋₁(x))/(n+1)

# Arguments
- `res::AbstractVector`: Pre-allocated destination array to store results, must be of length `kmax+1`
- `x::Number`: Point at which to evaluate the polynomials
- `kmax::Int64`: Maximum degree of Legendre polynomials to compute

# Notes
- The function modifies the input array `res` in-place
- Output array includes polynomials from degree 0 to kmax
- res[k] contains the (k-1)th degree Legendre polynomial value

"""
function legendre_polynomials!(res, x, kmax::Int64) ### Res

    @assert length(res) == kmax + 1 "Destination array has wrong dimensions."

    res[1] = 1.0

    if kmax == 0
        return
    end

    res[2] = x

    for k in 3:(kmax + 1)

        n = k - 2
        @inbounds res[k] = ((2.0 * n + 1.0) * x * res[k - 1] - n * res[k - 2]) / (n + 1.0)
    end

    return

end
end
