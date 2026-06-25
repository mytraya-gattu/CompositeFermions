module LegendrePolynomials
export legendre_polynomials!
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
