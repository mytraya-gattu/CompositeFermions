# Eigenstates of J_y in the J_z basis, used to assemble Wigner-d / monopole-harmonic
# matrix elements. Results are cached to `tempdir()` keyed by angular momentum.

"""
    calculate_j_y_eigenstates(l::Rational) -> Dict{NTuple{3,Rational{Int64}},ComplexF64}

Return a dictionary `d[(μ, m1, m2)] = ⟨l,m1|l,μ⟩_y · conj(⟨l,m2|l,μ⟩_y)`, i.e. the product
of `J_y`-eigenvector components needed to build rotation / Wigner-d matrix elements at
angular momentum `l`. The result is memoised on disk in `tempdir()`.
"""
function calculate_j_y_eigenstates(l::Rational)

    @assert denominator(2 * l) == 1 "Invalid angular momentum."

    filename = joinpath(tempdir(), "j_y_eigenstates_at_angular_momentum_$(numerator(l))_$(denominator(l)).jls")
    if isfile(filename)
        return deserialize(filename)
    end

    dim_j = numerator(2 * l + 1)

    j_y_in_j_z_basis = zeros(ComplexF64, dim_j, dim_j)

    for m = -l:l-1
        miter = round(Int64, m + l + 1)
        j_y_in_j_z_basis[miter+1, miter] = sqrt(l * (l + 1) - m * (m + 1)) / (2.0im)
    end

    for m = -l+1:l
        miter = round(Int64, m + l + 1)
        j_y_in_j_z_basis[miter-1, miter] = -sqrt(l * (l + 1) - m * (m - 1)) / (2.0im)
    end

    _, jy_eigvecs = eigen(j_y_in_j_z_basis)
    ans_dict = Dict{NTuple{3,Rational{Int64}},ComplexF64}()
    for μ in collect(-l:1:l)
        for m1 in collect(-l:1:l)
            for m2 in collect(-l:1:l)
                ans_dict[(μ, m1, m2)] =
                    jy_eigvecs[round(Int64, m1 + l + 1), round(Int64, μ + l + 1)] *
                    conj(jy_eigvecs[round(Int64, m2 + l + 1), round(Int64, μ + l + 1)])
            end
        end
    end

    serialize(filename, ans_dict)
    return ans_dict

end
