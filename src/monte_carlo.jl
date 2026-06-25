# Monte Carlo machinery: random initial positions, the sphere proposal, ARM step-size
# adaptation, the generic Gibbs thermalization driver, density binning, and the
# shared-row determinant-ratio builder.

"""
    rand_θ_ϕ_gen(RNG, n_samples::Int) -> (θlist, ϕlist)

Generate `n_samples` points uniformly on the unit sphere, returned as polar angles
`θ ∈ [0, π]` and azimuthal angles `ϕ ∈ (-π, π]` (via `atan`). Uses the Marsaglia method
(normalize 3D Gaussians).
"""
function rand_θ_ϕ_gen(RNG, n_samples::Int)
    Xmat = randn(RNG, Float64, 3, n_samples)
    θlist = zeros(Float64, n_samples)
    ϕlist = zeros(Float64, n_samples)
    @inbounds for i in axes(Xmat, 2)
        x = Xmat[1, i]; y = Xmat[2, i]; z = Xmat[3, i]
        r = sqrt(x * x + y * y + z * z)
        θlist[i] = acos(clamp(z / r, -1.0, 1.0))
        ϕlist[i] = atan(y, x)
    end
    return θlist, ϕlist
end

"""
    proposal(RNG, θcurrent::Float64, ϕcurrent::Float64, σ::Float64) -> (θnew, ϕnew)

Propose a new point on the sphere by rotating the current position along a great circle by a
Gaussian angle `δθ ~ N(0, σ)` in a uniformly random tangent direction.

The current position `r̂` is taken as a Cartesian unit vector; a random 3D Gaussian is
projected onto the tangent plane at `r̂` and normalized to give the direction `ê`, and the
new point is `r̂ cos(δθ) + ê sin(δθ)`. This proposal is isotropic about the current point
(hence symmetric, preserving detailed balance). Returns `(θ, ϕ)` in the same convention as
[`rand_θ_ϕ_gen`](@ref).
"""
function proposal(RNG, θcurrent::Float64, ϕcurrent::Float64, σ::Float64)

    sθ, cθ = sincos(θcurrent)
    sϕ, cϕ = sincos(ϕcurrent)

    # Current position as a Cartesian unit vector.
    rx = sθ * cϕ
    ry = sθ * sϕ
    rz = cθ

    # Random direction in the tangent plane (Gram-Schmidt against r̂).
    wx = randn(RNG); wy = randn(RNG); wz = randn(RNG)
    d = wx * rx + wy * ry + wz * rz
    ex = wx - d * rx
    ey = wy - d * ry
    ez = wz - d * rz
    en = sqrt(ex * ex + ey * ey + ez * ez)
    en < 1e-300 && return θcurrent, ϕcurrent
    ex /= en; ey /= en; ez /= en

    # Rotate along the great circle by a Gaussian angle.
    sδ, cδ = sincos(randn(RNG) * σ)
    x = cδ * rx + sδ * ex
    y = cδ * ry + sδ * ey
    z = cδ * rz + sδ * ez

    return acos(clamp(z, -1.0, 1.0)), atan(y, x)
end

"""
    arm_parameters(ideal_acceptance_ratio::Float64, r::Float64) -> (a, b)

Precompute the parameters `(a, b)` of the ARM (acceptance-rate monitoring) step-size scheme
for a target acceptance ratio, via a fixed-point iteration.
"""
function arm_parameters(ideal_acceptance_ratio::Float64, r::Float64)
    a = 1.0
    b = 0.0
    for i = 1:1000
        c = (a * ideal_acceptance_ratio + b)^r
        a = (a * ideal_acceptance_ratio + b)^(1 / r) - c
        b = c
    end
    return a, b
end

"""
    arm_scale_factor(p, p_i, a, b) -> Float64

Multiplicative step-size rescale factor given current acceptance `p`, target `p_i`, and ARM
parameters `(a, b)`: `> 1` when acceptance is too high, `< 1` when too low.
"""
function arm_scale_factor(p, p_i, a, b)
    return log(a * p_i + b) / log(a * p + b)
end

"""
    gibbs_thermalization!(RNG, Ψcurrent, Ψnext, θcurrent, ϕcurrent, θnext, ϕnext, σinit, logpdf, num_thermalization)

Thermalize a single-particle Gibbs / Metropolis-Hastings walk targeting `logpdf(Ψ)` (a real
scalar), tuning the step size `σ` toward a 50% acceptance rate via the ARM scheme. Works for
any wavefunction type (`Ψproj`, `Ψparton`, `ΨoneLL`, `Ψunproj`).

Returns `(sampling_iter, σ, δt_therm, acceptance_rate)`.
"""
function gibbs_thermalization!(RNG, Ψcurrent::T, Ψnext::T, θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64}, θnext::Vector{Float64}, ϕnext::Vector{Float64}, σinit::Float64, logpdf::Function, num_thermalization::Int64) where {T <: Union{Ψproj, Ψparton, ΨoneLL, Ψunproj}}

    acceptance_target::Float64 = 0.50
    a::Float64, b::Float64 = arm_parameters(acceptance_target, 3.0)

    num_samples_accepted_thermalization::Int64 = 0
    δ::Float64 = 1.0
    σ::Float64 = σinit

    logpdf_current::Float64 = 0.0
    logpdf_next::Float64 = 0.0

    update_wavefunction!(Ψcurrent, θcurrent, ϕcurrent)
    copy!(Ψnext, Ψcurrent)

    logpdf_current = logpdf(Ψcurrent)

    tuning_schedule::Vector{Int64} = round.(Int64, exp.(LinRange(log(10.0), log(num_thermalization), 25)))

    sampling_iter::Int64 = 1
    t0::Float64 = time()
    for monte_carlo_iter in 1:num_thermalization

        θnext[sampling_iter], ϕnext[sampling_iter] = proposal(RNG, θcurrent[sampling_iter], ϕcurrent[sampling_iter], σ)
        update_wavefunction!(Ψnext, θnext[sampling_iter], ϕnext[sampling_iter], sampling_iter)

        logpdf_next = logpdf(Ψnext)

        if logpdf_next - logpdf_current >= log(rand())

            θcurrent[sampling_iter] = θnext[sampling_iter]
            ϕcurrent[sampling_iter] = ϕnext[sampling_iter]

            copy!(Ψcurrent, Ψnext, sampling_iter)
            logpdf_current = logpdf_next

            num_samples_accepted_thermalization += 1

        else

            θnext[sampling_iter] = θcurrent[sampling_iter]
            ϕnext[sampling_iter] = ϕcurrent[sampling_iter]

            copy!(Ψnext, Ψcurrent, sampling_iter)
            logpdf_next = logpdf_current

        end

        if monte_carlo_iter ∈ tuning_schedule

            δ = arm_scale_factor(num_samples_accepted_thermalization/monte_carlo_iter, acceptance_target, a, b)
            σ *= δ
        end

        sampling_iter = mod(sampling_iter, Ψcurrent.system_size) + 1

    end

    δt_therm::Float64 = time()-t0
    return sampling_iter, σ, δt_therm, num_samples_accepted_thermalization/num_thermalization
end

"""
    update_density!(θmesh, θcurrent, accumulated_density)

Accumulate a 1D polar-angle density histogram in place: for each `θ` in `θcurrent`, increment
the bin of `accumulated_density` located by `searchsortedfirst(θmesh, θ) - 1`.
"""
function update_density!(θmesh::Vector{Float64}, θcurrent::Vector{Float64}, accumulated_density::Vector{Float64})

    for θ in θcurrent
        accumulated_density[searchsortedfirst(θmesh, θ)-1] += 1.0
    end

    return

end

"""
    update_density!(θmesh, ϕmesh, θcurrent, ϕcurrent, accumulated_density)

Accumulate a 2D `(θ, ϕ)` density histogram in place.
"""
function update_density!(θmesh::Vector{Float64}, ϕmesh::Vector{Float64}, θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64}, accumulated_density::Matrix{Float64})

    for iter in eachindex(θcurrent)
        accumulated_density[searchsortedfirst(θmesh, θcurrent[iter])-1, searchsortedfirst(ϕmesh, ϕcurrent[iter])-1] += 1.0
    end

    return

end

"""
    construct_det_ratios(denominator_rows, numerator_rows)

Build a closure `helper!(res, S, Sinv)` computing the determinant ratios
`det(S[numerator_rows[i], :]) / det(S[denominator_rows, :])` efficiently by reducing to the
sub-determinant over rows not shared with the denominator. `Sinv` is the inverse of
`S[denominator_rows, :]`. All numerator row-sets must have the same length as
`denominator_rows`; matrix elements are assumed `ComplexF64`.
"""
function construct_det_ratios(denominator_rows::Vector{Int64}, numerator_rows::Vector{Vector{Int64}})

    @assert length(denominator_rows) == length(numerator_rows[1]) && length(unique(length, numerator_rows)) == 1 "Lengths of denominator and numerator rows do not match."

    common_elements = copy(denominator_rows)

    for numerator_row in numerator_rows
        common_elements = intersect(common_elements, numerator_row)
    end

    denominator_diff = setdiff(denominator_rows, common_elements)

    numerator_diffs = [setdiff(row, common_elements) for row in numerator_rows]
    numerator_diffs_union = union(numerator_diffs...)

    iters = Matrix{Int64}(undef, length(denominator_diff), length(numerator_rows))
    for i in eachindex(numerator_rows)

        numerator_diff = numerator_diffs[i]

        for j in eachindex(denominator_diff)

            iters[j, i] = findfirst(isequal(numerator_diff[j]), numerator_diffs_union)

        end

    end

    function get_sign(rows1, rows2)

        p = [findfirst(isequal(elem), rows1) for elem in rows2]

        return levicivita(p)

    end

    denom_sign = get_sign(denominator_rows, vcat(denominator_diff, common_elements))
    numerators_signs = [get_sign(numerator_rows[iter], vcat(numerator_diffs[iter], common_elements)) for iter in eachindex(numerator_rows)]

    temp = zeros(ComplexF64, length(numerator_diffs_union), length(denominator_diff))

    intra_denom_diff = [findfirst(isequal(elem), denominator_rows) for elem in denominator_diff]

    function helper!(res, S, Sinv)

        @views @inbounds temp .= S[numerator_diffs_union, :] * Sinv[:, intra_denom_diff]

        @simd for i in eachindex(numerator_rows)

            @views @inbounds res[i] = det(temp[iters[:, i],:]) * numerators_signs[i] / denom_sign

        end

        return
    end

    return helper!

end
