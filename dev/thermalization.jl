mutable struct ThermalizationState
    ψ::Ψproj
    θ::Vector{Float64}
    ϕ::Vector{Float64}
    logpdf::Float64
end

function Base.copy!(obj_current::ThermalizationState, obj_next::ThermalizationState, sampling_iter::Int64)
    copy!(obj_current.ψ, obj_next.ψ, sampling_iter)
    obj_current.θ[sampling_iter] = obj_next.θ[sampling_iter]
    obj_current.ϕ[sampling_iter] = obj_next.ϕ[sampling_iter]
    obj_current.logpdf = obj_next.logpdf
    return
end

function Base.copy!(obj_current::ThermalizationState, obj_next::ThermalizationState)
    copy!(obj_current.ψ, obj_next.ψ)
    obj_current.θ .= obj_next.θ
    obj_current.ϕ .= obj_next.ϕ
    obj_current.logpdf = obj_next.logpdf
    return
end

function Base.copy(obj_current::ThermalizationState)
    return ThermalizationState(copy(obj_current.ψ), copy(obj_current.θ), copy(obj_current.ϕ), obj_current.logpdf)
end

function thermalization_step!(RNG::AbstractRNG, logpdf!::Function, obj_current::ThermalizationState, obj_next::ThermalizationState, σ::Float64, sampling_iter::Int64)

    obj_next.θ[sampling_iter], obj_next.ϕ[sampling_iter] = proposal(RNG, obj_current.θ[sampling_iter], obj_current.ϕ[sampling_iter], σ)
    update_wavefunction!(obj_next.ψ, obj_next.θ[sampling_iter], obj_next.ϕ[sampling_iter], sampling_iter)
    logpdf!(obj_next)

    sample_accepted::Bool = false
    if obj_next.logpdf - obj_current.logpdf >= log(rand())

        copy!(obj_current, obj_next, sampling_iter)
        sample_accepted = true
    else
        copy!(obj_next, obj_current, sampling_iter)
    end

    return sample_accepted

end

function thermalize!(RNG::AbstractRNG, Ψcurrent::Ψproj, θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64}, logpdf::Function, num_thermalization::Int64 = 5 * 10^5, σ0::Float64 = pi / sqrt(12.0))

    update_wavefunction!(Ψcurrent, θcurrent, ϕcurrent)
    obj_current = ThermalizationState(Ψcurrent, θcurrent, ϕcurrent, 0.0)

    function logpdf!(obj::ThermalizationState)
        obj.logpdf = logpdf(obj.ψ) ## Okay this is good, I think at least.
        return
    end

    logpdf!(obj_current)
    obj_next = copy(obj_current)

    acceptance_target::Float64 = 0.5 ### Gibbs sampling.
    a::Float64, b::Float64 = arm_parameters(acceptance_target, 3.0)

    num_samples_accepted_thermalization::Int64 = 0
    δ::Float64 = 1.0

    σ = σ0
    tuning_schedule::Vector{Int64} = round.(Int64, exp.(LinRange(log(10.0), log(num_thermalization), 25)))
    sampling_iter::Int64 = 1
    t0::Float64 = time()

    for monte_carlo_iter in 1:num_thermalization

        num_samples_accepted_thermalization += Int(thermalization_step!(RNG, logpdf!, obj_current, obj_next, σ, sampling_iter))

        if monte_carlo_iter in tuning_schedule

            δ = arm_scale_factor(num_samples_accepted_thermalization / monte_carlo_iter, acceptance_target, a, b)

            σ *= δ

        end
        sampling_iter = mod(sampling_iter, Ψcurrent.system_size) + 1

    end

    return time() - t0, σ, num_samples_accepted_thermalization / num_thermalization
end
