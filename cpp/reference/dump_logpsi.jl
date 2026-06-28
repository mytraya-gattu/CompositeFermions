# Large cross-check: for many systems (sizes × Λ-level fillings × proj/unproj),
# generate K random configurations and dump log Ψ = logdet(slater_det) + jastrow_factor_log.
# Configs are written to CSV and read verbatim by the C++ test, so the comparison is purely
# numerical. Run from repo root:  julia +lts --project=. cpp/reference/dump_logpsi.jl
using CFsOnSphere
using Random
using LinearAlgebra

const OUT = @__DIR__
const K = 1000   # configurations per system

# (tag, kind, N, n, p)
const SYSTEMS = [
    ("p1_4_1",  "proj",   4, 1, 1),
    ("p1_10_2", "proj",  10, 2, 1),
    ("p1_9_3",  "proj",   9, 3, 1),
    ("p1_16_4", "proj",  16, 4, 1),
    ("p1_20_2", "proj",  20, 2, 1),
    ("p1_32_4", "proj",  32, 4, 1),
    ("p2_10_2", "proj",  10, 2, 2),
    ("p2_9_3",  "proj",   9, 3, 2),
    ("p2_16_4", "proj",  16, 4, 4),
    ("u_9_3",   "unproj", 9, 3, 1),
    ("u_16_4",  "unproj",16, 4, 1),
    ("u_25_5",  "unproj",25, 5, 1),
]

function build_lm(N, n)
    Qstar = (N // n - n) // 2
    l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar)+abs(n)-1) for Lz in -L:1:L]
    return Qstar, l_m_list
end

# manifest
open(joinpath(OUT, "logpsi_systems.csv"), "w") do io
    for (tag, kind, N, n, p) in SYSTEMS
        println(io, "$(tag) $(kind) $(N) $(n) $(p) $(K)")
    end
end

tim_io = open(joinpath(OUT, "timing_julia.csv"), "w")
for (si, (tag, kind, N, n, p)) in enumerate(SYSTEMS)
    Qstar, lm = build_lm(N, n)
    Nsys = length(lm)
    ψ = kind == "proj" ? Ψproj(Qstar, p, Nsys, lm) : Ψunproj(Qstar, p, Nsys, lm)

    rng = MersenneTwister(1000 + si)
    configs = [rand_θ_ϕ_gen(rng, Nsys) for _ in 1:K]
    logs = Vector{ComplexF64}(undef, K)

    update_wavefunction!(ψ, configs[1]...)          # warm up JIT (untimed)
    logdet(ψ.slater_det)

    # reference values (one pass)
    for k in 1:K
        θ, φ = configs[k]
        update_wavefunction!(ψ, θ, φ)
        logs[k] = logdet(ψ.slater_det) + ψ.jastrow_factor_log
    end

    # timing: average over R passes (after the warm-up above)
    R = 10
    acc = 0.0
    t = @elapsed for _ in 1:R, k in 1:K
        θ, φ = configs[k]
        update_wavefunction!(ψ, θ, φ)
        acc += real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)
    end
    t /= R
    acc == Inf && println(acc)  # keep `acc` live so the work is not elided

    open(joinpath(OUT, "cfg_$(tag).csv"), "w") do io
        for (θ, φ) in configs
            for x in θ; print(io, x, " "); end
            for x in φ; print(io, x, " "); end
            println(io)
        end
    end
    open(joinpath(OUT, "logpsi_$(tag).csv"), "w") do io
        for z in logs; println(io, real(z), " ", imag(z)); end
    end
    println(tim_io, "$(tag) $(K) $(t) $(1e6 * t / K)")
    println("dumped $(tag): kind=$(kind) Nsys=$(Nsys) K=$(K)  julia=$(round(1e6*t/K,sigdigits=4)) us/eval")
end
close(tim_io)
println("done -> ", OUT)
