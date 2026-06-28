# Dump Julia reference values for the C++ cross-check. Uses deterministic (θ,φ) written to
# CSV (read verbatim by the C++ test, so configs match exactly across languages).
#
# Run from the repo root:  julia +lts --project=. cpp/reference/dump_reference.jl

using CFsOnSphere
using LinearAlgebra

const OUT = @__DIR__

write_vec(path, v) = open(path, "w") do io
    for x in v
        println(io, x)
    end
end

# Complex matrix: first line "rows cols", then rows*cols lines "re im" in column-major order.
function write_cmat(path, M)
    open(path, "w") do io
        println(io, size(M, 1), " ", size(M, 2))
        for x in M
            println(io, real(x), " ", imag(x))
        end
    end
end

write_scalar(path, z) = open(path, "w") do io
    println(io, real(z), " ", imag(z))
end

fixed_theta(N) = [acos(1.0 - 2.0 * (i - 0.5) / N) for i in 1:N]
fixed_phi(N)   = [(-pi + 2pi * (i - 1) / N) + 0.13 * i for i in 1:N]

# --- 1. ESP isolated -------------------------------------------------------------------
let
    nroots = 12
    roots = [complex(cos(0.7i), sin(0.3i - 0.2)) for i in 1:nroots]   # deterministic
    b = 8
    write_cmat(joinpath(OUT, "esp_roots.csv"), reshape(roots, :, 1))
    Npart = 13
    reg = [i / ((Npart - 1) - i + 1) for i in 1:b]
    dest = zeros(ComplexF64, b + 1)
    get_symmetric_polynomials!(dest, roots, b, reg)
    write_cmat(joinpath(OUT, "esp.csv"), reshape(dest, :, 1))
end

# --- 2. projection_coeff isolated -----------------------------------------------------
let
    N = 10
    two_Qstar = 2          # Q* = 1
    rows = String[]
    Q1 = (N - 1) // 2
    two_Q1 = N - 1
    for two_L in (2, 4, 6)
        L = two_L // 2
        Qstar = two_Qstar // 2
        for two_m in two_Qstar:2:two_L
            m = two_m // 2
            val = CFsOnSphere.projection_coeff(L, Qstar, Q1, m)
            push!(rows, "$(two_L) $(two_Qstar) $(two_Q1) $(two_m) $(val)")
        end
    end
    open(joinpath(OUT, "projection_coeff.csv"), "w") do io
        for r in rows
            println(io, r)
        end
    end
end

# --- 3. PsiProj slater_det and PsiUnproj slater_det -----------------------------------
function dump_proj(tag, N, n, p)
    Qstar = (N // n - n) // 2
    l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar)+abs(n)-1) for Lz in -L:1:L]
    Nsys = length(l_m_list)
    θ = fixed_theta(Nsys); φ = fixed_phi(Nsys)
    ψ = Ψproj(Qstar, p, Nsys, l_m_list)
    update_wavefunction!(ψ, θ, φ)
    write_vec(joinpath(OUT, "theta_$(tag).csv"), θ)
    write_vec(joinpath(OUT, "phi_$(tag).csv"), φ)
    write_cmat(joinpath(OUT, "slater_$(tag).csv"), ψ.slater_det)
    write_scalar(joinpath(OUT, "jastrow_$(tag).csv"), ψ.jastrow_factor_log)
end

function dump_unproj(tag, N, n, p)
    Qstar = (N // n - n) // 2
    l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar)+abs(n)-1) for Lz in -L:1:L]
    Nsys = length(l_m_list)
    θ = fixed_theta(Nsys); φ = fixed_phi(Nsys)
    ψ = Ψunproj(Qstar, p, Nsys, l_m_list)
    update_wavefunction!(ψ, θ, φ)
    write_vec(joinpath(OUT, "theta_$(tag).csv"), θ)
    write_vec(joinpath(OUT, "phi_$(tag).csv"), φ)
    write_cmat(joinpath(OUT, "slater_$(tag).csv"), ψ.slater_det)
    write_scalar(joinpath(OUT, "jastrow_$(tag).csv"), ψ.jastrow_factor_log)
end

dump_proj("proj1", 10, 2, 1)
dump_unproj("unproj", 9, 3, 1)

println("wrote reference CSVs to ", OUT)
