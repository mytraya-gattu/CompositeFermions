# Collect the per-system energy summaries of the validation campaign and compare them to
# ref/cf_ground_state_energies.csv. Prints a table (one row per system: E_measured ± σ,
# E_reference ± σ, deviation in combined σ, R̂, ESS) and writes it to
# <outdir>/validation_table.csv. Convention (raw vs density-corrected) is chosen by the
# CONVENTION constant below, set from the local benchmark finding.
#
# Usage: julia collect_results.jl <csv> <campaign_outdir>

using JLD2
using Printf

const CONVENTION = "E_raw"   # set to the convention pinned by the local benchmark

function main()
    csv_path = ARGS[1]
    outdir   = ARGS[2]

    rows = []
    for line in readlines(csv_path)[2:end]
        isempty(strip(line)) && continue
        f = split(line, ',')
        N, n, p = parse(Int, f[1]), parse(Int, f[2]), parse(Int, f[3])
        Eref, σref = parse(Float64, f[4]), parse(Float64, f[5])

        sfile = joinpath(outdir, "N$(N)_n$(n)_p$(p)", "energy_summary.jld2")
        if !isfile(sfile)
            push!(rows, (N, n, p, NaN, NaN, Eref, σref, NaN, NaN, NaN))
            continue
        end
        s = load(sfile)
        E, σE = s[CONVENTION], s[CONVENTION * " stderr"]
        dev = (E - Eref) / sqrt(σE^2 + σref^2)
        push!(rows, (N, n, p, E, σE, Eref, σref, dev, s["rhat"], s["ess"]))
    end

    @printf("%4s %3s %3s  %12s %10s  %12s %10s  %8s  %7s %9s\n",
            "N", "n", "p", "E_meas", "σ_meas", "E_ref", "σ_ref", "dev[σ]", "R̂", "ESS")
    open(joinpath(outdir, "validation_table.csv"), "w") do io
        println(io, "N,n,p,E_meas,E_meas_stderr,E_ref,E_ref_stderr,deviation_sigma,rhat,ess")
        for r in rows
            @printf("%4d %3d %3d  %12.6f %10.6f  %12.6f %10.6f  %8.2f  %7.4f %9.1f\n", r...)
            println(io, join(r, ','))
        end
    end

    done = count(r -> !isnan(r[4]), rows)
    bad  = [r for r in rows if !isnan(r[8]) && abs(r[8]) > 4.0]
    println("\n$done/$(length(rows)) systems complete; $(length(bad)) with |deviation| > 4σ.")
    for r in bad
        println("  DEVIANT: N=$(r[1]) n=$(r[2]) p=$(r[3]) dev=$(round(r[8], digits=2))σ")
    end
end

main()
