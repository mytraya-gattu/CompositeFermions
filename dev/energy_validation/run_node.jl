# Per-node driver for the ground-state-energy validation campaign.
#
# Each Slurm array task owns one full node and calls this script once. The script takes the
# system list from ref/cf_ground_state_energies.csv, deterministically partitions it across
# the array tasks (greedy balance on an estimated wall-time cost), and runs its share
# sequentially — each system's chains run in parallel over all worker processes via
# `run_chains`. Completed systems (energy_summary.jld2 present) are skipped, so a resubmitted
# task resumes where it left off; partially-run systems resume from their chain checkpoints.
#
# Usage (workers already spawned with `julia -p <cores>`):
#   julia -p <n> run_node.jl <csv> <node_index> <n_nodes> <outdir> <n_chains> <num_thermalization> <num_steps>

using Distributed
@everywhere using CFsOnSphere
@everywhere include(joinpath(@__DIR__, "energy_chains.jl"))

function parse_systems(csv_path::String)
    systems = NTuple{3,Int64}[]
    for line in readlines(csv_path)[2:end]
        isempty(strip(line)) && continue
        f = split(line, ',')
        push!(systems, (parse(Int64, f[1]), parse(Int64, f[2]), parse(Int64, f[3])))
    end
    return systems
end

# Wall-time cost model: with n_chains spread over the node's cores, wall time per system is
# roughly (num_thermalization + num_steps) × t_step(N); t_step grows ~ N^α (α from the local
# benchmark; the balance only needs to be rough).
const ALPHA = 2.0
cost(N::Int64) = Float64(N)^ALPHA

"""
    assign_systems(systems, n_nodes) -> Vector{Int}

Deterministic greedy partition: sort by descending cost, place each system on the currently
least-loaded node. Returns the owning node index per system (same order as `systems`).
"""
function assign_systems(systems::Vector{NTuple{3,Int64}}, n_nodes::Int)
    order = sortperm([cost(s[1]) for s in systems], rev = true)
    load = zeros(Float64, n_nodes)
    owner = zeros(Int, length(systems))
    for idx in order
        node = argmin(load)
        owner[idx] = node
        load[node] += cost(systems[idx][1])
    end
    return owner
end

function main()
    csv_path   = ARGS[1]
    node_index = parse(Int, ARGS[2])
    n_nodes    = parse(Int, ARGS[3])
    outdir     = ARGS[4]
    n_chains   = parse(Int, ARGS[5])
    num_thermalization = parse(Int, ARGS[6])
    num_steps  = parse(Int, ARGS[7])

    systems = parse_systems(csv_path)
    owner = assign_systems(systems, n_nodes)
    mine = [systems[i] for i in eachindex(systems) if owner[i] == node_index]

    println("[node $node_index/$n_nodes] $(length(mine)) systems: $mine")
    flush(stdout)

    for (N, n, p) in mine
        dir = joinpath(outdir, "N$(N)_n$(n)_p$(p)")
        if isfile(joinpath(dir, "energy_summary.jld2"))
            println("[node $node_index] N=$N n=$n p=$p already complete; skipping.")
            flush(stdout)
            continue
        end
        resume = isdir(dir) && !isempty(filter(f -> startswith(f, "chain_"), readdir(dir)))
        t0 = time()
        summary = run_energy_system(dir, N, n, p;
                                    n_chains = n_chains,
                                    num_thermalization = num_thermalization,
                                    num_steps = num_steps,
                                    record_every = 1_000,
                                    checkpoint_every = 500_000,
                                    resume = resume,
                                    monitor_every = 120.0)
        println("[node $node_index] N=$N n=$n p=$p done in $(round(time()-t0, digits=1)) s: " *
                "E_raw = $(summary["E_raw"]) ± $(summary["E_raw stderr"]), " *
                "R̂ = $(summary["rhat"]), ESS = $(summary["ess"])")
        flush(stdout)
    end

    println("[node $node_index] all assigned systems complete.")
end

main()
