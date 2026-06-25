using BenchmarkTools
using Statistics
using Printf

## ─── Full recomputation (current code) ───────────────────────────────────────

function get_symmetric_polynomials_full!(dest, roots, b, reg_coeffs)
    dest[1] = one(eltype(dest))
    if b == 0
        return
    elseif b == 1
        dest[2] = sum(roots) * reg_coeffs[1]
        return
    end
    dest[2:end] .= zero(eltype(dest))
    for i in eachindex(roots)
        r = roots[i]
        for j in min(i, b):-1:1
            @inbounds dest[j + 1] += r * dest[j] * reg_coeffs[j]
        end
    end
    return
end

## ─── Incremental update: remove old root + add new root ──────────────────────

function update_symmetric_polynomials!(dest, r_old, r_new, b, reg_coeffs)
    # Step 1: Remove old root (forward recurrence, k = 1 → b)
    for k in 1:b
        @inbounds dest[k + 1] -= r_old * reg_coeffs[k] * dest[k]
    end
    # Step 2: Add new root (backward recurrence, k = b → 1)
    for k in b:-1:1
        @inbounds dest[k + 1] += r_new * reg_coeffs[k] * dest[k]
    end
    return
end

## ─── Test correctness ────────────────────────────────────────────────────────

function test_correctness(; N_values=[50, 100, 150], kmax_values=1:20)
    println("=" ^ 70)
    println("CORRECTNESS TESTS")
    println("=" ^ 70)
    all_pass = true

    for N in N_values
        for kmax in kmax_values
            reg_coeffs = [i / (N - i) for i in 1:kmax]

            # Generate random roots for one column
            roots = randn(ComplexF64, N - 1)

            # Compute full symmetric polynomials
            dest_full = zeros(ComplexF64, kmax + 1)
            get_symmetric_polynomials_full!(dest_full, roots, kmax, reg_coeffs)

            ## ─── Test Branch 1: single root change (j ≠ iter) ───────────
            dest_inc = copy(dest_full)  # start from current state

            # Pick a random root to change
            idx = rand(1:N-1)
            r_old = roots[idx]
            r_new = randn(ComplexF64)

            # Incremental update
            update_symmetric_polynomials!(dest_inc, r_old, r_new, kmax, reg_coeffs)

            # Full recompute with the new root
            roots_new = copy(roots)
            roots_new[idx] = r_new
            dest_ref = zeros(ComplexF64, kmax + 1)
            get_symmetric_polynomials_full!(dest_ref, roots_new, kmax, reg_coeffs)

            err1 = maximum(abs.(dest_inc .- dest_ref))
            pass1 = err1 < 1e-10

            ## ─── Test Branch 2: all roots change (j == iter) ─────────────
            # For column iter, all roots are different → must do full recompute
            roots_all_new = randn(ComplexF64, N - 1)
            dest_full_new = zeros(ComplexF64, kmax + 1)
            get_symmetric_polynomials_full!(dest_full_new, roots_all_new, kmax, reg_coeffs)

            # Verify full recompute is self-consistent by also building incrementally
            # (change roots one by one from the original)
            dest_sequential = copy(dest_full)
            for i in 1:N-1
                update_symmetric_polynomials!(dest_sequential, roots[i], roots_all_new[i], kmax, reg_coeffs)
            end
            err2 = maximum(abs.(dest_sequential .- dest_full_new))
            pass2 = err2 < 1e-10

            if !pass1 || !pass2
                all_pass = false
                @printf("  FAIL  N=%3d  kmax=%2d  err_single=%.2e  err_all=%.2e\n", N, kmax, err1, err2)
            end
        end
    end

    if all_pass
        println("  All correctness tests PASSED ✓")
    end
    println()
end

## ─── Benchmark: single-particle MC step ──────────────────────────────────────

function bench_full_recompute(esp_matrix, uv_matrix, N, kmax, reg_coeffs)
    # Current code: recompute ALL columns from scratch
    @inbounds for j in axes(esp_matrix, 2)
        get_symmetric_polynomials_full!(
            view(esp_matrix, :, j),
            view(uv_matrix, :, j),
            kmax, reg_coeffs
        )
    end
end

function bench_incremental(esp_matrix, uv_matrix_old, uv_matrix_new, iter, N, kmax, reg_coeffs)
    @inbounds for j in axes(esp_matrix, 2)
        if j == iter
            # Full recompute for the moved particle's column
            get_symmetric_polynomials_full!(
                view(esp_matrix, :, j),
                view(uv_matrix_new, :, j),
                kmax, reg_coeffs
            )
        else
            # Find the root that changed: the entry corresponding to `iter` in column j
            # In the (N-1)×N storage: for pair (i,j) with i<j, entry is at row i of column j
            #                          for pair (i,j) with i>j, entry is at row i-1 of column j
            row = iter < j ? iter : iter - 1
            r_old = uv_matrix_old[row, j]
            r_new = uv_matrix_new[row, j]
            update_symmetric_polynomials!(
                view(esp_matrix, :, j),
                r_old, r_new, kmax, reg_coeffs
            )
        end
    end
end

function run_benchmarks(; N_values=[50, 100, 150], kmax_values=1:20)
    println("=" ^ 70)
    println("BENCHMARKS: full recompute vs incremental (single MC step)")
    println("=" ^ 70)
    @printf("%5s  %5s  %12s  %12s  %8s\n", "N", "kmax", "full (μs)", "incr (μs)", "speedup")
    println("-" ^ 50)

    for N in N_values
        for kmax in kmax_values
            reg_coeffs = [Float64(i) / (N - i) for i in 1:kmax]
            iter = rand(1:N)

            # Setup: random u_v_ratio matrices (old and new)
            uv_old = randn(ComplexF64, N - 1, N)
            uv_new = copy(uv_old)
            # Simulate moving particle `iter`: change iter's entries in all columns
            for j in 1:N
                if j != iter
                    row = iter < j ? iter : iter - 1
                    uv_new[row, j] = randn(ComplexF64)
                else
                    uv_new[:, j] .= randn(ComplexF64, N - 1)
                end
            end

            # ESP matrix for full recompute
            esp_full = zeros(ComplexF64, kmax + 1, N)
            get_symmetric_polynomials_full!(view(esp_full, :, 1), view(uv_old, :, 1), kmax, reg_coeffs)
            for j in 1:N
                get_symmetric_polynomials_full!(view(esp_full, :, j), view(uv_old, :, j), kmax, reg_coeffs)
            end

            # ESP matrix for incremental (start from old state)
            esp_inc = copy(esp_full)

            # Benchmark full recompute
            esp_bench_full = copy(esp_full)
            t_full = @belapsed bench_full_recompute($esp_bench_full, $uv_new, $N, $kmax, $reg_coeffs)

            # Benchmark incremental
            esp_bench_inc = copy(esp_full)
            t_inc = @belapsed bench_incremental($esp_bench_inc, $uv_old, $uv_new, $iter, $N, $kmax, $reg_coeffs)

            # Verify they agree
            bench_full_recompute(esp_full, uv_new, N, kmax, reg_coeffs)
            bench_incremental(esp_inc, uv_old, uv_new, iter, N, kmax, reg_coeffs)
            err = maximum(abs.(esp_full .- esp_inc))

            speedup = t_full / t_inc
            check = err < 1e-10 ? "" : "  ⚠ err=$(err)"
            @printf("%5d  %5d  %12.2f  %12.2f  %7.1fx%s\n",
                    N, kmax, t_full * 1e6, t_inc * 1e6, speedup, check)
        end
        println("-" ^ 50)
    end
end

## ─── Run ─────────────────────────────────────────────────────────────────────

test_correctness()
run_benchmarks()
