# Compare the two §4e refinement corrections (draft Remark `rem:block`):
#   entrywise sweep   Xᵢⱼ = −Ãᵢⱼ/(dᵢ−dⱼ)              (scalar reciprocal gaps)
#   block-Sylvester   M_kk X − X M_ll = −M_kl          (full diagonal blocks)
# on matrices with working clusters of increasing internal coupling. They COINCIDE on
# fully-separated spectra and at the κ·eps fixed point; any difference lives in the
# transient on near-defective / tightly-coupled clusters, where M_kk is not ≈ diagonal.
#
# It also evaluates two stall-recovery policies (draft Remark `rem:exp`): the two
# single corrections tie with partly-disjoint stalls, so (switch) run one to a stall
# and retry with the other, or (perturb) apply a ~10% random kick on a stall — both
# recover the disjoint failures; bases that stall for BOTH are first-order-obstructed.
#
# Run:  julia --project=. bench/refine_variants.jl

using RigorousPseudospectra
using LinearAlgebra
using Random
using Printf

# off-diagonal residue of W⁻¹BW (Frobenius) — what refinement shrinks
offres(B, W) = (A = ComplexF64.(W) \ (ComplexF64.(B) * ComplexF64.(W));
sqrt(sum(abs2, A - Diagonal(diag(A)))))

# similarity-embed a list of diagonal blocks in a well-conditioned S
function embed(blocks; cond_scale = 0.3, seed = 0)
    rng = MersenneTwister(seed)
    M = cat(blocks...; dims = (1, 2))
    n = size(M, 1)
    S = I + cond_scale * randn(rng, ComplexF64, n, n)
    return ComplexF64.(S * M / S)
end

jb(m, λ) = (J = Matrix{ComplexF64}(λ * I, m, m); for i in 1:(m-1); J[i, i+1] = 1.0; end; J)

# ---- test gallery: name => (B, rough-basis seed) -------------------------------------
function gallery()
    cases = Pair{String, Matrix{ComplexF64}}[]
    # a near-defective 3×3 cluster (perturbed Jordan) + separated eigenvalues
    rng = MersenneTwister(1)
    J = jb(3, 1.0) .+ 1e-4 .* randn(rng, ComplexF64, 3, 3)
    push!(cases, "jordan3(1e-4) + {5,9,13}" => embed([J, fill(5.0+0im,1,1), fill(9.0+0im,1,1), fill(13.0+0im,1,1)]; seed = 2))
    # two coupled 2×2 near-defective clusters
    rng = MersenneTwister(3)
    J1 = jb(2, 0.0) .+ 1e-3 .* randn(rng, ComplexF64, 2, 2)
    J2 = jb(2, 6.0) .+ 1e-3 .* randn(rng, ComplexF64, 2, 2)
    push!(cases, "2×jordan2(1e-3)"           => embed([J1, J2, fill(12.0+0im,1,1)]; seed = 4))
    # a tighter Jordan (harder)
    rng = MersenneTwister(5)
    J = jb(4, 2.0) .+ 1e-6 .* randn(rng, ComplexF64, 4, 4)
    push!(cases, "jordan4(1e-6) + {8,14}"     => embed([J, fill(8.0+0im,1,1), fill(14.0+0im,1,1)]; seed = 6))
    # plain random (all singletons — control, should tie exactly)
    push!(cases, "random (separated)"         => embed([randn(MersenneTwister(7), ComplexF64, 8, 8)]; seed = 7))
    return cases
end

function compare(B; θ = 0.9, maxsteps = 12, perturb = 0.05, seed = 99)
    n = size(B, 1)
    rng = MersenneTwister(seed)
    W0 = eigen(B).vectors * (I + perturb * randn(rng, ComplexF64, n, n))
    traj = Dict(:entrywise => Float64[offres(B, W0)], :block => Float64[offres(B, W0)])
    W = Dict(:entrywise => copy(W0), :block => copy(W0))
    for v in (:entrywise, :block)
        for _ in 1:maxsteps
            W[v] = refine_basis(B, W[v]; steps = 1, θ = θ, mode = v)
            push!(traj[v], offres(B, W[v]))
        end
    end
    # certify the refined bases
    Re = miyajima_rump(B, W[:entrywise])
    Rb = miyajima_rump(B, W[:block])
    return traj, Re, Rb
end

# does a refinement policy converge (offdiag ‖·‖_F < tol within `maxsteps`)?
function converges(B, W0, policy; tol = 1e-8, maxsteps = 20, kick = 0.1,
        rounds = 6, seed = 0)
    n = size(B, 1)
    if policy in (:entrywise, :block, :auto)         # the three wired-in modes
        W = refine_basis(B, W0; steps = maxsteps, θ = 0.9, mode = policy, tol = tol)
        return offres(B, W) < tol
    elseif policy === :perturb                       # entrywise; on stall, kick & retry
        rng = MersenneTwister(seed)
        W = copy(W0)
        for _ in 0:rounds
            W = refine_basis(B, W; steps = maxsteps, θ = 0.9, mode = :entrywise)
            offres(B, W) < tol && return true
            W = W * (I + kick * randn(rng, ComplexF64, n, n))
        end
        return false
    end
end

# robustness sweep: over many rough-basis seeds, how often does each policy converge?
function robustness(B; seeds = 1:40, perturb = 0.05)
    n = size(B, 1)
    policies = (:entrywise, :block, :auto, :perturb)
    conv = Dict(p => 0 for p in policies)
    for s in seeds
        rng = MersenneTwister(1000 + s)
        W0 = eigen(B).vectors * (I + perturb * randn(rng, ComplexF64, n, n))
        for p in policies
            conv[p] += Int(converges(B, W0, p; seed = 7000 + s))
        end
    end
    return conv, length(seeds)
end

function main()
    @printf("%-26s | %-7s | %-8s | %-6s | %-8s | %-8s\n",
        "matrix", "κ₂", "entrywise", "block", "auto", "perturb")
    println("-"^78)
    for (name, B) in gallery()
        _, Re, _ = compare(B)
        conv, ns = robustness(B)
        @printf("%-26s | %.1e | %4d/%-3d | %3d/%-2d | %4d/%-3d | %4d/%-3d\n",
            name, Re.kappa, conv[:entrywise], ns, conv[:block], ns,
            conv[:auto], ns, conv[:perturb], ns)
    end
    println("\n(converged rough-basis seeds (offdiag ‖·‖_F < 1e-8) out of N, per policy:")
    println("   entrywise / block — the two single corrections (Remark rem:block);")
    println("   auto    — run entrywise; on a stall retry block from W₀ (the wired-in")
    println("             mode = :auto, parameter-free);")
    println("   perturb — entrywise; on a stall apply a ~10% random kick and retry.")
    println(" The two single variants tie with partly-disjoint stalls; auto and a tuned")
    println(" perturb both recover the disjoint failures. Bases that stall for both are")
    println(" first-order-obstructed (need extended precision), not fixable by either.)")
end

main()
