# The classical non-normal gallery, tested against the MIYAJIMA–RUMP ENCLOSURE (paper §2–§3).
#
# These are SMALL DENSE matrices, so the dense enclosure is what one would actually run on them —
# the Krylov reduction of §5 has nothing to do here (and applying it is a category error: the
# singular deflation has nothing to remove, and for n small the captured frame can swallow the whole
# space, leaving no far block at all). What we probe is whether the enclosure behaves correctly on
# defective, near-defective and strongly non-normal spectra:
#
#   * CONTAINMENT — every true eigenvalue must lie in some certified region (the primary rigor test);
#   * SEPARATION — eigenvalues that CAN be separated should come out as singletons with tight radii,
#     and eigenvalues that cannot (defective, or an eigenbasis past 1/u) must come out as ONE cluster
#     whose region tracks |z−λ|^m rather than being split into m discs;
#   * the Neumann test ‖R₂‖ < 1 must REFUSE rather than lie when no usable basis exists.
#
# The enclosure comes from BallArithmetic (the verified-linalg kernel); this package layers the
# pseudospectral method on top of it. `schur_newton_vbd` races several candidate bases and keeps the
# best certified one, so it picks the eigenvector frame where that separates (Kahan, convection–
# diffusion) and the unitary Schur frame where the eigenbasis is hopeless (Grcar, Jordan, Demmel).
#
#   julia --project=. bench/nonnormal_gallery.jl
using RigorousPseudospectra, BallArithmetic, LinearAlgebra, Printf, Random
Random.seed!(1)

grcar(n; k=3)   = (A=zeros(n,n); for i in 1:n; A[i,i]=1; i<n && (A[i+1,i]=-1); for j in 1:k; i+j<=n && (A[i,i+j]=1); end; end; ComplexF64.(A))
function kahan(n; θ=1.2)
    s=sin(θ); c=cos(θ); S=Diagonal([s^(i-1) for i in 1:n]); K=Matrix(1.0I,n,n)
    for i in 1:n, j in i+1:n; K[i,j]=-c; end; ComplexF64.(S*K)
end
jordan(n; λ=0.5) = (A=λ*Matrix(1.0I,n,n); for i in 1:n-1; A[i,i+1]=1; end; ComplexF64.(A))
function convdiff(n; ε=0.02)   # -ε u'' + u' on [0,1], Dirichlet, central differences
    h=1/(n+1); A=zeros(n,n)
    for i in 1:n; A[i,i]=2ε/h^2; i<n && (A[i,i+1]=-ε/h^2+1/(2h)); i>1 && (A[i,i-1]=-ε/h^2-1/(2h)); end
    ComplexF64.(A)
end
demmel(n; β=1e4) = (A=zeros(n,n); for i in 1:n; A[i,i]=-1.0; for j in i+1:n; A[i,j]=β^(j-i); end; end; ComplexF64.(A))

function run_case(name, A; maxsteps=6, note="")
    n = size(A,1); ev = eigvals(A)
    condV = cond(eigen(A).vectors)
    @printf("%-22s n=%3d  ‖A‖₂=%.2e  cond(V)=%.2e%s\n", name, n, opnorm(Matrix(A),2), condV, note)
    local r, t
    try
        t = @elapsed r = BallArithmetic.schur_newton_vbd(BallMatrix(Matrix(A));
                                                         maxsteps=maxsteps, kappa_mode=:svdbox)
    catch e
        @printf("      REFUSED — %s\n\n", first(replace(sprint(showerror,e), "\n"=>" "), 84)); return nothing
    end
    if !(r.nrmR2 < 1)
        @printf("      REFUSED — ‖R₂‖=%.3e ≥ 1 (Neumann); a refusal, not an unverified answer\n\n", r.nrmR2)
        return r
    end
    cls  = [collect(c) for c in r.clusters]
    rad  = [BallArithmetic.rad(d) for d in r.cluster_intervals]
    radv = [rad[findfirst(c -> i in c, cls)] for i in 1:n]
    ins  = count(λ -> minimum(abs(λ - r.eigenvalues[i]) - radv[i] for i in 1:n) <= 0, ev)
    @printf("      clusters=%3d (largest %3d)  κ₂=%.2e  β=%.2e  ‖R₂‖=%.1e  radii ≤ %.2e\n",
            length(cls), maximum(length.(cls)), r.kappa, r.beta, r.nrmR2, maximum(radv))
    @printf("      CONTAINMENT %d/%d %s   (%.2fs)\n\n", ins, n, ins == n ? "✓" : "✗ FAIL", t)
    return r
end

println("Miyajima–Rump enclosure on the classical dense non-normal gallery\n" * "="^78)
run_case("Grcar",                grcar(100))
run_case("Kahan",                kahan(60))
run_case("Jordan block λ=0.5",   jordan(20);  note="   [defective: expect ONE cluster of n]")
run_case("Convection–diffusion", convdiff(80))
run_case("Demmel",               demmel(10);  note="   [Demmel 1987 counterexample]")
println("="^78)
println("A defective block must appear as a SINGLE cluster: its region {z : σ_min(P−zI) ≤ β}")
println("tracks |z−λ|^m, which is the whole reason for using σ_min of a block over an")
println("eigenvalue distance. Splitting it into m discs would be the failure mode.")
