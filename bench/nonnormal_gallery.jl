# Non-normal test-matrix gallery for the rigorous Krylov method (paper §6.2).
# Constructs the standard corner cases and runs certify_leading_eigs_sing on each (dense, via
# x->A*x / x->A'*x). Reports: true leading eigenvalues (Float64 reference), the rigorous σ_min
# floor + rigorous enclosure per certified eigenvalue, timing, and — for the cyclic shift — the
# raw Arnoldi Ritz values (to exhibit symptom S1: Ritz collapses to the origin).
#   julia --project=. bench/nonnormal_gallery.jl
using RigorousPseudospectra, LinearAlgebra, Printf, Random
Random.seed!(1)

grcar(n; k=3)   = (A=zeros(n,n); for i in 1:n; A[i,i]=1; i<n && (A[i+1,i]=-1); for j in 1:k; i+j<=n && (A[i,i+j]=1); end; end; A)
function kahan(n; θ=1.2)
    s=sin(θ); c=cos(θ); S=Diagonal([s^(i-1) for i in 1:n]); K=Matrix(1.0I,n,n)
    for i in 1:n, j in i+1:n; K[i,j]=-c; end; S*K
end
jordan(n; λ=0.5) = (A=λ*Matrix(1.0I,n,n); for i in 1:n-1; A[i,i+1]=1; end; A)
cyclicshift(n)  = (A=zeros(n,n); for i in 1:n; A[mod(i,n)+1,i]=1; end; A)
function convdiff(n; ε=0.02)   # -ε u'' + u' on [0,1], Dirichlet, central differences
    h=1/(n+1); A=zeros(n,n)
    for i in 1:n; A[i,i]=2ε/h^2; i<n && (A[i,i+1]=-ε/h^2+1/(2h)); i>1 && (A[i,i-1]=-ε/h^2-1/(2h)); end; A
end

function run_case(name, A; neig=4, Rcirc=nothing, showritz=false)
    n=size(A,1); Ac=ComplexF64.(A)
    ev=eigvals(Ac); ord=sortperm(abs.(ev);rev=true)
    println("\n","="^72); @printf("%s  (n=%d, ‖A‖₂=%.3f)\n", name, n, opnorm(A,2)); println("="^72)
    @printf("  true leading |λ|: %s\n", join((@sprintf("%.4f",abs(ev[ord[k]])) for k in 1:min(6,n)), " "))
    if showritz
        H,V,βN = RigorousPseudospectra._arnoldi(x->Ac*x, n, min(40,n-2))
        rz=eigvals(ComplexF64.(H))
        @printf("  Arnoldi(%d) Ritz |λ| (S1 check): %s\n", size(H,1),
            join((@sprintf("%.3f",abs(r)) for r in sort(abs.(rz);rev=true)[1:min(6,length(rz))])," "))
    end
    local out, t
    try
        t=@elapsed out = Rcirc===nothing ?
            certify_leading_eigs_sing(Ac; neig=neig, nblock_eig=min(40,n-2), margin=0.8) :
            certify_leading_eigs_sing(Ac; Rcirc=Rcirc, nblock_eig=min(40,n-2), margin=0.8)
    catch e
        @printf("  certify_leading_eigs_sing REFUSED: %s\n", sprint(showerror, e) |> x->first(x,80))
        return nothing
    end
    @printf("  certify_leading_eigs_sing: %.2fs  nsing=%d dim(Q)=%d ‖G₂₂‖=%.4f κ₂=%.1e\n",
        t, out.nsing, out.dimQ, out.G22norm, out.mr.kappa)
    for r in out.results
        @printf("    λ=%+.4f%+.4fi  σ_min≥%.3e %s  enclosure=%.2e\n",
            real(r.λ),imag(r.λ), r.cert.floor, r.cert.certified ? "✓" : "✗", r.cert.enclosure)
    end
    out
end

run_case("Grcar",              grcar(100);        neig=4)
run_case("Kahan",              kahan(60);         neig=4)
run_case("Jordan block λ=0.5", jordan(30);        neig=2)
run_case("Convection–diffusion", convdiff(100);   neig=4)
run_case("Cyclic shift",       cyclicshift(48);   neig=4, showritz=true)
