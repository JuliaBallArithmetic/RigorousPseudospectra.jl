# The regime figure (paper §6.2): the method certifies exactly when the singular values of A DECAY
# (a low-dimensional resolvent tail — the transfer-operator regime) and fails safely when they are
# FLAT. We plot the normalized singular-value decay σ_k/σ_1 for the gallery matrices; the decaying
# ones are certifiable, the flat ones are the correct negatives.
#   julia --project=bench bench/regime_figure.jl
using RigorousPseudospectra, CairoMakie, LinearAlgebra, Printf
include(joinpath(@__DIR__, "matrices.jl"))

# flat / full-rank (method fails safely)
convdiff(n; ε=0.02) = (h=1/(n+1); A=zeros(n,n); for i in 1:n; A[i,i]=2ε/h^2; i<n && (A[i,i+1]=-ε/h^2+1/(2h)); i>1 && (A[i,i-1]=-ε/h^2-1/(2h)); end; A)
cyclicshift(n)      = (A=zeros(n,n); for i in 1:n; A[mod(i,n)+1,i]=1; end; A)
# decaying (method certifies) — a Gaussian-smoothing convolution, a transfer-operator proxy
function smoothing(n; w=0.05)
    xs=range(0,1;length=n); A=[exp(-((xs[i]-xs[j])/w)^2) for i in 1:n, j in 1:n]; A./=sum(A;dims=2); A
end

cases = [
    ("Grcar (flat)",              grcar(120),        :fail),
    ("Convection–diffusion (flat)", convdiff(120),   :fail),
    ("Cyclic shift (flat, σ≡1)",  cyclicshift(120),  :fail),
    ("Graded chain (decays)",     Matrix(graded_chain(120)), :ok),
    ("Smoothing / transfer-like (decays)", smoothing(120), :ok),
]

palette = [:firebrick, :orange, :gray50, :seagreen, :royalblue]
fig = Figure(size=(720,520))
ax = Axis(fig[1,1]; xlabel="index k", ylabel="σ_k / σ_1", yscale=log10,
    title="Singular-value decay: decaying ⇒ certifies, flat ⇒ fails safely")
for (i,(name,A,tag)) in enumerate(cases)
    s = svdvals(Matrix{Float64}(real.(A))); s = max.(s ./ s[1], 1e-16)
    lines!(ax, collect(1:length(s)), s; linewidth=2.4, color=palette[i],
        linestyle = tag==:ok ? :solid : :dash, label=name)
    @printf("%-38s σ_60/σ_1 = %.2e  (%s)\n", name, s[min(60,end)], tag==:ok ? "certifies" : "fails safely")
end
ylims!(ax, 1e-8, 2.0)
axislegend(ax; position=:lb, labelsize=9)
save(joinpath(@__DIR__, "figures", "regime_decay.png"), fig)
println("saved bench/figures/regime_decay.png")
