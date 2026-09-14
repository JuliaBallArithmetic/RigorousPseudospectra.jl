# Singular-value decay and the certified far block. Proposition prop:frame bounds the far block of
# a frame containing the leading k right and left singular vectors by s_{k+1}(A), and no deflation
# of dimension k' brings it below s_{2k'+1}(A); so the far block that the certificate can reach is
# set by the decay of the singular values. For five matrices of size 120 we plot s_k/s_1, and
# print, for a two-sided frame with ks = 10 singular directions per side, the certified ḡ ≥ ‖G22‖
# (the sweep of sec:fpsweep) against s_11/s_1.
#   julia --project=bench bench/regime_figure.jl
using RigorousPseudospectra, CairoMakie, LinearAlgebra, Printf
include(joinpath(@__DIR__, "matrices.jl"))

convdiff(n; ε = 0.02) = (h = 1 / (n + 1); A = zeros(n, n);
    for i in 1:n
        A[i, i] = 2ε / h^2
        i < n && (A[i, i + 1] = -ε / h^2 + 1 / (2h))
        i > 1 && (A[i, i - 1] = -ε / h^2 - 1 / (2h))
    end; A)
cyclicshift(n) = (A = zeros(n, n);
    for i in 1:n
        A[mod(i, n) + 1, i] = 1
    end; A)
function smoothing(n; w = 0.05)
    xs = range(0, 1; length = n)
    A = [exp(-((xs[i] - xs[j]) / w)^2) for i in 1:n, j in 1:n]
    A ./= sum(A; dims = 2)
    return A
end

cases = [
    ("Grcar", grcar(120)),
    ("Convection–diffusion", convdiff(120)),
    ("Cyclic shift", cyclicshift(120)),
    ("Graded chain", Matrix(graded_chain(120))),
    ("Smoothing kernel", smoothing(120)),
]

palette = [:firebrick, :orange, :gray50, :seagreen, :royalblue]
fig = Figure(size = (720, 520))
ax = Axis(fig[1, 1]; xlabel = "index k", ylabel = "s_k / s_1", yscale = log10,
    title = "Singular-value decay")
@printf("%-22s %10s %10s %10s %6s\n", "matrix", "s_11/s_1", "ḡ/s_1", "‖G22‖/s_1", "dim Q")
for (i, (name, A)) in enumerate(cases)
    Ar = Matrix{ComplexF64}(A)
    s = svdvals(Ar)
    lines!(ax, collect(1:length(s)), max.(s ./ s[1], 1e-16); linewidth = 2.4, color = palette[i],
        label = name)
    n = size(Ar, 1)
    Q, _ = two_sided_frame(x -> Ar * x, x -> Ar' * x, n, 0.0; ks = 10, keig = false,
        T = ComplexF64)
    sc = certify_split(Ar, Q; compression = false)
    S = sqrt(Hermitian(Q' * Q))
    Z = Q / S
    P = I - Z * Z'
    @printf("%-22s %10.3e %10.3e %10.3e %6d\n", name, s[11] / s[1], sc.gbar / s[1],
        opnorm(P * Ar * P) / s[1], sc.k)
end
ylims!(ax, 1e-8, 2.0)
axislegend(ax; position = :lb, labelsize = 9)
mkpath(joinpath(@__DIR__, "figures"))
save(joinpath(@__DIR__, "figures", "regime_decay.png"), fig)
println("saved bench/figures/regime_decay.png")
