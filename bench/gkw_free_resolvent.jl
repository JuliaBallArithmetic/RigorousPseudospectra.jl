# Free resolvent bound on the K=256 Gauss–Kuzmin–Wirsing discretization, Float64 vs
# BigFloat: how many GKW eigenvalues the small-gain condition closes "for free" (one
# miyajima_rump enclosure, O(#clusters) per z, NO SVD anywhere), and the zoom-near-0
# picture of the grey (free-bound-undefined) region.
#
# Bridge (GKWExperiments Thm 2.3): on a circle Γⱼ encircling only λⱼ,
#     αⱼ = ε_K · sup_{z∈Γⱼ} ‖R_{A_K}(z)‖ < 1  ⇒  L₁ has exactly one eigenvalue inside Γⱼ.
# Float64 grey radius β₂+‖E‖ ~ κ(W)·u ~ 1e-2 (a conditioning artifact) swallows the small
# eigenvalues; BigFloat collapses it (~κ(W)·2^{-p}) and they become certifiable.
#
# Heavy run (computes both, caches, plots):   julia --project=bench bench/gkw_free_resolvent.jl [h]
# Cheap re-zoom from cache:                    julia --project=bench bench/gkw_free_resolvent.jl replot <h>

using RigorousPseudospectra
using LinearAlgebra
using GenericSchur          # BigFloat eigen for the candidate (resolves close eigenvalues)
using BallArithmetic
using Serialization
using CairoMakie
using Printf

const DATA = "/home/isaia/Code/GKWExperiments.jl/data"
const CACHE = joinpath(@__DIR__, "figures", "gkw_free_cache.jls")

eps_K(::Type{T}; K = 256, N = 5000) where {T} =
    T(sum(sqrt(2n + 1) / (n - 0.5)^2 for n in 1:(N - 1))) * (T(2) / 3)^(K + 1)

function compute(prec, refine)
    T = prec == 0 ? Float64 : BigFloat
    label = prec == 0 ? "Float64 (refine=$refine)" : "BigFloat-$prec (refine=$refine)"
    if prec == 0
        A = ComplexF64.(BallArithmetic.mid(deserialize(joinpath(DATA, "ball_matrix_K256.jls"))))
    else
        setprecision(BigFloat, prec)
        A = Complex{BigFloat}.(BallArithmetic.mid(deserialize(joinpath(DATA, "ball_matrix_bf_K256.jls"))))
    end
    n = size(A, 1)
    ε = eps_K(T)
    @info "$label: miyajima_rump + free_resolvent on $(n)×$(n) (cheap κ₂, no SVD) …"
    t = @elapsed begin
        R = miyajima_rump(A; refine = refine, mode = :entrywise)
        fr = free_resolvent(A, R)
    end
    λs = R.centres[sortperm(R.centres; by = abs, rev = true)]
    dnear(j) = minimum(abs(λs[j] - λs[k]) for k in eachindex(λs) if k != j)
    nsamp = 96
    free = falses(length(λs))
    for j in eachindex(λs)
        rj = dnear(j) / 3
        rj <= 0 && continue
        M = zero(T); ok = true
        for q in 0:(nsamp - 1)
            rb = resolvent_bound(fr, λs[j] + rj * cis(2 * T(π) * q / nsamp))
            isfinite(rb) || (ok = false; break)
            M = max(M, rb)
        end
        ok && ε * M < 1 && (free[j] = true)
    end
    @printf("[%s] κ₂=%.3e  max row-tail=%.3e  max col-tail=%.3e  → %d/%d FOR FREE  (%.0fs)\n",
        label, Float64(fr.kappa), Float64(maximum(fr.rowsum)), Float64(maximum(fr.colsum)),
        count(free), length(λs), t)
    return (; label, bits = prec, fr, λs, free)
end

function panel!(fig, pos, res, h; Nx = 340)
    res.bits == 0 || setprecision(BigFloat, res.bits)
    T = res.bits == 0 ? Float64 : BigFloat
    fr, λs, free = res.fr, res.λs, res.free
    dnear(j) = minimum(abs(λs[j] - λs[k]) for k in eachindex(λs) if k != j)
    xs = range(-h, h; length = Nx)
    ys = range(-h, h; length = Nx)
    Z = [(s = sigma_min_lower(fr, T(x) + im * T(y)); s > 0 ? Float64(-log10(s)) : NaN)
         for x in xs, y in ys]
    ax = Axis(fig[1, pos]; title = "$(res.label)  (grey = free bound undefined)",
        xlabel = "Re z", ylabel = "Im z", aspect = DataAspect())
    fin = filter(isfinite, Z)
    hm = heatmap!(ax, xs, ys, Z; colormap = :viridis, nan_color = RGBAf(0.55, 0.55, 0.55, 1),
        colorrange = isempty(fin) ? (0, 1) : (minimum(fin), maximum(fin)))
    contour!(ax, xs, ys, Z; levels = 12, color = (:white, 0.4), linewidth = 0.5)
    Colorbar(fig[2, pos], hm; vertical = false, label = "log₁₀ ‖(A−zI)⁻¹‖ bound")
    θ = range(0, 2π; length = 100)
    for j in eachindex(λs)
        x0, y0 = Float64(real(λs[j])), Float64(imag(λs[j]))
        (-h <= x0 <= h && -h <= y0 <= h) || continue
        rj = Float64(dnear(j) / 3)
        rj > 0 && lines!(ax, x0 .+ rj .* cos.(θ), y0 .+ rj .* sin.(θ);
            color = free[j] ? :lime : :red, linewidth = 1.3)
        scatter!(ax, [x0], [y0]; marker = :xcross, color = :red, markersize = 7)
    end
    limits!(ax, -h, h, -h, h)
end

function render(r64, rbf, h)
    fig = Figure(; size = (1180, 560))
    panel!(fig, 1, r64, h)
    panel!(fig, 2, rbf, h)
    mkpath(joinpath(@__DIR__, "figures"))
    out = joinpath(@__DIR__, "figures", "gkw_free_resolvent_zoom0.png")
    save(out, fig)
    println("Figure (h=$h) → ", out)
end

function main()
    if !isempty(ARGS) && ARGS[1] == "replot"
        c = deserialize(CACHE)
        render(c.r64, c.rbf, parse(Float64, ARGS[2]))
    else
        prec = isempty(ARGS) ? 256 : parse(Int, ARGS[1])   # BigFloat precision bits
        r64 = compute(0, 0)          # Float64 baseline (refine=0)
        rbf = compute(prec, 6)       # BigFloat: GenericSchur eigen candidate + refine
        serialize(CACHE, (; r64, rbf))
        render(r64, rbf, 0.02)
    end
end

main()
