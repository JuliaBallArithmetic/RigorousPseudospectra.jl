# Floating point, no certificate. For one (k, ξ) and one eigenvalue count, build frames of growing
# size from ONE Golub-Kahan-Lanczos run and, for each, scan R over the window between the neig-th
# and (neig+1)-th eigenvalue class (any R there leaves exactly those classes outside), reporting
# the margin of the depth rule
#     γ1γ2 < σ11(R)(R − ĝ).
# Usage: julia -t 32 --project=. explore2.jl KE XI NEIG ks1,ks2,...

include("bzcommon.jl")

ke   = parse(Int, ARGS[1])
ξ    = parse(Float64, ARGS[2])
neig = parse(Int, ARGS[3])
kss  = parse.(Int, split(ARGS[4], ','))
nR   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 9
nθ   = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 8

k = 2^ke
L, binfo = build_L(k)
l = noise_halfwidth(ξ, k)
Amul, Aadj, η, oi = make_operator(L, l)
op = RP.LinOp(Amul, Aadj, k; epsA = 0.0, etaA = η)

function classes(λ; tol = 1e-8)
    a = sort(abs.(λ); rev = true)
    out = Float64[]
    for x in a
        (isempty(out) || abs(x - out[end]) > tol * max(1.0, x)) && push!(out, x)
    end
    return out
end

λ = RP._ritz_leading(Amul, k, Float64; L = max(150, 8neig))
cl = classes(λ)
lo, hi = cl[neig + 1], cl[neig]
R0 = sqrt(lo * hi)
@printf("n = %d, ξ = %.0e, l = %d, η = %.3e, neig = %d, window R ∈ (%.6f, %.6f), ‖A‖≈%.4f\n",
    k, ξ, l, η, neig, lo, hi, 0.0)

t0 = time()
Wλ = first(RP.eigendirections(Amul, k, R0; L = max(150, 8neig), T = Float64))
Lsv = min(k, 2 * maximum(kss) + 80)
Vr, Ul, σ = RP.gkl(Amul, Aadj, k, Lsv; T = Float64)
@printf("eigendirections %d, GKL %d steps, %.0f s; σ1 = %.4f\n",
    size(Wλ, 2), Lsv, time() - t0, σ[1])
flush(stdout)

for ks_ in kss
    t0 = time()
    Q = RP.rrqr_frame(hcat(Wλ, Vr[:, 1:min(ks_, size(Vr, 2))], Ul[:, 1:min(ks_, size(Ul, 2))]))
    tf = time() - t0
    X = RP.apply_cols(op, Q, false); Xt = RP.apply_cols(op, Q, true)
    C = X - Q * (Q' * X); D = Xt - Q * (Q' * Xt)
    γ2 = opnorm(C); γ1 = opnorm(D)
    g = minimum(RP.estimate_orders(op, Q, 8; nsamples = 6))
    G = Q' * X; M = Q' * Q
    best = (; R = NaN, σ11 = 0.0, margin = 0.0)
    for R in range(lo * 1.002, hi * 0.998; length = nR)
        s = minimum(minimum(svdvals(G - R * cis(2π * t / nθ) * M)) for t in 0:(nθ - 1))
        m = s * max(R - g, 0.0) / (γ1 * γ2)
        m > best.margin && (best = (; R, σ11 = s, margin = m))
    end
    @printf("ks=%4d dimQ=%5d  ĝ=%.4e γ1=%.4e γ2=%.4e | bestR=%.6f σ11=%.4e margin=%.4f %s (%.0fs)\n",
        ks_, size(Q, 2), g, γ1, γ2, best.R, best.σ11, best.margin,
        best.margin > 1 ? "RULE PASSES" : "", time() - t0)
    flush(stdout)
end
@printf("σ decay: ")
for i in (1, 8, 16, 32, 64, 128, 200, 300, 400); i <= length(σ) && @printf("σ%d=%.3e ", i, σ[i]); end
println()
@printf("classes: "); for i in 1:min(14, length(cl)); @printf("%.5f ", cl[i]); end; println()
