# Uncertified helpers. Nothing in this file is certified and nothing certified calls it: it reports
# floating-point estimates of the quantities a run of the certificate will compute, so that the
# radius, the frame size and the order of the sweep can be chosen before the run. The choice is the
# caller's; these functions report and do not decide.

# the operator as two functions on vectors, for the floating-point Krylov processes of frame.jl
_vecmul(op::LinOp, adj::Bool) = x -> vec(apply_cols(op, reshape(x, :, 1), adj))

"""
    helper_gaps(op::LinOp; L = 200) -> Vector{NamedTuple}

Uncertified. The gaps between consecutive distinct moduli of the leading Ritz values of an
Arnoldi run of length `L`, as candidate radii: for each `N` with `|θ_N| > |θ_{N+1}|`, the radius
`R = √(|θ_N| |θ_{N+1}|)` with `N` Ritz values outside it. Ritz values approximate the outer
spectrum only, so the list is a starting point for [`helper_diagnose`](@ref), not a statement about
the eigenvalues of `op`.
"""
function helper_gaps(op::LinOp; L::Integer = 200)
    θ = _ritz_leading(_vecmul(op, false), op.n, op.T; L)
    a = abs.(θ)
    return [(; N, R = sqrt(a[N] * a[N + 1]), outer = a[N], inner = a[N + 1])
            for N in 1:(length(a) - 1) if a[N] > a[N + 1] * (1 + 1e-9)]
end

"""
    helper_diagnose(op::LinOp, R; ks = nothing, Leig = 200, Lsv = 200, mmax = 16, nθ = 32,
                    nsamples = 8, method = :qr, verbose = true) -> NamedTuple

Uncertified. Floating-point estimates, at the radius `R`, of what [`certify_split`](@ref) and the
cover of the circle `|z| = R` will compute:

* `nout`, the number of Ritz values of modulus greater than `R` (an Arnoldi run of length `Leig`);
* `K = #{σ_i ≥ R}` and `dmin`, the least `d` with `σ_{2d+1} < R`, from the singular values of a
  Golub-Kahan-Lanczos run of length `Lsv` (`K_reached = true` when all of them are `≥ R`, so that
  `K` and `dmin` are only lower bounds); by Proposition prop:frame no frame of dimension below
  `dmin` gives `‖G22‖ < R`;
* for the frame of def:Q with `ks` singular vectors per side (default `K`): `dimQ`, the
  predictions `gbar_est[t] = v_t (n − k)^{1/2t}` of the bound the sweep of order `t ≤ mmax` would
  certify, `v_t` the floating-point estimate of [`estimate_orders`](@ref), and the order `m` that
  [`certify_far_block`](@ref) would choose for the target `R` from the same predictions. The factor
  `(n − k)^{1/2t}` is exact for a far block with `n − k` equal singular values and overestimates
  when they decay, by a factor 9 at order 1 on the test matrix of `test/runtests.jl`;
* `γ1`, `γ2` and the least `σ11 = σ_min(G11 − zI)` over `nθ` points of the circle, formed in floating
  point on that frame, and `margin = σ11 (R − gbar_est[m]) − γ1γ2`, whose sign predicts whether the
  hypothesis of thm:assembly holds at the samples.

With `verbose = true` the numbers are also printed.
"""
function helper_diagnose(op::LinOp, R::Real; ks = nothing, Leig::Integer = 200,
        Lsv::Integer = 200, mmax::Integer = 16, nθ::Integer = 32, nsamples::Integer = 8,
        method::Symbol = :qr, verbose::Bool = true)
    n = op.n
    T = op.T
    R = Float64(R)
    Amul, Aadj = _vecmul(op, false), _vecmul(op, true)
    θ = _ritz_leading(Amul, n, T; L = Leig)
    nout = count(x -> abs(x) > R, θ)
    _, _, σ = gkl(Amul, Aadj, n, min(n, Lsv); T)
    K = count(>=(R), σ)
    K_reached = K == length(σ)
    j = findfirst(<(R), σ)
    dmin = j === nothing ? cld(length(σ), 2) : max(cld(j - 1, 2), 0)
    ksuse = ks === nothing ? K : Int(ks)
    Q, _ = two_sided_frame(Amul, Aadj, n, R; ks = ksuse, Leig, Lsv = min(n, 2max(ksuse, 1) + 60),
        T, method)
    k = size(Q, 2)
    est = estimate_orders(op, Q, mmax; nsamples)
    gbar_est = _predicted_gbar(est, n, k)
    m = _choose_order(gbar_est, R, 0.05)
    X = apply_cols(op, Q, false)
    Xt = apply_cols(op, Q, true)
    G = Q' * X
    γ2 = k == 0 ? 0.0 : opnorm(X - Q * G)
    γ1 = k == 0 ? 0.0 : opnorm(Xt - Q * (Q' * Xt))
    σ11 = k == 0 ? Inf : minimum(svdvals(G - R * cis(2π * t / nθ) * I)[end] for t in 0:(nθ - 1))
    margin = σ11 * (R - gbar_est[m]) - γ1 * γ2
    d = (; R, nout, K, K_reached, dmin, ks = ksuse, dimQ = k, gbar_est, m, γ1, γ2, σ11, margin)
    verbose && _helper_print(d)
    return d
end

function _helper_print(d)
    println("helper_diagnose (uncertified, floating point) at R = ", d.R)
    println("  Ritz values outside R: ", d.nout)
    println("  K = #{σ_i ≥ R}: ", d.K, d.K_reached ? " (at least: every computed σ is ≥ R)" : "",
        ";  least frame dimension with σ_{2d+1} < R: ", d.dmin, d.K_reached ? " (at least)" : "")
    println("  frame: ks = ", d.ks, ", dim Q = ", d.dimQ)
    println("  far block, predicted bound by order: ",
        join(("$t: $(round(d.gbar_est[t]; sigdigits = 4))" for t in eachindex(d.gbar_est)), ", "))
    println("  order the sweep would use: ", d.m, " (predicted ḡ = ",
        round(d.gbar_est[d.m]; sigdigits = 4), ")")
    println("  γ1 ≈ ", round(d.γ1; sigdigits = 4), ", γ2 ≈ ", round(d.γ2; sigdigits = 4),
        ", min σ11 on the circle ≈ ", round(d.σ11; sigdigits = 4))
    println("  margin σ11 (R − ḡ) − γ1γ2 ≈ ", round(d.margin; sigdigits = 4),
        d.margin > 0 ? " (hypothesis predicted to hold at the samples)" :
        " (hypothesis predicted to fail)")
end
