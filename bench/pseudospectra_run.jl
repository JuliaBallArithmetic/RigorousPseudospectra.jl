# The paper's certificate, end to end, on one operator and two radii.
#
# For each operator and each target count c (10 and 5 eigenvalues outside the circle |z| = R):
#
#   1. R from a floating-point eigenvalue estimate (dense `eigvals` where n allows, Arnoldi Ritz
#      values otherwise), placed at the geometric mean of |λ_c| and |λ_{c+1}|; if those two moduli
#      agree to 1e-9 relative (a conjugate pair) the count is raised until they separate.
#   2. the frame of def:Q by the depth search of sec:depth (`choose_frame`), then one certified
#      split (`certify_split`): α, ḡ, ā (prop:fpchecks), γ1, γ2, the compression.
#   3. the arc cover of |z| = R (`check_circle`, prop:count): whether eq:assembly's hypothesis
#      holds on the whole circle, and the certified minima of σ11, σ22 and of the floor.
#   4. the outputs: the resolvent floor on circles of radius ρR, the inclusion of prop:complete,
#      the exterior count and the isolation discs of cor:agree, the projector bounds of
#      prop:projector.
#   5. the reach: with the frame fixed, the smallest R at which the exterior count still passes,
#      by bisection, and the full cor:agree there.
#
# Floating-point baselines (never certified, always reported beside the certified number): dense
# `eigvals`, dense `opnorm` of the three blocks, and σ_min(A − zI) by `svdvals` (dense, n ≤ 2200)
# or by inverse power iteration on a sparse LU (n > 2200).
#
# Usage, from the worktree root:
#
#   julia --project=. -t 32 bench/pseudospectra_run.jl <op> [<op> ...]
#   julia --project=. -t 32 bench/pseudospectra_run.jl free 100 400
#
# Results go to ~/ResearchData/KrylovPseudospectra_Data/pseudospectra/ (one set of CSVs, appended;
# one log per operator under logs/). Nothing in src/ is touched and nothing is committed.

using RigorousPseudospectra
using LinearAlgebra, SparseArrays, Random, Printf, Dates, SHA

include(joinpath(@__DIR__, "pseudospectra_ops.jl"))

# Several of these runs share the machine; cap BLAS so that four processes do not ask for
# 4 × 64 threads. PS_BLAS overrides.
BLAS.set_num_threads(parse(Int, get(ENV, "PS_BLAS", "8")))

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")
const LOGDIR = joinpath(OUT, "logs")
mkpath(LOGDIR)

const NDENSE_EIG = 6000      # dense eigvals up to this n
const NDENSE_BLOCK = 2200    # dense opnorm of the three blocks up to this n (one SVD per run)
const NDENSE_SVD = 800       # dense svdvals for the 120-point σ_min grid up to this n; above it,
                             # inverse power iteration on a sparse LU
const NTHETA_GRID = 24       # angles per circle of the resolvent grid
const RADII = [1.0, 1.05, 1.2, 1.5, 2.0]
const NTHETA = parse(Int, get(ENV, "PS_NTHETA", "64"))       # arc cover resolution
const MAXDEPTH = parse(Int, get(ENV, "PS_MAXDEPTH", "6"))    # bisections of a failing arc
const RETRY = get(ENV, "PS_RETRY", "1") == "1"               # one finer cover after a failure

# ------------------------------------------------------------------------------------------------
# CSV plumbing
# ------------------------------------------------------------------------------------------------
const HEADERS = Dict(
    "frame.csv" => ["op", "n", "target", "count_used", "R", "absl1", "R_over_absl1", "dimQ",
        "ks", "dimQ_info", "depth_rule", "alpha", "time_frame", "time_far", "time_compression",
        "time_split", "nmatvec_far", "order_m", "passed_half", "epsA", "epsP", "epsB", "gbar",
        "abar", "gamma1", "gamma2", "snorm", "dense_G22", "dense_g12", "dense_g21", "dense_normA",
        "sound_gbar", "sound_abar", "sound_gamma", "baseline_src", "maxrss_GB"],
    "circle.csv" => ["op", "n", "target", "R", "hyp_ok", "region_ok", "floor_min", "min_sigma11",
        "min_sigma22", "sigma11_samples_min", "nevals", "gamma1gamma2", "time"],
    "resolvent.csv" => ["op", "n", "target", "R", "rho", "theta", "re_z", "im_z", "floor",
        "assembly", "blockdom", "sigma11", "sigma22", "ok", "dense_sigmin", "ratio"],
    "resolvent_min.csv" => ["op", "n", "target", "R", "rho", "npts", "nok", "min_floor",
        "min_dense_sigmin", "min_ratio"],
    "inclusion.csv" => ["op", "n", "target", "R", "eps_over_normA", "eps", "delta", "ok",
        "delta_over_eps"],
    "counts.csv" => ["op", "n", "target", "R", "certified_count", "dense_count", "count_ok",
        "agree_ok", "ndiscs", "sum_disc_counts", "time"],
    "discs.csv" => ["op", "n", "target", "R", "idx", "re_c", "im_c", "radius", "kind", "count",
        "fp_count", "abs_lo", "abs_hi", "arg_lo", "arg_hi", "fp_abs_min", "fp_abs_max",
        "fp_arg_min", "fp_arg_max", "proj_ok", "proj_bound", "proj_paper_M2", "proj_M", "proj_M0",
        "proj_normP", "proj_dim", "proj_subspace_gap"],
    "projector.csv" => ["op", "n", "target", "R", "region", "re_c", "im_c", "radius", "ok",
        "bound", "M", "M0", "paper_bound_M2", "arclen", "normP", "gap", "dim", "subspace_gap",
        "eE"],
    "reach.csv" => ["op", "n", "target", "R_start", "gbar", "R_min", "R_min_over_absl1",
        "count_at_Rmin", "dense_count_at_Rmin", "agree_ok_at_Rmin", "nsteps", "time"],
    "selfadj.csv" => ["op", "n", "k", "gbar", "gammabar", "kind", "idx", "lo", "hi", "count",
        "dense_lambda", "contains"],
)

# Several operators can be run as separate processes; set PS_TAG to keep their CSVs apart
# (`frame.csv` becomes `frame.<tag>.csv`), then concatenate.
const TAG = get(ENV, "PS_TAG", "")

function row!(file, vals)
    path = joinpath(OUT, isempty(TAG) ? file : replace(file, ".csv" => ".$TAG.csv"))
    new = !isfile(path)
    open(path, "a") do io
        new && println(io, join(HEADERS[file], ","))
        println(io, join(map(_fmt, vals), ","))
    end
end
_fmt(x::AbstractString) = x
_fmt(x::Bool) = x ? "true" : "false"
_fmt(x::Integer) = string(x)
_fmt(x::Real) = isfinite(x) ? @sprintf("%.10g", x) : string(x)
_fmt(x) = string(x)

const LOGIO = Ref{IO}(stdout)
"Append a line to this operator's log and to stdout. Named `note`, not `log`: a function called
`log` in Main shadows `Base.log`, which silently broke the disc radius search once."
function note(args...)
    s = string(args...)
    println(LOGIO[], s)
    flush(LOGIO[])
    println(stdout, s)
    flush(stdout)
end

gb() = Sys.maxrss() / 2^30

"A single digest of every file of `src/`: the package is a live working tree, so the state that\nproduced a number is recorded with it."
function src_digest()
    d = joinpath(dirname(@__DIR__), "src")
    parts = String[]
    for f in sort(readdir(d))
        endswith(f, ".jl") || continue
        push!(parts, string(f, ":", bytes2hex(open(sha256, joinpath(d, f)))[1:8]))
    end
    return join(parts, " ")
end

# ------------------------------------------------------------------------------------------------
# Floating-point baselines
# ------------------------------------------------------------------------------------------------

"Eigenvalues by decreasing modulus: dense where n allows, otherwise Arnoldi Ritz values."
function fp_eigs(o)
    if o.dense && o.A !== nothing && o.n <= NDENSE_EIG
        λ = eigvals(Matrix(o.A))
        return λ[sortperm(abs.(λ); rev = true)], :dense
    end
    T = ComplexF64
    L = min(o.n - 1, 300)
    V, H = arnoldi(o.matvec, o.n, L; T)
    Lh = size(H, 2)
    λ = eigvals(H[1:Lh, 1:Lh])
    return λ[sortperm(abs.(λ); rev = true)], :arnoldi
end

"""
    admissible_count(absλ, c) -> c′

The least `c′ ≥ c` with `|λ_{c′}|` and `|λ_{c′+1}|` separated by more than 1e-9 relative: a
conjugate pair is never split.
"""
function admissible_count(a::AbstractVector, c::Integer)
    c = min(c, length(a) - 1)
    while c < length(a) - 1 && abs(a[c] - a[c + 1]) <= 1e-9 * a[c]
        c += 1
    end
    return c
end

"""
    _pownorm(X, Xadj, ncols, nrows = ncols; iters = 60) -> Float64

`‖X‖₂` estimated from below by power iteration on `X*X` (floating point, no certificate).
`X` maps `ℂ^ncols → ℂ^nrows`.
"""
function _pownorm(X, Xadj, ncols::Integer, nrows::Integer = ncols; iters::Integer = 60)
    ncols == 0 && return 0.0
    rng = MersenneTwister(11)
    v = randn(rng, ComplexF64, ncols)
    v ./= norm(v)
    s = 0.0
    for _ in 1:iters
        w = Xadj(X(v))
        s = norm(w)
        s > 0 || return 0.0
        v = w ./ s
    end
    return sqrt(s)
end



"""
    disc_enclosure(c, r) -> (abs_lo, abs_hi, arg_lo, arg_hi)

The enclosures of `|λ|` and of `arg λ` that follow from `λ ∈ D(c, r)`: `|λ| ∈ [|c| − r, |c| + r]`,
and, when `r` is below the lower bound of `|c|`, `arg λ ∈ arg c ± asin(r/|c|)`, the half-angle the
disc subtends at the origin. The moduli are rounded outward throughout; the angle is evaluated in
Float64 from an outward-rounded `r/|c|` and then widened by four ulps, so the outward rounding of
`asin` and `angle` is done once at the end rather than step by step.
"""
function disc_enclosure(c::Number, r::Real)
    c = ComplexF64(c)
    r = Float64(r)
    lo = RigorousPseudospectra.subd(RigorousPseudospectra.absd(c), r)
    hi = RigorousPseudospectra.addu(RigorousPseudospectra.absu(c), r)
    (lo > 0 && r < lo) || return (max(lo, 0.0), hi, -Float64(π), Float64(π))
    half = asin(min(1.0, RigorousPseudospectra.divu(r, lo)))
    θ = angle(c)
    w = 4 * eps(max(abs(θ) + half, 1.0))
    return (lo, hi, θ - half - w, θ + half + w)
end

"""
    circle_floor0(sc, c, r; nθ0 = 64, nmax = 8192) -> (floor0, nθ)

A certified lower bound on `min_Γ min(σ11(z), σ22(z))` over the circle `Γ = {|z − c| = r}`, by a
uniform cover doubled until the bound is positive. Both functions are 1-Lipschitz, so on the arc
between two samples the value is at least the smaller sample minus `h = rΔθ/2 + δ`, with
`δ = 8u(|c| + r)` the same allowance for the sample point that `check_circle` uses.

This exists because `CircleCheck.minσ11` and `CircleCheck.minσ22` are the arc-wise minima over
*every* arc the cover examined, coarse arcs included, and a coarse arc that was subsequently
bisected leaves a much smaller (often negative) value behind. `certify_projector` divides by
`min(minσ11, minσ22)`, so it returns `M₀ = ∞` and no bound whenever any arc needed bisection.
Feeding the value below back in through the `check` keyword uses the package's own code with a
valid, and tighter, lower bound.
"""
function circle_floor0(sc, c::Number, r::Real; nθ0::Int = 64, nmax::Int = 1024)
    c = ComplexF64(c)
    r = Float64(r)
    u = RigorousPseudospectra.unit_roundoff(Float64)
    δ = RigorousPseudospectra.mulu(8u, RigorousPseudospectra.addu(abs(c), r))
    nθ = nθ0
    while nθ <= nmax
        h = RigorousPseudospectra.addu(RigorousPseudospectra.mulu(r,
                RigorousPseudospectra.divu(RigorousPseudospectra.nextfloat(Float64(π)),
                    Float64(nθ))), δ)
        m = Inf
        for t in 0:(nθ - 1)
            z = c + r * cis(2π * t / nθ)
            m = min(m, min(sigma11_lo(sc, z), sigma22_lo(sc, z)))
        end
        v = RigorousPseudospectra.subd(m, h)
        v > 0 && return (v, nθ)
        nθ *= 2
    end
    return (0.0, nθ)
end

const SRCSTATE = Ref("")

"A `CircleCheck` with `minσ11`, `minσ22` replaced by the certified `floor0` of [`circle_floor0`](@ref)."
function with_floor0(sc, chk)
    chk.ok || return chk
    min(chk.minσ11, chk.minσ22) > 0 && return chk      # the package's own minima are usable
    f0, _ = circle_floor0(sc, chk.center, chk.radius)
    f0 > 0 || return chk
    return CircleCheck(chk.center, chk.radius, chk.ok, chk.region_ok, chk.region, chk.floor,
        chk.nevals, f0, f0)
end

"The operator as a sparse matrix when one is available (the stencil included); else `nothing`."
function sparse_form(o)
    o.A !== nothing && issparse(o.A) && return o.A
    hasproperty(o, :stencil) && return ps_convdiff_fd_sparse(o.stencil)
    return nothing
end

"σ_min(A − zI) in floating point: dense SVD for small n, inverse power iteration on a sparse LU otherwise."
function fp_sigmin(o, z; cache = Dict{Symbol, Any}())
    n = o.n
    if o.A !== nothing && (n <= NDENSE_SVD || sparse_form(o) === nothing)
        Ad = get!(cache, :dense) do
            Matrix{ComplexF64}(o.A)
        end
        return svdvals(Ad - ComplexF64(z) * I)[end]
    end
    As = get!(cache, :sparsec) do
        S = sparse_form(o)
        S === nothing ? nothing : SparseMatrixCSC{ComplexF64, Int}(S)
    end
    As === nothing && return NaN
    F = try
        lu(As - ComplexF64(z) * I)
    catch
        return NaN
    end
    rng = MersenneTwister(7)
    x = randn(rng, ComplexF64, n)
    x ./= norm(x)
    λ = 0.0
    for _ in 1:80
        y = F \ x
        w = F' \ y
        λ = norm(w)
        λ > 0 || return NaN
        x = w ./ λ
    end
    return 1 / sqrt(λ)
end

# ------------------------------------------------------------------------------------------------
# One (operator, target count)
# ------------------------------------------------------------------------------------------------

function run_target(o, λ, λsrc, target::Int, cache)
    a = abs.(λ)
    c = admissible_count(a, target)
    R = sqrt(a[c] * a[c + 1])
    absl1 = a[1]
    note("\n--- ", o.name, "  target = ", target, " -> count ", c, "  (", λsrc, ")")
    note("    |λ_$c| = ", a[c], "  |λ_$(c+1)| = ", a[c + 1], "  R = ", R, "  R/|λ1| = ", R / absl1)
    c == target || note("    NOTE: count raised from $target to $c (|λ_$target| and |λ_$(target+1)| ",
        "agree to 1e-9: a conjugate pair)")

    op = o.A === nothing ? LinOp(o.matvec, o.adjoint, o.n; epsA = o.epsA) : LinOp(o.A)

    t_frame = @elapsed ((Q, finfo) = choose_frame(op, o.matvec, o.adjoint, R; kmax = o.kmax))
    k = size(Q, 2)
    note("    frame: dim Q = ", k, "  ks = ", finfo.ks, "  depth rule = ", finfo.rule,
        "  (", round(t_frame; digits = 2), " s)")

    t_split = @elapsed (sc = certify_split(op, Q; target = R, blocksize = 32))
    far = sc.far
    note("    split: α ≤ ", sc.α, "  ḡ < ", sc.gbar, "  ā < ", sc.abar,
        "  γ1 ≤ ", sc.γ1, "  γ2 ≤ ", sc.γ2)
    note("           m = ", far === nothing ? -1 : far.m, "  half = ",
        far === nothing ? "-" : far.passed, "  ε_A = ", op.epsA, "  ε_P = ",
        far === nothing ? NaN : far.epsP, "  ε_B = ", far === nothing ? NaN : far.epsB,
        "  matvecs = ", far === nothing ? -1 : far.nmatvec, "  (", round(t_split; digits = 2), " s)")

    # floating-point baselines for the three blocks: a dense `opnorm` where n allows, otherwise a
    # power iteration on X*X (a lower bound on the norm, which is the direction a soundness check
    # on an upper bound needs)
    dG22 = dg12 = dg21 = dnA = NaN
    basesrc = "none"
    if o.A !== nothing && o.n <= NDENSE_BLOCK
        basesrc = "dense opnorm"
        Ad = get!(cache, :dense) do
            Matrix{ComplexF64}(o.A)
        end
        Qm = Matrix{ComplexF64}(Q)
        PA = Ad - (Qm * (Qm' * Ad))
        dG22 = opnorm(PA - ((PA * Qm) * Qm'))
        dg12 = opnorm(Qm' * (Ad - (Ad * Qm) * Qm'))
        dg21 = opnorm(Ad * Qm - Qm * (Qm' * Ad * Qm))
        dnA = opnorm(Ad)
    else
        basesrc = "power iteration (lower bound)"
        Qm = Matrix{ComplexF64}(Q)
        pp(x) = x - Qm * (Qm' * x)
        mv(x) = ComplexF64.(o.matvec(x))
        mvt(x) = ComplexF64.(o.adjoint(x))
        dnA = _pownorm(mv, mvt, o.n)
        dG22 = _pownorm(x -> pp(mv(pp(x))), x -> pp(mvt(pp(x))), o.n)
        dg12 = _pownorm(x -> Qm' * mv(pp(x)), x -> pp(mvt(Qm * x)), o.n, size(Qm, 2))
        dg21 = _pownorm(x -> pp(mv(Qm * x)), x -> Qm' * mvt(pp(x)), size(Qm, 2), o.n)
    end
    note("    baseline (", basesrc, "): ‖G22‖ = ", dG22, "  ‖G12‖ = ", dg12, "  ‖G21‖ = ", dg21,
        "  ‖A‖ = ", dnA)
    sound_g = isnan(dG22) ? true : sc.gbar >= dG22
    sound_a = isnan(dnA) ? true : sc.abar >= dnA
    sound_c = (isnan(dg12) ? true : sc.γ1 >= dg12) && (isnan(dg21) ? true : sc.γ2 >= dg21)
    (sound_g && sound_a && sound_c) ||
        note("!!! CERTIFIED QUANTITY BELOW ITS DENSE VALUE: gbar=$sound_g abar=$sound_a gamma=$sound_c")

    row!("frame.csv", [o.name, o.n, target, c, R, absl1, R / absl1, k, finfo.ks, finfo.dimQ,
        finfo.rule, sc.α, sc.time.frame, sc.time.far, sc.time.compression, t_split,
        far === nothing ? -1 : far.nmatvec, far === nothing ? -1 : far.m,
        far === nothing ? "-" : string(far.passed), op.epsA,
        far === nothing ? NaN : far.epsP, far === nothing ? NaN : far.epsB,
        sc.gbar, sc.abar, sc.γ1, sc.γ2, sc.snorm, dG22, dg12, dg21, dnA,
        sound_g, sound_a, sound_c, basesrc, gb()])

    # --- the arc cover of |z| = R -----------------------------------------------------------
    t_chk = @elapsed begin
        chk = check_circle(sc, 0.0, R; region = :exterior, nθ = NTHETA, maxdepth = MAXDEPTH)
        # the cover, not the hypothesis, can be what fails: the arc-wise bound subtracts
        # h = rΔθ/2, so a small σ11 needs a fine cover. Retry once, four times finer.
        if !chk.ok && RETRY
            chk2 = check_circle(sc, 0.0, R; region = :exterior, nθ = 4 * NTHETA, maxdepth = 12)
            chk2.ok && note("    (the coarse cover nθ = ", NTHETA, ", maxdepth = ", MAXDEPTH,
                " failed; nθ = ", 4 * NTHETA, ", maxdepth = 12 succeeds)")
            chk = chk2.ok ? chk2 : chk
        end
    end
    # the certified σ11 at the NTHETA sample points themselves (the arc-wise minimum `chk.minσ11`
    # is taken over every arc the cover examined, coarse arcs included, so it is a valid but much
    # more pessimistic lower bound)
    s11s = minimum(sigma11_lo(sc, R * cis(2π * t / NTHETA)) for t in 0:(NTHETA - 1))
    note("    circle |z| = R: hypothesis on the whole circle = ", chk.ok, "  region = ",
        chk.region_ok, "  floor ≥ ", chk.floor, "  min σ11 over the cover ≥ ", chk.minσ11,
        "  min σ11 at the ", NTHETA, " samples = ", s11s, "  min σ22 ≥ ",
        chk.minσ22, "  (", chk.nevals, " evaluations, ", round(t_chk; digits = 2), " s)")
    row!("circle.csv", [o.name, o.n, target, R, chk.ok, chk.region_ok, chk.floor, chk.minσ11,
        chk.minσ22, s11s, chk.nevals, sc.γ1 * sc.γ2, t_chk])

    # --- the resolvent bound on {|z| > R} ----------------------------------------------------
    for ρ in RADII
        mn = Inf
        mnd = Inf
        mr = Inf
        nok = 0
        for t in 0:(NTHETA_GRID - 1)
            θ = 2π * t / NTHETA_GRID
            z = ρ * R * cis(θ)
            r = resolvent_floor(sc, z)
            ds = fp_sigmin(o, z; cache)
            rat = (isfinite(ds) && ds > 0) ? r.floor / ds : NaN
            if isfinite(ds) && r.ok && r.floor > ds * (1 + 1e-8) + 1e-300
                note("!!! FLOOR ABOVE THE DENSE σ_min AT z = ", z, ": ", r.floor, " > ", ds)
            end
            r.ok && (nok += 1)
            mn = min(mn, r.floor)
            isfinite(ds) && (mnd = min(mnd, ds))
            isfinite(rat) && (mr = min(mr, rat))
            row!("resolvent.csv", [o.name, o.n, target, R, ρ, θ, real(z), imag(z), r.floor,
                r.assembly, r.blockdom, r.σ11, r.σ22, r.ok, ds, rat])
        end
        note("    |z| = ", ρ, "R: ", nok, "/", NTHETA_GRID, " points bounded, min floor = ", mn,
            ", min dense σ_min = ", mnd, ", min ratio = ", mr)
        row!("resolvent_min.csv", [o.name, o.n, target, R, ρ, NTHETA_GRID, nok, mn, mnd, mr])
    end

    # --- prop:complete -----------------------------------------------------------------------
    nA = sc.abar
    for e in 1:8
        ε = 10.0^(-e) * nA
        δ, okd = pseudospectrum_outside(sc, ε, R)
        row!("inclusion.csv", [o.name, o.n, target, R, 10.0^(-e), ε, δ, okd, okd ? δ / ε : NaN])
    end
    let ε1 = 1e-1 * nA, ε8 = 1e-8 * nA
        d1, o1 = pseudospectrum_outside(sc, ε1, R)
        d8, o8 = pseudospectrum_outside(sc, ε8, R)
        note("    inclusion: ε = 1e-1‖A‖ -> δ = ", d1, " (δ/ε = ", o1 ? d1 / ε1 : NaN,
            "),  ε = 1e-8‖A‖ -> δ = ", d8, " (δ/ε = ", o8 ? d8 / ε8 : NaN, ")")
    end

    # --- cor:agree ---------------------------------------------------------------------------
    # `count_exterior` returns 0 with `ok = false` when the hypothesis is not certified on the
    # circle: the count is then absent, not wrong, so it is compared only when `okcnt` is true
    t_ext = @elapsed begin
        cnt, okcnt, _ = count_exterior(sc, R; nθ = NTHETA, maxdepth = MAXDEPTH)
        ext = certify_exterior(sc, R; nθ = NTHETA, nradii = 10, maxdepth = MAXDEPTH)
    end
    dense_count = λsrc === :dense ? count(>(R), a) : -1
    ok_count = !okcnt ? "no count" :
               (dense_count < 0 ? "no baseline" : string(cnt == dense_count))
    (okcnt && dense_count >= 0 && cnt != dense_count) &&
        note("!!! CERTIFIED EXTERIOR COUNT ", cnt, " ≠ FLOATING-POINT COUNT ", dense_count)
    note("    exterior: count certified = ", okcnt, ", certified count = ", cnt, " (floating point ",
        dense_count, "), cor:agree ok = ", ext.ok, ", ", length(ext.discs), " discs  (",
        round(t_ext; digits = 2), " s)")
    row!("counts.csv", [o.name, o.n, target, R, okcnt ? cnt : -1, dense_count, ok_count, ext.ok,
        length(ext.discs), sum(d -> d.count, ext.discs; init = 0), t_ext])
    onepu = nextfloat(1.0)
    eE = max(sc.γ1, sc.γ2)
    function put_disc(i, c, r, kind, cnt_i)
        # a disc with a non-finite radius is one `certify_exterior` could not close: there is no
        # certified count, no enclosure and no projector
        if !isfinite(r)
            row!("discs.csv", [o.name, o.n, target, R, i, real(c), imag(c), NaN, kind, cnt_i, -1,
                NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, false, NaN, NaN, NaN, NaN, NaN, -1, NaN])
            return
        end
        al, ah, gl, gh = disc_enclosure(c, r)
        inside = λsrc === :dense ? [z for z in λ if abs(z - c) < r] : ComplexF64[]
        fpn = λsrc === :dense ? length(inside) : -1
        (fpn >= 0 && fpn != cnt_i) &&
            note("!!! DISC ", i, " (", kind, ") CERTIFIED COUNT ", cnt_i,
                " ≠ FLOATING-POINT COUNT ", fpn)
        p = certify_projector(sc, c, r; region = :interior, nθ = NTHETA, maxdepth = MAXDEPTH,
            check = with_floor0(sc, check_circle(sc, c, r; region = :interior, nθ = NTHETA,
                maxdepth = MAXDEPTH)))
        row!("discs.csv", [o.name, o.n, target, R, i, real(c), imag(c), r, kind, cnt_i, fpn,
            al, ah, gl, gh,
            isempty(inside) ? NaN : minimum(abs, inside),
            isempty(inside) ? NaN : maximum(abs, inside),
            isempty(inside) ? NaN : minimum(angle, inside),
            isempty(inside) ? NaN : maximum(angle, inside),
            p.ok, p.bound, r * p.M^2 * eE, p.M, p.M0, p.normP, p.count, p.subspace_gap])
        note("      disc ", i, " (", kind, " radius) c = ", c, ", r = ", r, ", count = ", cnt_i,
            " (fp ", fpn, "), |λ| ∈ [", al, ", ", ah, "], arg λ ∈ [", gl, ", ", gh,
            "], ‖P − P₀‖ ≤ ", p.bound, " (ok = ", p.ok, ", M = ", p.M, ", M₀ = ", p.M0, ")")
    end
    for (i, d) in enumerate(ext.discs)
        if !isfinite(d.radius)
            put_disc(i, d.center, NaN, "not closed", d.count)
            continue
        end
        put_disc(i, d.center, d.radius, "min", d.count)
        # prop:projector degrades as the contour approaches the spectrum, so also take the contour
        # as large as prop:count allows: inside {|z| > R} and disjoint from the other components
        rmax = RigorousPseudospectra.subd(RigorousPseudospectra.absd(d.center), Float64(R))
        for (j, e) in enumerate(ext.discs)
            j == i && continue
            rmax = min(rmax, RigorousPseudospectra.divd(
                RigorousPseudospectra.absd(d.center - e.center), 2 * onepu))
        end
        rbig = NaN
        if rmax > d.radius
            for r in reverse(exp.(range(Base.log(d.radius * (1 + 1e-6)),
                Base.log(rmax * (1 - 1e-6)); length = 10)))
                c2, okc, _ = count_disc(sc, d.center, r; nθ = NTHETA, maxdepth = MAXDEPTH)
                if okc && c2 == d.count
                    rbig = r
                    break
                end
            end
        end
        isfinite(rbig) && put_disc(i, d.center, rbig, "max", d.count)
    end

    # --- prop:projector ----------------------------------------------------------------------
    t_pr = @elapsed begin
        eE = max(sc.γ1, sc.γ2)
        # the package evaluates prop:projector as r·M·M₀·‖E‖ (the second resolvent identity),
        # which is at or below the paper's eq:projector r·M²·‖E‖; both are recorded
        putpr(reg, c, r, p) = row!("projector.csv", [o.name, o.n, target, R, reg, real(c),
            imag(c), r, p.ok, p.bound, p.M, p.M0, r * p.M^2 * eE, p.arclen, p.normP, p.gap,
            p.count, p.subspace_gap, eE])
        pc = certify_projector(sc, 0.0, R; region = :exterior, nθ = NTHETA, maxdepth = MAXDEPTH,
            check = with_floor0(sc, chk))
        putpr("exterior", 0.0 + 0.0im, R, pc)
        note("    projector (exterior |z| > R): ok = ", pc.ok, "  ‖P − P₀‖ ≤ ", pc.bound,
            " (paper eq:projector r M² ‖E‖ = ", R * pc.M^2 * eE, ")  M ≤ ", pc.M, "  M₀ ≤ ",
            pc.M0, "  ‖P‖ ≤ ", pc.normP, "  dim = ", pc.count, "  subspace gap ≤ ",
            pc.subspace_gap)
    end
    note("    projector: ", round(t_pr; digits = 2), " s for the exterior contour")

    # --- the reach: bisection in R with the frame fixed ---------------------------------------
    t_re = @elapsed begin
        lo = nextfloat(sc.gbar)               # nothing below ḡ can pass
        hi = R
        passhi = count_exterior(sc, R; nθ = NTHETA, maxdepth = MAXDEPTH)[2]
        nsteps = 0
        if !passhi
            lo = hi = NaN
        else
            for _ in 1:14
                mid = sqrt(lo * hi)
                nsteps += 1
                if count_exterior(sc, mid; nθ = NTHETA, maxdepth = MAXDEPTH)[2]
                    hi = mid
                else
                    lo = mid
                end
                hi / lo < 1.002 && break
            end
        end
    end
    Rmin = hi
    if isfinite(Rmin)
        cm, okm, _ = count_exterior(sc, Rmin; nθ = NTHETA, maxdepth = MAXDEPTH)
        extm = certify_exterior(sc, Rmin; nθ = NTHETA, nradii = 10, maxdepth = MAXDEPTH)
        dcm = λsrc === :dense ? count(>(Rmin), a) : -1
        (okm && dcm >= 0 && cm != dcm) &&
            note("!!! CERTIFIED COUNT AT R MIN ", cm, " ≠ FLOATING-POINT COUNT ", dcm)
        note("    reach: smallest certified R = ", Rmin, " (", Rmin / absl1, " of |λ1|; ḡ = ",
            sc.gbar, "), count outside = ", cm, " (floating point ", dcm, "), cor:agree = ",
            extm.ok, "  (", round(t_re; digits = 2), " s, ", nsteps, " bisection steps)")
        row!("reach.csv", [o.name, o.n, target, R, sc.gbar, Rmin, Rmin / absl1, okm ? cm : -1, dcm,
            extm.ok, nsteps, t_re])
    else
        note("    reach: the exterior count does not pass even at R itself")
        row!("reach.csv", [o.name, o.n, target, R, sc.gbar, NaN, NaN, -1, -1, false, nsteps, t_re])
    end
    return sc
end

# ------------------------------------------------------------------------------------------------
# Self-adjoint extra (cor:selfadj, cor:weyl, rem:lanczos)
# ------------------------------------------------------------------------------------------------
function run_selfadjoint(o, λ)
    note("\n--- ", o.name, ": rem:lanczos frame, cor:selfadj and cor:weyl")
    t = @elapsed (sa = certify_selfadjoint(o.A; L = min(300, o.n - 1), tol = 1e-10,
        kmax = o.kmax, blocksize = 32))
    sc = sa.split
    note("    Lanczos frame: k = ", sc.k, " of ", sa.frame.L, " steps, α ≤ ", sc.α,
        ", ḡ < ", sa.ḡ, ", γ̄ ≤ ", sa.γ̄, "  (", round(t; digits = 2), " s)")
    dec = sort(real.(λ); rev = true)          # λ_1 ≥ λ_2 ≥ … ≥ λ_n
    put(kind, idx, lo, hi, cnt, dl) = row!("selfadj.csv",
        [o.name, o.n, sc.k, sa.ḡ, sa.γ̄, kind, idx, lo, hi, cnt, dl,
            isnan(dl) ? true : (lo <= dl <= hi)])
    for (i, th) in enumerate(sa.θ)
        put("theta", i, th.lo, th.hi, th.count, NaN)
    end
    for (i, en) in enumerate(sa.enclosure)
        put("enclosure", i, en.lo, en.hi, en.count, NaN)
    end
    w = sa.weyl
    note("    cor:weyl: ok = ", w.ok, ", j₊ = ", w.jplus, ", j₋ = ", w.jminus,
        ", λ_{j₊+1} ≤ ", w.top_rest, ", λ_{n−j₋} ≥ ", w.bottom_rest)
    if w.ok
        for t in w.top
            dl = t.i <= length(dec) ? dec[t.i] : NaN
            put("weyl_top", t.i, t.lo, t.hi, 1, dl)
            (isnan(dl) || t.lo <= dl <= t.hi) ||
                note("!!! cor:weyl TOP INTERVAL ", t.i, " [", t.lo, ", ", t.hi, "] MISSES λ = ", dl)
        end
        for t in w.bottom
            dl = t.i <= length(dec) ? dec[t.i] : NaN
            put("weyl_bottom", t.i, t.lo, t.hi, 1, dl)
            (isnan(dl) || t.lo <= dl <= t.hi) ||
                note("!!! cor:weyl BOTTOM INTERVAL ", t.i, " [", t.lo, ", ", t.hi, "] MISSES λ = ", dl)
        end
    end
    # cor:selfadj (i): every λ with |λ| > ḡ must lie in one of the enclosure intervals
    if !isempty(sa.enclosure)
        miss = [x for x in dec if abs(x) > sa.ḡ &&
                !any(J -> J.lo <= x <= J.hi, sa.enclosure)]
        isempty(miss) ||
            note("!!! cor:selfadj (i) MISSES ", length(miss), " dense eigenvalues: ",
                join(round.(miss; sigdigits = 8), ", "))
        note("    cor:selfadj (i): ", count(x -> abs(x) > sa.ḡ, dec),
            " dense eigenvalues with |λ| > ḡ, ", length(miss), " outside the enclosure")
    end
    # cor:selfadj (ii): a count on an interval about the largest Ritz interval
    if !isempty(sa.θ)
        J = sa.θ[argmax([j.hi for j in sa.θ])]
        wdt = max(sa.γ̄ * 4, 1e-6 * abs(J.hi))
        cnt, okc = selfadj_count(sa.θ, sa.ḡ, sa.γ̄, J.lo - wdt, J.hi + wdt)
        dc = count(x -> J.lo - wdt <= x <= J.hi + wdt, dec)
        note("    cor:selfadj (ii) on [", J.lo - wdt, ", ", J.hi + wdt, "]: count = ", cnt,
            " (ok = ", okc, "), dense = ", dc)
        okc && cnt != dc && note("!!! cor:selfadj (ii) COUNT ", cnt, " ≠ DENSE ", dc)
        put("count_ii", 0, J.lo - wdt, J.hi + wdt, cnt, NaN)
    end
    return sa
end

# ------------------------------------------------------------------------------------------------
function run_operator(o; targets = (10, 5), selfadj = false)
    LOGIO[] = open(joinpath(LOGDIR, "$(o.name).log"), "a")
    try
        note("\n", "="^92)
        note(o.name, "   n = ", o.n, "   ", o.source)
        note("started ", now(), "  threads = ", Threads.nthreads(), "  BLAS = ",
            BLAS.get_num_threads())
        note("src/ state at process load: ", SRCSTATE[])
        te = @elapsed ((λ, src) = fp_eigs(o))
        note("eigenvalue estimate (", src, ", ", round(te; digits = 2), " s): |λ| = ",
            join(round.(abs.(λ[1:min(13, end)]); sigdigits = 7), ", "), " ...")
        for c in targets
            try
                run_target(o, λ, src, c, Dict{Symbol, Any}())
            catch e
                note("!!! target ", c, " failed: ", sprint(showerror, e))
                for l in stacktrace(catch_backtrace())[1:min(12, end)]
                    note("      ", l)
                end
            end
        end
        if selfadj
            try
                run_selfadjoint(o, λ)
            catch e
                note("!!! self-adjoint branch failed: ", sprint(showerror, e))
            end
        end
        note("finished ", now(), "  peak RSS = ", round(gb(); digits = 2), " GB")
    finally
        close(LOGIO[])
        LOGIO[] = stdout
    end
end

# ------------------------------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    SRCSTATE[] = src_digest()
    args = copy(ARGS)
    while !isempty(args)
        name = popfirst!(args)
        if name == "free"
            N1 = parse(Int, popfirst!(args))
            N2 = parse(Int, popfirst!(args))
            km = (!isempty(args) && all(isdigit, args[1])) ? parse(Int, popfirst!(args)) : 150
            run_operator(ps_free_operator(N1, N2; kmax = km))
        else
            o = ps_operator(name)
            run_operator(o; selfadj = o.hermitian)
        end
    end
end
