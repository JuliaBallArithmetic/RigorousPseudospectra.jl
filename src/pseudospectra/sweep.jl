# Box-bisection sweep + adaptive refinement (implementation.md §6).
#
# NOT a uniform grid. z ↦ σ_min(A−zI) is 1-Lipschitz (Lemma `lipschitz`), so one
# certified centre evaluation settles a whole cell: on a cell of half-diagonal r
# centred at z₀,
#   σ_min(A−zI) ≥ σ_lo(z₀) − r      (EXCLUSION: cell outside σ_ε if σ_lo(z₀)−r > ε)
#   σ_min(A−zI) ≤ σ_hi(z₀) + r      (INCLUSION: cell inside  σ_ε if σ_hi(z₀)+r < ε)
# Cells that straddle ε are quadrisected until a target resolution. Refinement
# concentrates on the ε-contour (a 1-D set), so cell count scales with
# contour-length/resolution, not area/resolution².
#
# Poles: an eigenvalue has σ_min = 0, so cells reaching it can never be EXCLUDED —
# they are INCLUDED via σ_hi (or refined to the boundary set). Exclusion is for the
# pole-free bulk; the encircling-curve certificate (cluster.jl) handles the rest.

"""
    Cell

An axis-aligned rectangular cell in the complex plane: centre `c`, real half-width
`hx`, imaginary half-height `hy`.
"""
struct Cell
    c::ComplexF64
    hx::Float64
    hy::Float64
end

"""
    Cell(lo::Complex, hi::Complex) -> Cell

Cell spanning the rectangle with opposite corners `lo`, `hi`.
"""
Cell(lo::Complex, hi::Complex) = Cell((lo + hi) / 2,
    abs(real(hi - lo)) / 2, abs(imag(hi - lo)) / 2)

halfdiag(cell::Cell) = hypot(cell.hx, cell.hy)

function quadrisect(cell::Cell)
    hx, hy = cell.hx / 2, cell.hy / 2
    return (Cell(cell.c + complex(-hx, -hy), hx, hy),
        Cell(cell.c + complex(hx, -hy), hx, hy),
        Cell(cell.c + complex(-hx, hy), hx, hy),
        Cell(cell.c + complex(hx, hy), hx, hy))
end

"""
    PseudospectrumEnclosure

Result of a box-bisection sweep at level `ε`: cells proven outside σ_ε
(`excluded`), proven inside (`included`), and unresolved at target resolution
(`boundary`, straddling the ε-contour), plus the approximate eigenvalues for
overlay and the per-excluded-cell routing label (which certificate settled it).
"""
struct PseudospectrumEnclosure
    excluded::Vector{Cell}
    included::Vector{Cell}
    boundary::Vector{Cell}
    eigs::Vector{ComplexF64}
    ε::Float64
    routing::Vector{Symbol}     # parallel to `excluded`
end

"""
    default_box(S::CertSetup; pad = 0.5) -> (lo, hi)

Bounding box of the approximate spectrum, padded by `pad` (absolute) on each side.
"""
function default_box(S::CertSetup; pad::Real = 0.5)
    xs = real.(S.eigs)
    ys = imag.(S.eigs)
    lo = complex(minimum(xs) - pad, minimum(ys) - pad)
    hi = complex(maximum(xs) + pad, maximum(ys) + pad)
    return (lo, hi)
end

# distance from a point p to the (closed) rectangular cell (0 if inside)
function _dist_point_to_cell(p, cell::Cell)
    dx = max(abs(real(p) - real(cell.c)) - cell.hx, 0.0)
    dy = max(abs(imag(p) - imag(cell.c)) - cell.hy, 0.0)
    return hypot(dx, dy)
end

# Provably eigenvalue-free: the cell is disjoint from every Gershgorin disc of the
# collapsed matrix M₀ (= Z⁻¹AZ, so σ(A)=σ(M₀)), B(centresᵢ, radrowᵢ), inflated by
# the slack β. The maximum principle for the (subharmonic) resolvent norm needs
# the region to contain no pole, which this certifies.
function _eigfree(cell::Cell, S::CertSetup)
    @inbounds for i in eachindex(S.centres)
        if _dist_point_to_cell(S.centres[i], cell) <= S.radrow[i] + S.beta_inf
            return false
        end
    end
    return true
end

# σ_min lower bound on a segment [za,zb] by 1-Lipschitz arc-covering at the given
# spacing; slack = half the sample spacing. Returns (bound, worst-point).
#
# Sample count is capped (≤ _MAX_EDGE_SAMPLES) so boundary exclusion is O(1) per
# cell regardless of size: a big benign cell has huge margin and needs only a few
# boundary points (its slack is then coarse but far below the margin), while a
# small near-contour cell already needs few samples at the target spacing. Without
# the cap, a box-sized cell would be sampled at the ε-resolution — O(box/ε) waste.
const _MAX_EDGE_SAMPLES = 33

function _segment_lower(f, za, zb, spacing::Real)
    L = abs(zb - za)
    m = clamp(ceil(Int, L / spacing) + 1, 2, _MAX_EDGE_SAMPLES)
    slack = L / (2 * (m - 1))
    best = Inf
    zworst = za
    for k in 0:(m - 1)
        z = za + (zb - za) * (k / (m - 1))
        v = f(z)
        v < best && (best = v; zworst = z)
    end
    return best - slack, zworst
end

# σ_min lower bound over the cell BOUNDARY (4 edges). On an eigenvalue-free cell,
# the maximum principle promotes this to a lower bound over the whole cell.
function _boundary_lower(f, cell::Cell, spacing::Real)
    c, hx, hy = cell.c, cell.hx, cell.hy
    corners = (c + complex(-hx, -hy), c + complex(hx, -hy),
        c + complex(hx, hy), c + complex(-hx, hy))
    best = Inf
    zworst = c
    for e in 1:4
        v, zw = _segment_lower(f, corners[e], corners[e % 4 + 1], spacing)
        v < best && (best = v; zworst = zw)
    end
    return best, zworst
end

# core bisection over callables; returns the raw cell lists + routing labels.
# Tiered, cheapest-first, all certified:
#   cheap EXCLUSION  — (B) boundary + max principle if eigfree, else (A) centre
#                      Lipschitz lower(z₀) − r > ε;
#   cheap INCLUSION  — upper(z₀) + r < ε;
#   ESCALATE         — only when the cheap layers leave the cell unresolved: the
#                      block/cluster tier `escalate(z₀) = (lo, hi)` (a tight per-z
#                      verified-SVD bound on the relevant cluster), tried for both
#                      exclusion (lo − r > ε) and inclusion (hi + r < ε);
#   else subdivide / mark boundary at target resolution.
function _bisect_cells(lower, upper, eigfree, escalate, router, root::Cell,
        ε::Real, min_halfdiag::Real, bnd_spacing::Real)
    excluded = Cell[]
    included = Cell[]
    boundary = Cell[]
    routing = Symbol[]
    stack = Cell[root]
    while !isempty(stack)
        cell = pop!(stack)
        z0 = cell.c
        r = halfdiag(cell)
        # --- cheap exclusion ---
        if eigfree(cell)
            cheapL, zw = _boundary_lower(lower, cell, bnd_spacing)   # (B) max principle
        else
            cheapL, zw = lower(z0) - r, z0                           # (A) centre Lipschitz
        end
        if cheapL > ε
            push!(excluded, cell)
            push!(routing, router(zw))
            continue
        end
        # --- cheap inclusion ---
        if upper(z0) + r < ε
            push!(included, cell)
            continue
        end
        # --- escalate to the block/cluster tier (only the unresolved cells) ---
        escL, escH = escalate(z0)
        if escL - r > ε
            push!(excluded, cell)
            push!(routing, :cluster)
            continue
        elseif isfinite(escH) && escH + r < ε
            push!(included, cell)
            continue
        end
        # --- still unresolved ---
        if r < min_halfdiag
            push!(boundary, cell)
        else
            for sub in quadrisect(cell)
                push!(stack, sub)
            end
        end
    end
    return excluded, included, boundary, routing
end

# which scalar floor is binding at z (for the routing histogram).
function _binding_layer(z, S::CertSetup, frame::SVDFrame)
    gt = gap_tri(z, S)
    gs = gap_svd(z, frame)
    gr = gap_recentred(z, S)
    best = max(gt, gs, gr)
    best == gr ? :recentred : (best == gs ? :svd : :triangular)
end

# Cached block-tier certificates for the sweep: one re-certified ClusterBlock per
# coupling-graph component with ≥ 2 eigenvalues (singletons are handled by the
# scalar floors). Built once; reused across all cells (the O(n³) reorder is NOT
# repeated per cell). `n` is the matrix size, used to spot a whole-matrix cluster.
struct SweepClusters{C}
    blocks::Vector{C}
    n::Int
end

function _build_sweep_clusters(S::CertSetup; θ::Real, ε_target::Real)
    n = size(S.A, 1)
    blocks = ClusterBlock[]
    for c in cluster_components(S; θ, ε_target)
        length(c) >= 2 && push!(blocks, reorder_cluster(S, c))
    end
    return SweepClusters(blocks, n)
end

# escalated σ_min bracket at z: the tightest cluster LOWER bound (gap_cluster) over
# the cached blocks, and — for a whole-matrix block, where the cluster SVD is the
# exact σ_min — a tight UPPER bound σ_min·κ₂ as well (else Inf).
function _cluster_bracket(z, sc::SweepClusters)
    lo = -Inf
    hi = Inf
    for cb in sc.blocks
        g = gap_cluster(z, cb)
        g > lo && (lo = g)
        if cb.m == sc.n
            hi = min(hi, sigma_min_block(z, cb) * cb.rs.kappa_2)
        end
    end
    return lo, hi
end

# z-ADAPTIVE active set (the failing / almost-failing gaps at z): eigenvalues whose
# free triangular gap |z − t̃ᵢ| − τᵢ is within `margin` of failing. The complement
# is then automatically well-separated (its triangular gaps exceed `margin`), so the
# block assembly's far-gap/localization hold by construction. Returned sorted, so
# equal active sets share a cache key.
function _active_set(z, S::CertSetup, margin::Real)
    A = Int[]
    @inbounds for i in eachindex(S.eigs)
        d = abs(z - S.eigs[i])
        if d - max(S.tailrow[i], S.tailcol[i]) <= margin
            push!(A, i)
        end
    end
    return A
end

# adaptive escalation: build (or reuse) the block on the active set at z, cached by
# its signature. A generous margin makes the active set vary slowly in z, so the
# O(n³) reorder+re-certification is amortized over many cells (few distinct windows).
function _adaptive_bracket(z, S::CertSetup, cache::Dict{Vector{Int}, ClusterBlock},
        margin::Real)
    A = _active_set(z, S, margin)
    isempty(A) && return (-Inf, Inf)
    cb = get!(() -> reorder_cluster(S, A), cache, A)
    lo = gap_cluster(z, cb)
    hi = cb.m == size(S.A, 1) ? sigma_min_block(z, cb) * cb.rs.kappa_2 : Inf
    return lo, hi
end

"""
    bisect(cf::CheapFloor; box, ε, min_halfdiag, cluster = true, θ = 0.25) -> PseudospectrumEnclosure

Adaptive box-bisection enclosure of the ε-pseudospectrum, tiered cheapest-first:
the cheap floor `max(gap_tri, gap_svd, gap_recentred)` (with the resolvent
maximum principle on eigenvalue-free cells) drives EXCLUSION and `sigma_min_upper`
drives INCLUSION; when both are inconclusive on a cell, it **escalates to the
block/cluster tier** — a tight per-z verified-SVD bound (`gap_cluster`) on each
coupling-graph cluster (built once, cached). `cluster = false` disables the
escalation (cheap floors only); `θ` is the coupling-graph threshold (§4b).
`box` is a `(lo, hi)` corner pair or a `Cell`.
"""
function bisect(cf::CheapFloor; box = default_box(cf.setup), ε::Real,
        min_halfdiag::Real, cluster::Bool = true, θ::Real = 0.25,
        adaptive::Bool = false, active_margin::Union{Nothing, Real} = nothing)
    S = cf.setup
    frame = cf.frame
    root = box isa Cell ? box : Cell(box[1], box[2])
    lower = z -> max(cf(z), gap_recentred(z, S))
    upper = z -> sigma_min_upper(z, S)
    eigfree = cell -> _eigfree(cell, S)
    router = z -> _binding_layer(z, S, frame)
    escalate = if !cluster
        _ -> (-Inf, Inf)
    elseif adaptive
        # z-adaptive: block on the failing gaps at z, cached by active-set signature.
        # margin defaults to the matrix's own off-diagonal scale (max tail).
        margin = active_margin === nothing ?
                 max(maximum(S.tailrow), maximum(S.tailcol)) : active_margin
        cache = Dict{Vector{Int}, ClusterBlock}()
        z -> _adaptive_bracket(z, S, cache, margin)
    else
        sc = _build_sweep_clusters(S; θ, ε_target = ε)
        isempty(sc.blocks) ? (_ -> (-Inf, Inf)) : (z -> _cluster_bracket(z, sc))
    end
    excluded, included, boundary, routing = _bisect_cells(lower, upper, eigfree,
        escalate, router, root, ε, min_halfdiag, min_halfdiag)
    return PseudospectrumEnclosure(excluded, included, boundary, S.eigs,
        Float64(ε), routing)
end

"""
    bisect(A::AbstractMatrix; ε, min_halfdiag, box = nothing) -> PseudospectrumEnclosure

Convenience: build the certified setup + cheap floor for `A`, then sweep.
"""
function bisect(A::AbstractMatrix; ε::Real, min_halfdiag::Real, box = nothing,
        cluster::Bool = true, θ::Real = 0.25, adaptive::Bool = false,
        active_margin::Union{Nothing, Real} = nothing)
    cf = cheap_floor(A)
    b = box === nothing ? default_box(cf.setup) : box
    return bisect(cf; box = b, ε, min_halfdiag, cluster, θ, adaptive, active_margin)
end

"""
    routing_histogram(enc::PseudospectrumEnclosure) -> Dict{Symbol,Int}

Counts of which certificate settled each excluded cell — the central benchmark
(implementation.md §9): how much of the plane the cheap layers settle.
"""
function routing_histogram(enc::PseudospectrumEnclosure)
    h = Dict{Symbol, Int}()
    for s in enc.routing
        h[s] = get(h, s, 0) + 1
    end
    return h
end
