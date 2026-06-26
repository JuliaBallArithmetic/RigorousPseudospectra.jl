# Clustering by the coupling graph (implementation.md §4b).
#
# Block membership for the cluster path is decided ONCE, from the Schur factor T̃,
# by the structure of the off-diagonal mass — NOT by a modulus/annulus heuristic.
# The strict-upper entries T̃_ij are exactly what a scalar (per-eigenvalue) bound
# cannot absorb, so eigenvalues linked by above-threshold coupling must share a
# block. Clusters are the connected components of the coupling graph, which
# automatically capture transitive coupling (a chain i–k–j groups i and j even
# with T̃_ij ≈ 0, because the resolvent sees the composite path).
#
# Threshold is gap-relative and ε-tied:
#     θ_ij = θ · min(|t̃_i − t̃_j|, ε_target),     θ ∈ (0,1)
# coupling matters when comparable to the separation it bridges, and couplings
# below the resolution being certified need not be blocked. This is a Float64
# (graph-structural) computation — no certified arithmetic is involved in deciding
# membership; the certified work happens later, on the resulting blocks.

"""
    coupling_graph(T, eigs; θ, ε_target = Inf) -> BitMatrix

Symmetric boolean adjacency on the strict upper triangle of the Schur factor `T`:
`G[i,j] = G[j,i] = |T[i,j]| > θ·min(|eigs[i]−eigs[j]|, ε_target)` for `i < j`.
"""
function coupling_graph(T::AbstractMatrix, eigs::AbstractVector; θ::Real, ε_target::Real = Inf)
    n = size(T, 1)
    G = falses(n, n)
    @inbounds for j in 1:n, i in 1:(j - 1)
        θij = θ * min(abs(eigs[i] - eigs[j]), ε_target)
        if abs(T[i, j]) > θij
            G[i, j] = true
            G[j, i] = true
        end
    end
    return G
end

# connected components of an undirected boolean adjacency, via union–find.
# Returns a Vector of sorted index-sets, ordered by least element.
function _connected_components(G::AbstractMatrix{Bool})
    n = size(G, 1)
    parent = collect(1:n)
    find(x) = begin
        r = x
        while parent[r] != r
            r = parent[r]
        end
        while parent[x] != r          # path compression
            parent[x], x = r, parent[x]
        end
        r
    end
    @inbounds for j in 1:n, i in 1:(j - 1)
        if G[i, j] || G[j, i]
            ri, rj = find(i), find(j)
            ri != rj && (parent[ri] = rj)
        end
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, find(i), Int[]), i)
    end
    comps = collect(values(groups))
    sort!(comps; by = first)
    return comps
end

"""
    cluster_components(T, eigs; θ, ε_target = Inf) -> Vector{Vector{Int}}

Cluster the indices `1:n` into blocks: connected components of `coupling_graph`.
Each component is a sorted index-set, suitable as the `select` mask for
`ordschur` when realizing the cluster as a contiguous leading block (§5).

Uses a complex Schur factor (`T` upper triangular), so there are no 2×2
conjugate-pair blocks to force atomic; for a real (quasi-triangular) Schur form,
add conjugate-pair edges before taking components.
"""
function cluster_components(T::AbstractMatrix, eigs::AbstractVector; θ::Real,
        ε_target::Real = Inf)
    return _connected_components(coupling_graph(T, eigs; θ, ε_target))
end

cluster_components(S::CertSetup; θ::Real, ε_target::Real = Inf) =
    cluster_components(S.T, S.eigs; θ, ε_target)

"""
    percolation_scan(T, eigs; θs, ε_target = Inf) -> Vector{NamedTuple}

Diagnostic for choosing `θ` (implementation.md §4b, percolation caveat). For each
`θ` in `θs` returns `(; θ, ncomponents, maxsize)`. As `θ` decreases the graph
connects more; pick `θ` just above the giant-component transition, where `maxsize`
jumps. Pass `θs` in decreasing order to read the transition directly.
"""
function percolation_scan(T::AbstractMatrix, eigs::AbstractVector; θs,
        ε_target::Real = Inf)
    return [let comps = cluster_components(T, eigs; θ, ε_target)
                (; θ, ncomponents = length(comps), maxsize = maximum(length, comps))
            end
            for θ in θs]
end
