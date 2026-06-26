module MakieExt

# Makie.jl visualization for RigorousPseudospectra (package extension; loaded only
# when Makie is present). Renders a PseudospectrumEnclosure: excluded cells (proven
# outside σ_ε), included cells (proven inside), and boundary cells (straddling the
# ε-contour), with the approximate eigenvalues overlaid.

using RigorousPseudospectra
using RigorousPseudospectra: Cell, PseudospectrumEnclosure, halfdiag, routing_histogram
using Makie

const _CLASS_COLORS = (excluded = (:gray85, 0.9), included = (:steelblue, 0.55),
    boundary = (:orangered, 0.9))
const _LAYER_COLORS = (triangular = (:seagreen, 0.85), svd = (:mediumpurple, 0.85),
    recentred = (:goldenrod, 0.85))

_rect(cell::Cell) = Makie.Rect2f(real(cell.c) - cell.hx, imag(cell.c) - cell.hy,
    2cell.hx, 2cell.hy)

function _draw_cells!(ax, cells, color)
    isempty(cells) && return
    Makie.poly!(ax, [_rect(c) for c in cells]; color = color, strokewidth = 0)
    return
end

"""
    plot_enclosure(enc; by_layer = false, resolution = (760, 720)) -> Figure

Plot a box-bisection pseudospectrum enclosure. Excluded cells are drawn grey
(or coloured by which certificate settled them when `by_layer = true`), included
cells blue, boundary cells (the ε-contour) red; eigenvalues are red crosses.
"""
function RigorousPseudospectra.plot_enclosure(enc::PseudospectrumEnclosure;
        by_layer::Bool = false, resolution = (760, 720))
    fig = Makie.Figure(size = resolution)
    ax = Makie.Axis(fig[1, 1]; xlabel = "Re z", ylabel = "Im z", aspect = Makie.DataAspect(),
        title = "ε-pseudospectrum enclosure (ε = $(enc.ε))")
    RigorousPseudospectra.plot_enclosure!(ax, enc; by_layer)
    return fig
end

"""
    plot_enclosure!(ax, enc; by_layer = false) -> ax

In-place variant drawing into an existing Makie axis.
"""
function RigorousPseudospectra.plot_enclosure!(ax, enc::PseudospectrumEnclosure;
        by_layer::Bool = false)
    if by_layer && !isempty(enc.excluded)
        for (layer, col) in pairs(_LAYER_COLORS)
            cells = [c for (c, s) in zip(enc.excluded, enc.routing) if s === layer]
            _draw_cells!(ax, cells, col)
        end
    else
        _draw_cells!(ax, enc.excluded, _CLASS_COLORS.excluded)
    end
    _draw_cells!(ax, enc.included, _CLASS_COLORS.included)
    _draw_cells!(ax, enc.boundary, _CLASS_COLORS.boundary)
    if !isempty(enc.eigs)
        Makie.scatter!(ax, real.(enc.eigs), imag.(enc.eigs);
            color = :black, marker = :xcross, markersize = 11)
    end
    return ax
end

end # module
