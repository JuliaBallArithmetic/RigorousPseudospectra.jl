module PlotsExt

# Plots.jl visualization for RigorousPseudospectra (package extension; loaded only
# when Plots is present). Same content as MakieExt — renders a
# PseudospectrumEnclosure with cells coloured by class and eigenvalues overlaid.

using RigorousPseudospectra
using RigorousPseudospectra: Cell, PseudospectrumEnclosure
using Plots

function _shape(cell::Cell)
    x0, y0 = real(cell.c) - cell.hx, imag(cell.c) - cell.hy
    x1, y1 = real(cell.c) + cell.hx, imag(cell.c) + cell.hy
    return Plots.Shape([x0, x1, x1, x0], [y0, y0, y1, y1])
end

"""
    plot_enclosure(enc; size = (760, 720)) -> Plots.Plot

Plot a box-bisection pseudospectrum enclosure: excluded cells grey, included blue,
boundary (ε-contour) red, eigenvalues as black crosses.
"""
function RigorousPseudospectra.plot_enclosure(enc::PseudospectrumEnclosure;
        size = (760, 720))
    plt = Plots.plot(; aspect_ratio = :equal, legend = false, size = size,
        xlabel = "Re z", ylabel = "Im z",
        title = "ε-pseudospectrum enclosure (ε = $(enc.ε))")
    return RigorousPseudospectra.plot_enclosure!(plt, enc)
end

function RigorousPseudospectra.plot_enclosure!(plt, enc::PseudospectrumEnclosure)
    for (cells, col) in ((enc.excluded, :gray85), (enc.included, :steelblue),
        (enc.boundary, :orangered))
        isempty(cells) && continue
        Plots.plot!(plt, [_shape(c) for c in cells]; color = col, linewidth = 0,
            fillalpha = 0.8)
    end
    if !isempty(enc.eigs)
        Plots.scatter!(plt, real.(enc.eigs), imag.(enc.eigs);
            color = :black, marker = :x, markersize = 5)
    end
    return plt
end

end # module
