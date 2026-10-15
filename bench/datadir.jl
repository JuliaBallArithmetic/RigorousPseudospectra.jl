# Where the experiments of the numerics write, and where a script that reads an earlier experiment's
# output (certify_feasible.jl, paper_map_bz.jl, draw_bz_map.jl) reads it:
# ~/ResearchData/KrylovPseudospectra_Data/pseudospectra, or the directory in the environment variable
# PSOUT. With PSOUT set, the figures are copied to PSOUT/figures instead of the paper's figures/.
const PSOUT = get(ENV, "PSOUT",
    joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra"))
const PAPERFIGS = haskey(ENV, "PSOUT") ? joinpath(PSOUT, "figures") :
                  joinpath(homedir(), "Write", "KrylovPseudospectra", "figures")
