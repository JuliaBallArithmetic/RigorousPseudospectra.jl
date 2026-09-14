# Merge the per-process CSVs written by `bench/pseudospectra_run.jl` and print the tables of
# `~/Write/KrylovPseudospectra/20260912_pseudospectra_runs.md`.
#
#   julia --project=. bench/pseudospectra_tables.jl
#
# Reads ~/ResearchData/KrylovPseudospectra_Data/pseudospectra/<name>.<tag>.csv, writes the merged
# <name>.csv beside them and prints Markdown to stdout. No package code is used: this is plain
# CSV manipulation.

using Printf

const OUT = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "pseudospectra")

struct Table
    header::Vector{String}
    rows::Vector{Vector{String}}
end

function readcsv(path)
    ls = readlines(path)
    isempty(ls) && return Table(String[], Vector{String}[])
    return Table(split(ls[1], ','), [split(l, ',') for l in ls[2:end] if !isempty(l)])
end

"Merge every `<name>.<tag>.csv` into `<name>.csv`."
function merge_all()
    names = Set{String}()
    for f in readdir(OUT)
        m = match(r"^(\w+)\.\w+\.csv$", f)
        m === nothing || push!(names, m.captures[1])
    end
    out = Dict{String, Table}()
    for nm in sort(collect(names))
        hdr = String[]
        rows = Vector{String}[]
        for f in sort(readdir(OUT))
            startswith(f, nm * ".") && endswith(f, ".csv") && count(==('.'), f) == 2 || continue
            t = readcsv(joinpath(OUT, f))
            isempty(t.header) && continue
            isempty(hdr) && (hdr = String.(t.header))
            append!(rows, [String.(r) for r in t.rows])
        end
        out[nm] = Table(hdr, rows)
        open(joinpath(OUT, "$nm.csv"), "w") do io
            println(io, join(hdr, ","))
            for r in rows
                println(io, join(r, ","))
            end
        end
    end
    return out
end

col(t::Table, name) = findfirst(==(name), t.header)
cell(t::Table, r, name) = r[col(t, name)]
cellf(t::Table, r, name) = (v = r[col(t, name)]; v == "NaN" ? NaN : parse(Float64, v))

sig(x, d = 4) = isnan(x) ? "n/a" : (x == 0 ? "0" : @sprintf("%.*g", d, x))

function md(io, header, rows)
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", join(fill("---", length(header)), "|"), "|")
    for r in rows
        println(io, "| ", join(r, " | "), " |")
    end
    println(io)
end

function main()
    T = merge_all()
    io = stdout


    # --- frame and certified numbers ---------------------------------------------------------
    fr = T["frame"]
    println(io, "### Table 1: the frame and the certified numbers\n")
    md(io, ["operator", "n", "target", "count", "R", "R/\\|λ1\\|", "dim Q", "rule", "α", "m",
            "half", "ḡ", "‖G22‖ fp", "ā", "‖A‖ fp", "γ1", "γ2", "γ1γ2"],
        [[cell(fr, r, "op"), cell(fr, r, "n"), cell(fr, r, "target"), cell(fr, r, "count_used"),
          sig(cellf(fr, r, "R"), 6), sig(cellf(fr, r, "R_over_absl1"), 4), cell(fr, r, "dimQ"),
          cell(fr, r, "depth_rule"), sig(cellf(fr, r, "alpha"), 2), cell(fr, r, "order_m"),
          cell(fr, r, "passed_half"), sig(cellf(fr, r, "gbar"), 6), sig(cellf(fr, r, "dense_G22"), 6),
          sig(cellf(fr, r, "abar"), 6), sig(cellf(fr, r, "dense_normA"), 6),
          sig(cellf(fr, r, "gamma1"), 3), sig(cellf(fr, r, "gamma2"), 3),
          sig(cellf(fr, r, "gamma1") * cellf(fr, r, "gamma2"), 3)] for r in fr.rows])

    println(io, "### Table 2: the rounding constants and the cost\n")
    md(io, ["operator", "target", "dim Q", "ε_A", "ε_P", "ε_B", "m", "matvecs (sweep)",
            "t frame (s)", "t sweep (s)", "t compression (s)", "peak RSS (GB)"],
        [[cell(fr, r, "op"), cell(fr, r, "target"), cell(fr, r, "dimQ"),
          sig(cellf(fr, r, "epsA"), 3), sig(cellf(fr, r, "epsP"), 3), sig(cellf(fr, r, "epsB"), 3),
          cell(fr, r, "order_m"), cell(fr, r, "nmatvec_far"), sig(cellf(fr, r, "time_frame"), 3),
          sig(cellf(fr, r, "time_far"), 3), sig(cellf(fr, r, "time_compression"), 3),
          sig(cellf(fr, r, "maxrss_GB"), 3)] for r in fr.rows])

    # --- the circle --------------------------------------------------------------------------
    ci = T["circle"]
    println(io, "### Table 3: the arc cover of |z| = R\n")
    md(io, ["operator", "target", "R", "hypothesis on the whole circle", "region", "floor on Γ",
            "min σ11 (cover)", "min σ11 (samples)", "min σ22", "γ1γ2", "evaluations", "t (s)"],
        [[cell(ci, r, "op"), cell(ci, r, "target"), sig(cellf(ci, r, "R"), 6), cell(ci, r, "hyp_ok"),
          cell(ci, r, "region_ok"), sig(cellf(ci, r, "floor_min"), 4),
          sig(cellf(ci, r, "min_sigma11"), 4), sig(cellf(ci, r, "sigma11_samples_min"), 4),
          sig(cellf(ci, r, "min_sigma22"), 4), sig(cellf(ci, r, "gamma1gamma2"), 3),
          cell(ci, r, "nevals"), sig(cellf(ci, r, "time"), 3)] for r in ci.rows])

    # --- the resolvent grid --------------------------------------------------------------------
    rm = T["resolvent_min"]
    println(io, "### Table 4: the resolvent floor on circles of radius ρR\n")
    md(io, ["operator", "target", "ρ", "points bounded", "min floor", "min fp σ_min(A−zI)",
            "min ratio"],
        [[cell(rm, r, "op"), cell(rm, r, "target"), cell(rm, r, "rho"),
          cell(rm, r, "nok") * "/" * cell(rm, r, "npts"), sig(cellf(rm, r, "min_floor"), 4),
          sig(cellf(rm, r, "min_dense_sigmin"), 4), sig(cellf(rm, r, "min_ratio"), 3)]
         for r in rm.rows])

    # --- the inclusion ---------------------------------------------------------------------------
    inc = T["inclusion"]
    println(io, "### Table 5: prop:complete, δ(ε) and δ(ε)/ε\n")
    ops = unique([(cell(inc, r, "op"), cell(inc, r, "target")) for r in inc.rows])
    es = ["0.1", "0.01", "0.001", "0.0001", "1e-05", "1e-06", "1e-07", "1e-08"]
    rows = Vector{String}[]
    for (o, t) in ops
        rs = [r for r in inc.rows if cell(inc, r, "op") == o && cell(inc, r, "target") == t]
        line = [o, t]
        for e in es
            r = findfirst(r -> abs(cellf(inc, r, "eps_over_normA") - parse(Float64, e)) <
                               1e-3 * parse(Float64, e), rs)
            push!(line, r === nothing ? "" :
                        (cell(inc, rs[r], "ok") == "true" ?
                         sig(cellf(inc, rs[r], "delta_over_eps"), 4) : "no bound"))
        end
        push!(rows, line)
    end
    md(io, vcat(["operator", "target"], ["δ/ε at ε=$(e)‖A‖" for e in es]), rows)

    # --- counts and discs -------------------------------------------------------------------------
    cn = T["counts"]
    println(io, "### Table 6: cor:agree, the exterior count and the isolation discs\n")
    md(io, ["operator", "target", "R", "certified count", "fp count", "match", "cor:agree",
            "discs", "Σ disc counts", "t (s)"],
        [[cell(cn, r, "op"), cell(cn, r, "target"), sig(cellf(cn, r, "R"), 6),
          cell(cn, r, "certified_count"), cell(cn, r, "dense_count"), cell(cn, r, "count_ok"),
          cell(cn, r, "agree_ok"), cell(cn, r, "ndiscs"), cell(cn, r, "sum_disc_counts"),
          sig(cellf(cn, r, "time"), 3)] for r in cn.rows])

    di = T["discs"]
    println(io, "### Table 7: the isolation discs of cor:agree, one row per disc and radius\n")
    println(io, "`min` is the radius `certify_exterior` returns (the smallest in its sequence at",
        " which prop:count passes); `max` is the largest radius in a ten-point sequence up to the",
        " limit prop:count allows (inside `{|z| > R}`, half the distance to the nearest other",
        " component) at which it still passes. `not closed` is a component for which no radius",
        " worked.\n")
    md(io, ["operator", "target", "disc", "kind", "centre", "radius", "count", "fp count",
            "|λ| ∈", "arg λ ∈ (rad)", "fp |λ|", "fp arg λ", "‖P−P₀‖ ≤", "r M² ‖E‖ (eq:projector)",
            "M", "M₀"],
        [[cell(di, r, "op"), cell(di, r, "target"), cell(di, r, "idx"), cell(di, r, "kind"),
          string(sig(cellf(di, r, "re_c"), 8), cellf(di, r, "im_c") < 0 ? " − " : " + ",
              sig(abs(cellf(di, r, "im_c")), 6), "i"),
          sig(cellf(di, r, "radius"), 4), cell(di, r, "count"), cell(di, r, "fp_count"),
          string("[", sig(cellf(di, r, "abs_lo"), 12), ", ", sig(cellf(di, r, "abs_hi"), 12), "]"),
          string("[", sig(cellf(di, r, "arg_lo"), 10), ", ", sig(cellf(di, r, "arg_hi"), 10), "]"),
          sig(cellf(di, r, "fp_abs_min"), 12), sig(cellf(di, r, "fp_arg_min"), 10),
          sig(cellf(di, r, "proj_bound"), 4), sig(cellf(di, r, "proj_paper_M2"), 4),
          sig(cellf(di, r, "proj_M"), 4), sig(cellf(di, r, "proj_M0"), 4)] for r in di.rows])

    # --- projectors -----------------------------------------------------------------------------
    pr = T["projector"]
    println(io, "### Table 8: prop:projector\n")
    rows = Vector{String}[]
    for (o, t) in unique([(cell(pr, r, "op"), cell(pr, r, "target")) for r in pr.rows])
        rs = [r for r in pr.rows if cell(pr, r, "op") == o && cell(pr, r, "target") == t]
        e = rs[findfirst(r -> cell(pr, r, "region") == "exterior", rs)]
        push!(rows, [o, t, cell(pr, e, "ok"), sig(cellf(pr, e, "M"), 4),
            sig(cellf(pr, e, "M0"), 4), sig(cellf(pr, e, "bound"), 4),
            sig(cellf(pr, e, "paper_bound_M2"), 4), sig(cellf(pr, e, "normP"), 4),
            cell(pr, e, "dim"), sig(cellf(pr, e, "subspace_gap"), 3),
            sig(cellf(pr, e, "eE"), 3)])
    end
    md(io, ["operator", "target", "ok", "M", "M₀", "‖P−P₀‖ ≤ r M M₀ ‖E‖",
            "eq:projector r M² ‖E‖", "‖P‖ ≤", "dim", "subspace gap ≤", "‖E‖ ≤ max(γ1,γ2)"], rows)

    # --- reach ------------------------------------------------------------------------------------
    re = T["reach"]
    println(io, "### Table 9: how far inside the certificate reaches (frame fixed)\n")
    md(io, ["operator", "target", "R start", "ḡ", "R min", "R min/|λ1|", "R min/ḡ",
            "count outside at R min", "fp count", "cor:agree at R min", "steps", "t (s)"],
        [[cell(re, r, "op"), cell(re, r, "target"), sig(cellf(re, r, "R_start"), 6),
          sig(cellf(re, r, "gbar"), 6), sig(cellf(re, r, "R_min"), 6),
          sig(cellf(re, r, "R_min_over_absl1"), 4),
          sig(cellf(re, r, "R_min") / cellf(re, r, "gbar"), 5), cell(re, r, "count_at_Rmin"),
          cell(re, r, "dense_count_at_Rmin"), cell(re, r, "agree_ok_at_Rmin"),
          cell(re, r, "nsteps"), sig(cellf(re, r, "time"), 3)] for r in re.rows])

    if haskey(T, "framecost")
        fc = T["framecost"]
        println(io, "### Table 11: the frame alone: matvecs and wall clock\n")
        md(io, ["operator", "n", "target", "R", "dim Q", "depth rule", "matvecs", "depth iterations",
                "t (s)"],
            [[cell(fc, r, "op"), cell(fc, r, "n"), cell(fc, r, "target"),
              sig(cellf(fc, r, "R"), 6), cell(fc, r, "dimQ"), cell(fc, r, "depth_rule"),
              cell(fc, r, "nmatvec_frame"), cell(fc, r, "niter_depth"),
              sig(cellf(fc, r, "time_frame"), 3)] for r in fc.rows])
    end

    if haskey(T, "selfadj")
        sa = T["selfadj"]
        println(io, "### Table 10: cor:selfadj and cor:weyl\n")
        md(io, ["operator", "kind", "idx", "lo", "hi", "count", "fp λ", "contains"],
            [[cell(sa, r, "op"), cell(sa, r, "kind"), cell(sa, r, "idx"),
              sig(cellf(sa, r, "lo"), 8), sig(cellf(sa, r, "hi"), 8), cell(sa, r, "count"),
              sig(cellf(sa, r, "dense_lambda"), 8), cell(sa, r, "contains")] for r in sa.rows])
    end
    return nothing
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
