# Markdown tables from the CSV files written by run_gallery.jl, run_ballsweep.jl, run_threads.jl.
#
#   julia bench/fpsweep/make_tables.jl > tables.md

using Printf

const RES = joinpath(homedir(), "ResearchData", "KrylovPseudospectra_Data", "fpsweep", "results")

function readcsv(file)
    lines = readlines(joinpath(RES, file))
    hdr = split(lines[1], ",")
    rows = Dict{String, String}[]
    for l in lines[2:end]
        f = split(l, ","; limit = length(hdr))
        push!(rows, Dict(String(h) => String(v) for (h, v) in zip(hdr, f)))
    end
    return rows
end
num(r, k) = parse(Float64, r[k])
g3(x) = isnan(x) ? "n/a" : (x == 0 ? "0" : @sprintf("%.3g", x))
g4(x) = isnan(x) ? "n/a" : @sprintf("%.4g", x)
g6(x) = isnan(x) ? "n/a" : @sprintf("%.6g", x)
e2(x) = isnan(x) ? "n/a" : @sprintf("%.1e", x)
tf(x) = @sprintf("%.2f", x)

keyof(r) = (r["name"], parse(Int, r["ks"]))
function grouped(rows)
    keys_ = unique(keyof.(rows))
    return [(k, filter(r -> keyof(r) == k, rows)) for k in keys_]
end
bym(rs, m) = only(filter(r -> r["m"] == string(m), rs))

truG(r) = max(num(r, "normG22_f64"), num(r, "lbG_dd"))
truA(r) = max(num(r, "normA_f64"), num(r, "lbA_dd"))
nanmax(a, b) = isnan(a) ? b : (isnan(b) ? a : max(a, b))
nanmin(a, b) = isnan(a) ? b : (isnan(b) ? a : min(a, b))
truG2(r) = nanmax(num(r, "normG22_f64"), num(r, "lbG_dd"))
truA2(r) = nanmax(num(r, "normA_f64"), num(r, "lbA_dd"))

"κ, √n·p·ε_A·κ and soundness recomputed from the raw columns (NaN-ignoring maximum of the
Float64 and Double64 baselines)."
function derived!(rows)
    for r in rows
        A = truA2(r); G = truG2(r)
        κ = A / G
        p = parse(Int, r["m"]) ÷ 2
        r["kappa"] = string(κ)
        r["scale_est"] = string(sqrt(parse(Float64, r["n"])) * p * num(r, "epsA") * κ)
        ok(x, t) = isnan(x) || x >= t
        r["sound"] = isnan(G) ? "n/a" :
                     string(ok(num(r, "gF"), G) && ok(num(r, "gI"), G) && ok(num(r, "aF"), A) &&
                            ok(num(r, "aI"), A))
    end
    return rows
end
readcsv_d(file) = derived!(readcsv(file))
"Share of ε_B(ā,ḡ) carried by the operator term ε_A ā (1+ε_P) of lem:onestep (F half)."
opshare(r) = num(r, "epsA") * num(r, "aF") * (1 + num(r, "epsP")) / (num(r, "epsB_over_g") * num(r, "gF"))

function table_cert(file; herm = false)
    rows = readcsv_d(file)
    ms = sort(unique(parse.(Int, getindex.(rows, "m"))))
    println("\n| operator | n | w | dim Q | κ | ε_A | ε_P | α | " *
            join(["gap m=$m" for m in ms], " | ") * " | gap interp. | ε_B/ḡ (m=$(ms[end])) | operator share of ε_B (m=$(ms[end])) | √n·p·ε_A·κ (m=$(ms[end])) | sound |")
    println("|" * repeat("---|", 13 + length(ms)))
    for ((name, ks), rs) in grouped(rows)
        r = rs[1]; rl = bym(rs, ms[end])
        κ = num(r, "kappa")
        sd = all(x -> x["sound"] == "true", rs) ? "yes" :
             (any(x -> x["sound"] == "false", rs) ? "**NO**" : "n/a")
        println("| $name | $(r["n"]) | $(r["w"]) | $(r["dimQ"]) | $(g3(κ)) | $(e2(num(r, "epsA"))) | " *
                "$(e2(num(r, "epsP"))) | $(e2(num(r, "alpha"))) | " *
                join([e2(num(bym(rs, m), "gapF")) for m in ms], " | ") *
                " | $(e2(num(r, "gapI"))) | $(e2(num(rl, "epsB_over_g"))) | $(g3(opshare(rl))) | $(e2(num(rl, "scale_est"))) | $sd |")
    end
end

function table_values(file; herm = false)
    rows = readcsv_d(file)
    ms = sort(unique(parse.(Int, getindex.(rows, "m"))))
    extra = herm ? " j (m=$(ms[end])) |" : ""
    println("\n| operator | dim Q | ‖A‖ | ā (m=$(ms[end])) | ‖G22‖ | " *
            join(["ḡ m=$m" for m in ms], " | ") * " | ḡ interp. | ḡ/‖G22‖ (m=$(ms[end])) |$extra sweep bare (s) | sweep + sums (s) | checks (s) |")
    println("|" * repeat("---|", 11 + length(ms) + (herm ? 1 : 0)))
    for ((name, ks), rs) in grouped(rows)
        r = rs[1]; rl = bym(rs, ms[end])
        G = truG2(r); A = truA2(r)
        tch = sum(num(x, "t_checks") for x in rs)
        ex = herm ? " $(rl["j_pm_float"]) |" : ""
        println("| $name | $(r["dimQ"]) | $(g6(A)) | $(g6(num(rl, "a_cert"))) | $(g6(G)) | " *
                join([g6(num(bym(rs, m), "gF")) for m in ms], " | ") *
                " | $(g6(num(r, "gI"))) | $(g4(num(rl, "gF") / G)) |$ex $(tf(num(r, "t_sweep_bare"))) | " *
                "$(tf(num(r, "t_sweep_book"))) | $(tf(tch)) |")
    end
end

function table_kappa(file)
    rows = readcsv_d(file)
    println("\n| ks | dim Q | ‖G22‖ Float64 dense | ‖G22‖ Double64 lower bound | κ | fp trace m=2 | ḡ m=2 | gap m=2 | gap m=8 | ḡ/‖G22‖ m=8 | ε_B/ḡ m=8 | operator share of ε_B m=8 | √n·p·ε_A·κ m=2 | m=8 | passes at 1.01×fp (m=2, m=8) | sound |")
    println("|" * repeat("---|", 16))
    for ((name, ks), rs) in grouped(rows)
        r2 = bym(rs, 2); r8 = bym(rs, 8)
        G = truG2(r2)
        sd = all(x -> x["sound"] == "true", rs) ? "yes" : "**NO**"
        println("| $ks | $(r2["dimQ"]) | $(e2(num(r2, "normG22_f64"))) | $(e2(num(r2, "lbG_dd"))) | " *
                "$(g3(num(r2, "kappa"))) | $(e2(num(r2, "trfp"))) | $(e2(num(r2, "gF"))) | $(e2(num(r2, "gapF"))) | " *
                "$(e2(num(r8, "gapF"))) | $(g4(num(r8, "gF") / G)) | $(e2(num(r8, "epsB_over_g"))) | $(g3(opshare(r8))) | " *
                "$(e2(num(r2, "scale_est"))) | $(e2(num(r8, "scale_est"))) | $(r2["pass_1pct"]), $(r8["pass_1pct"]) | $sd |")
    end
end

function table_ball(file)
    rows = readcsv(file)
    println("\n| operator | n | ks | dim Q | columns | κ | m | Float64 partial sum | ball partial sum | ball/Float64 | t Float64 (s, m≤8) | t ball (s, m≤8) |")
    println("|" * repeat("---|", 12))
    for r in rows
        println("| $(r["name"]) | $(r["n"]) | $(r["ks"]) | $(r["dimQ"]) | $(r["ncols"]) | $(g3(num(r, "kappa"))) | $(r["m"]) | " *
                "$(e2(num(r, "fp_partial"))) | $(e2(num(r, "ball_partial"))) | $(e2(num(r, "ratio_ball_fp"))) | " *
                "$(tf(num(r, "t_fp_book"))) | $(tf(num(r, "t_ball"))) |")
    end
end

function table_radii(file)
    rows = readcsv(file)
    firstcol = Dict{Tuple{String, String}, String}()
    for r in rows
        haskey(firstcol, (r["name"], r["ks"])) || (firstcol[(r["name"], r["ks"])] = r["column"])
    end
    keys_ = unique([(r["name"], r["ks"], r["column"]) for r in rows
                    if r["column"] == firstcol[(r["name"], r["ks"])]])
    println("\n| operator | ks | dim Q | κ | column | ‖rad‖/‖mid‖ at the end of application 1, 2, …, 8 | growth through P⊥, A (or Aᵀ), P⊥: geometric mean over the sub-steps of applications 2 to 8 that end below 1e8 |")
    println("|---|---|---|---|---|---|---|")
    for k in keys_
        rs = filter(r -> (r["name"], r["ks"], r["column"]) == k, rows)
        sort!(rs; by = r -> (parse(Int, r["application"]), parse(Int, r["sub"])))
        v = [num(r, "rad_over_mid") for r in rs]
        ends = v[3:3:end]
        f = [Float64[] for _ in 1:3]
        for q in 4:length(v)             # applications 2..8, sub-steps ending below 1e8
            (v[q - 1] > 0 && v[q] < 1e8) && push!(f[mod1(q, 3)], v[q] / v[q - 1])
        end
        gm(x) = isempty(x) ? NaN : exp(sum(log.(x)) / length(x))
        println("| $(k[1]) | $(k[2]) | $(rs[1]["dimQ"]) | $(g3(num(rs[1], "kappa"))) | $(k[3]) | " *
                join([e2(x) for x in ends], ", ") * " | " * join([g3(gm(x)) for x in f], ", ") * " |")
    end
end

function table_threads(file)
    rows = readcsv(file)
    println("\n| operator | n | dim Q | m | threads | blocksize | sweep bare (s) | sweep + sums (s) | speed-up (bare) |")
    println("|---|---|---|---|---|---|---|---|---|")
    t1 = Dict{Tuple{String, String}, Float64}()
    for r in rows
        r["bookkeeping"] == "false" && r["nthreads"] == "1" && (t1[(r["name"], r["m"])] = num(r, "time"))
    end
    for r in filter(r -> r["bookkeeping"] == "false", rows)
        rb = only(filter(x -> x["bookkeeping"] == "true" && x["nthreads"] == r["nthreads"] &&
                           x["name"] == r["name"] && x["m"] == r["m"], rows))
        base = get(t1, (r["name"], r["m"]), NaN)
        println("| $(r["name"]) | $(r["n"]) | $(r["dimQ"]) | $(r["m"]) | $(r["nthreads"]) | $(r["blocksize"]) | " *
                "$(tf(num(r, "time"))) | $(tf(num(rb, "time"))) | $(g3(base / num(r, "time"))) |")
    end
end

const GALLERY = ["gallery_eigtool.csv", "gallery_nep.csv", "gallery_controls.csv",
    "gallery_hermitian.csv", "gallery_large.csv", "kappa_graded.csv", "kappa_landau.csv",
    "gallery_huge.csv"]

"Measured error / bound from run_constants.jl, one row per operator (maximum over samples)."
function table_constants(file = "constants.csv")
    rows = readcsv(file)
    qs = ["roundA/epsA", "proj/epsP", "onestep/epsB", "column_p1/epsB_Ej", "column_p2/epsB_Ej",
        "column_p3/epsB_Ej", "column_p4/epsB_Ej"]
    println("\n| operator | n | ks | dim Q | ‖fl(Ay)−Ay‖/(ε_A‖A‖‖y‖) | ‖fl(P⊥y)−P⊥y‖/(ε_P‖y‖) | ‖fl(By)−By‖/(ε_B‖y‖) | ‖H^pe_j−w_p‖/(ε_B E_j), p=1 | p=2 | p=3 | p=4 |")
    println("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in unique(getindex.(rows, "name"))
        rs = filter(r -> r["name"] == name, rows)
        v(q) = (x = filter(r -> r["quantity"] == q, rs); isempty(x) ? "n/a" : e2(num(x[1], "max_ratio")))
        println("| $name | $(rs[1]["n"]) | $(rs[1]["ks"]) | $(rs[1]["dimQ"]) | " * join(v.(qs), " | ") * " |")
    end
end

"Soundness summary over every row of every gallery file."
function table_soundness()
    println("\n| file | rows (operator, k, m) | rows with a baseline | rows certified ≥ baseline (ḡ and ā, both halves) | min ḡ/‖G22‖ − 1 | min ā/‖A‖ − 1 | max rel. diff. Float64 vs Double64 ‖G22‖ |")
    println("|---|---|---|---|---|---|---|")
    for f in GALLERY
        isfile(joinpath(RES, f)) || continue
        rows = readcsv_d(f)
        wb = filter(r -> r["sound"] != "n/a", rows)
        ok = count(r -> r["sound"] == "true", wb)
        fin(v) = filter(!isnan, collect(v))
        mg = minimum(fin(nanmin(num(r, "gF"), num(r, "gI")) / truG2(r) - 1 for r in wb); init = Inf)
        ma = minimum(fin(nanmin(num(r, "aF"), num(r, "aI")) / truA2(r) - 1 for r in wb); init = Inf)
        dd = maximum(fin(abs(num(r, "normG22_f64") - num(r, "lbG_dd")) / num(r, "lbG_dd") for r in wb);
            init = -Inf)
        println("| $f | $(length(rows)) | $(length(wb)) | $ok | $(e2(mg)) | $(e2(ma)) | $(e2(dd)) |")
    end
end

"One row per operator: the deepest frame, orders m = 2 and 8."
function table_condensed()
    println("\n| operator | n | w | dim Q | κ | ε_A | ε_P | α | ε_B/ḡ (m=8) | op. share | gap m=2 | gap m=8 | gap interp. | ḡ/‖G22‖ m=2 | ḡ/‖G22‖ m=8 | √n·4·ε_A·κ | sound (all k, m) |")
    println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in GALLERY[1:5]
        isfile(joinpath(RES, f)) || continue
        rows = readcsv_d(f)
        for name in unique(getindex.(rows, "name"))
            rn = filter(r -> r["name"] == name, rows)
            kmax = maximum(parse(Int, r["ks"]) for r in rn)
            rs = filter(r -> parse(Int, r["ks"]) == kmax, rn)
            r2 = bym(rs, 2); r8 = bym(rs, 8)
            G = truG2(r2)
            sd = all(x -> x["sound"] == "true", rn) ? "yes" :
                 (any(x -> x["sound"] == "false", rn) ? "**NO**" : "no baseline")
            println("| $name | $(r2["n"]) | $(r2["w"]) | $(r2["dimQ"]) | $(g3(num(r2, "kappa"))) | " *
                    "$(e2(num(r2, "epsA"))) | $(e2(num(r2, "epsP"))) | $(e2(num(r2, "alpha"))) | " *
                    "$(e2(num(r8, "epsB_over_g"))) | $(g3(opshare(r8))) | $(e2(num(r2, "gapF"))) | " *
                    "$(e2(num(r8, "gapF"))) | $(e2(num(r2, "gapI"))) | $(g4(num(r2, "gF") / G)) | " *
                    "$(g4(num(r8, "gF") / G)) | $(e2(num(r8, "scale_est"))) | $sd |")
        end
    end
end

"Source of every operator, from the `source` column."
function table_sources()
    println("\n| operator | n | nonzeros per row/column (w) | source |")
    println("|---|---|---|---|")
    seen = Set{String}()
    for f in GALLERY
        isfile(joinpath(RES, f)) || continue
        for r in readcsv(f)
            r["name"] in seen && continue
            push!(seen, r["name"])
            println("| $(r["name"]) | $(r["n"]) | $(r["w"]) | $(strip(r["source"], '"')) |")
        end
    end
end

println("\n### Rounding constants against BigFloat")
isfile(joinpath(RES, "constants.csv")) && table_constants()
println("\n### Sources")
table_sources()
println("\n### Soundness summary")
table_soundness()
println("\n### Condensed gallery (deepest frame per operator)")
table_condensed()

"All rows of a thread CSV, in file order."
function table_threads_all(file)
    rows = readcsv(file)
    println("\n| operator | n | dim Q | m | threads | b | running sums | time (s) |")
    println("|---|---|---|---|---|---|---|---|")
    for r in rows
        println("| $(r["name"]) | $(r["n"]) | $(r["dimQ"]) | $(r["m"]) | $(r["nthreads"]) | $(r["blocksize"]) | $(r["bookkeeping"]) | $(tf(num(r, "time"))) |")
    end
end

for (title, f, fn) in [
        ("Non-Hermitian, EigTool demos: certification", "gallery_eigtool.csv", table_cert),
        ("Non-Hermitian, EigTool demos: values and times", "gallery_eigtool.csv", table_values),
        ("Matrix Market NEP: certification", "gallery_nep.csv", table_cert),
        ("Matrix Market NEP: values and times", "gallery_nep.csv", table_values),
        ("Controls: certification", "gallery_controls.csv", table_cert),
        ("Controls: values and times", "gallery_controls.csv", table_values),
        ("Hermitian: certification", "gallery_hermitian.csv", (f) -> table_cert(f; herm = true)),
        ("Hermitian: values and times", "gallery_hermitian.csv", (f) -> table_values(f; herm = true)),
        ("Large operators: certification", "gallery_large.csv", table_cert),
        ("Large operators: values and times", "gallery_large.csv", table_values),
        ("κ scan, graded bidiagonal", "kappa_graded.csv", table_kappa),
        ("κ scan, Landau", "kappa_landau.csv", table_kappa),
        ("Ball sweep", "ballsweep.csv", table_ball),
        ("Ball radii", "ballsweep_radii.csv", table_radii),
        ("Thread scaling", "threads.csv", table_threads),
        ("Thread scaling, repeats (b = 32 and 4)", "threads_repeat.csv", table_threads_all),
        ("Block size b at 32 and 128 threads", "threads_block.csv", table_threads_all)]
    isfile(joinpath(RES, f)) || continue
    println("\n### $title  (`$f`)")
    fn(f)
end
