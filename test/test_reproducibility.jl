# Invariants a published dataset depends on. Each corresponds to a real failure
# that cost real time; all are cheap enough to run on every commit.
using Test

const ROOT = dirname(@__DIR__)

"Dependencies pinned to a local working copy, which no clone can reconstruct."
function dev_path_deps(manifest::AbstractString)
    offenders, current = String[], ""
    for line in eachline(manifest)
        m = match(r"^\[\[deps\.(.+)\]\]$", strip(line))
        m !== nothing && (current = m.captures[1])
        pm = match(r"^\s*path\s*=\s*\"(.*)\"\s*$", line)
        # path = "." is the project's own entry, which Julia always records.
        # Only a path pointing somewhere ELSE is a dev dependency.
        if pm !== nothing && pm.captures[1] != "."
            push!(offenders, "$current => $(pm.captures[1])")
        end
    end
    return offenders
end

@testset "reproducibility" begin

    # A `path = "../Foo.jl"` entry pins a dependency to a working copy nobody
    # else has, so the environment behind a published result cannot be rebuilt
    # from the repository -- and it silently stops matching the version it
    # claims as soon as that directory moves on.
    @testset "Manifest pins no dev paths" begin
        manifest = joinpath(ROOT, "Manifest.toml")
        @test isfile(manifest)
        if isfile(manifest)
            offenders = dev_path_deps(manifest)
            isempty(offenders) || @info "dev-path dependencies" offenders
            @test isempty(offenders)
        end
    end

    # The environment is only pinned if it is actually committed.
    @testset "Manifest is tracked by git" begin
        tracked = try
            readchomp(`git -C $ROOT ls-files Manifest.toml`)
        catch
            "Manifest.toml"   # no git here: do not fail the suite over it
        end
        @test !isempty(tracked)
    end

    # Data does not live in the code repository. A committed binary lets the
    # code quietly stop being able to regenerate it.
    @testset "no data committed under data/" begin
        tracked = try
            split(readchomp(`git -C $ROOT ls-files data/`), '\n'; keepempty=false)
        catch
            String[]
        end
        bad = filter(f -> any(endswith(f, e) for e in
                     (".jls", ".jld2", ".h5", ".hdf5", ".arrow", ".npy", ".mat")), tracked)
        isempty(bad) || @info "data committed to the code repo" bad
        @test isempty(bad)
    end

    # PIPELINE must not rot: every script it names has to exist.
    @testset "PIPELINE names existing scripts" begin
        p = joinpath(ROOT, "PIPELINE")
        @test isfile(p)
        if isfile(p)
            gone = String[]
            for line in eachline(p)
                s = strip(line)
                (isempty(s) || startswith(s, "#")) && continue
                script = strip(first(split(s, ':')))
                isfile(joinpath(ROOT, script)) || push!(gone, script)
            end
            isempty(gone) || @info "PIPELINE lists missing scripts" gone
            @test isempty(gone)
        end
    end
end
