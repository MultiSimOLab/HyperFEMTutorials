# Runs every tutorial in its reduced (FAST) form, each in a fresh module, in one process (the compilation of
# Gridap and HyperFEM is shared). The checks of each tutorial are its `@assert`s.
#   HYPERFEM_TUTORIALS_GROUP = basics | coupled | contact | fluids | solvers | dynamics | all (default)
#   HYPERFEM_TUTORIALS_ONLY  = comma-separated file prefixes, e.g. "E02,E10"
#   HYPERFEM_TUTORIALS_FAST  = 0 runs the documented (full) cases instead
#   HYPERFEM_TUTORIALS_PETSC = 0 skips the PETSc parts (read by the tutorials that use it)
using Test
using HyperFEMTutorials
using Printf

group = Symbol(get(ENV, "HYPERFEM_TUTORIALS_GROUP", "all"))
only_ = filter(!isempty, split(get(ENV, "HYPERFEM_TUTORIALS_ONLY", ""), ','))
fast = get(ENV, "HYPERFEM_TUTORIALS_FAST", "1") == "1"

selected = filter(TUTORIALS) do t
  (group === :all || t.group === group) && (isempty(only_) || any(p -> startswith(t.file, p), only_))
end

# The list of tutorials, the scripts in src/, the README and the documentation index must agree
@testset "list of tutorials" begin
  srcdir = dirname(pathof(HyperFEMTutorials))
  scripts = sort(filter(f -> occursin(r"^E\d\d_.*\.jl$", f), readdir(srcdir)))
  @test sort([t.file for t in TUTORIALS]) == scripts
  @test allunique(t.number for t in TUTORIALS) && allunique(t.file for t in TUTORIALS)
  @test all(t -> startswith(t.file, "E" * lpad(t.number, 2, '0') * "_"), TUTORIALS)
  @test all(t -> t.group in (:basics, :coupled, :contact, :fluids, :solvers, :dynamics), TUTORIALS)
  readme = read(joinpath(srcdir, "..", "README.md"), String)
  @test all(t -> occursin("src/$(t.file)", readme), TUTORIALS)
  index = read(joinpath(srcdir, "..", "docs", "src", "index.md"), String)
  @test all(t -> occursin("tutorials/$(splitext(t.file)[1]).md", index), TUTORIALS)
end

times = Tuple{String,Float64,Bool}[]
isempty(selected) && error("no tutorial selected (HYPERFEM_TUTORIALS_GROUP = $group, HYPERFEM_TUTORIALS_ONLY = $(join(only_, ',')))")
@testset "HyperFEMTutorials" begin
  for t in selected
    file = joinpath(dirname(pathof(HyperFEMTutorials)), t.file)
    if !isfile(file)
      @warn "missing tutorial $(t.file)"
      @test_broken isfile(file)
      continue
    end
    log = tempname() * ".log"
    ok = false
    t0 = time()
    @testset "$(t.file)" begin
      ok = try
        open(log, "w") do io
          redirect_stdout(io) do
            redirect_stderr(io) do
              run_tutorial(t.number; fast)
            end
          end
        end
        true
      catch err
        println("---- $(t.file) failed; last lines of its output:")
        foreach(println, last(readlines(log), 40))
        showerror(stdout, err, catch_backtrace())
        println()
        false
      end
      @test ok
    end
    push!(times, (t.file, time() - t0, ok))
    GC.gc()
  end
end

println("\nTutorial times ($(fast ? "FAST" : "full"), $(Threads.nthreads()) threads):")
for (f, s, ok) in times
  @printf("  %-40s %8.1f s  %s\n", f, s, ok ? "ok" : "FAILED")
end
