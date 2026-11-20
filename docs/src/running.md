# Running and testing

## Running a tutorial

From the root of a clone of the repository, after `julia --project=. -e 'using Pkg; Pkg.instantiate()'`:

```
julia -t 4 --project=. src/E02_Hyperelastic_stretch.jl
```

Every script is self-contained. Results (VTK files, CSV tables) are written to `data/sims/<tutorial>/`, which is
not versioned; open the `.pvd` collections with [ParaView](https://www.paraview.org/). Several tutorials assemble
or solve with threads: start Julia with `-t 4` or `-t auto`.

From the Julia REPL the same can be done with

```julia
using HyperFEMTutorials
run_tutorial(2)              # or run_tutorial("E02")
run_tutorial(2; fast=true)   # reduced case
```

## Reduced cases

With the environment variable `HYPERFEM_TUTORIALS_FAST=1` every tutorial runs a reduced case (coarser mesh, fewer
steps) that finishes in a few minutes at most. The test suite uses them.

## Notebooks

Each tutorial page links to a Jupyter notebook generated from the same script. To open them install
[IJulia](https://github.com/JuliaLang/IJulia.jl) and generate the notebooks inside your clone, so that the relative
paths to `data/` work:

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl notebooks     # writes notebooks/E<NN>_<Name>.ipynb
```

The notebooks are not executed: run the cells in order. Select a Julia kernel with threads (for example created
with `IJulia.installkernel("Julia 4 threads", env=Dict("JULIA_NUM_THREADS" => "4"))`) and activate the project of
the repository in the first cell (`using Pkg; Pkg.activate("..")`).

## Tests

```
julia -t 4 --project=. -e 'using Pkg; Pkg.test()'
```

runs every tutorial in its reduced case in a fresh module; the checks of each tutorial (its `@assert`s) are the
tests. Environment variables: `HYPERFEM_TUTORIALS_GROUP` (`basics`, `coupled`, `contact`, `fluids`, `solvers`,
`dynamics`), `HYPERFEM_TUTORIALS_ONLY` (comma-separated prefixes, e.g. `E02,E10`), `HYPERFEM_TUTORIALS_FAST=0`
(the documented cases) and `HYPERFEM_TUTORIALS_PETSC=0` (skip the parts that use PETSc).

## PETSc

The solver tutorials use PETSc through GridapPETSc. The project pins `PETSc_jll` to 3.22.1, the version these tutorials are
tested with (newer builds failed to load on macOS with GridapPETSc 0.5.8); with `HYPERFEM_TUTORIALS_PETSC=0` those
parts are skipped.
