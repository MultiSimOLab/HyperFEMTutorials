"""
    HyperFEMTutorials

Tutorials of [HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl). Each tutorial is a self-contained
script `src/E<NN>_<Name>.jl` that can be run on its own (`julia -t 4 --project=. src/E02_Hyperelastic_stretch.jl`).
This module only lists them, for the test suite and the documentation, and runs them on demand.
"""
module HyperFEMTutorials

export TUTORIALS, tutorial_path, run_tutorial

"""
    TUTORIALS

One entry per tutorial: `number`, script `file`, `title` and `group` (`:basics`, `:coupled`, `:contact`,
`:fluids`, `:solvers`, `:dynamics`). The test suite groups them by `group`; the documentation uses the same order.
"""
const TUTORIALS = [
  (number=1,  file="E01_Poisson.jl",                     title="Poisson equation",                                   group=:basics),
  (number=2,  file="E02_Hyperelastic_stretch.jl",        title="Uniaxial stretch of a hyperelastic bar",             group=:basics),
  (number=3,  file="E03_Fibre_reinforced_cylinder.jl",   title="Fibre-reinforced cylinder under internal pressure",  group=:basics),
  (number=4,  file="E04_ElectroMechanics.jl",            title="Electro-mechanics of a dielectric elastomer",        group=:coupled),
  (number=5,  file="E05_ElectroMechanics_anisotropic.jl", title="Anisotropic electro-mechanics",                     group=:coupled),
  (number=6,  file="E06_ThirdMedium_contact.jl",         title="Third-medium contact",                               group=:contact),
  (number=8,  file="E08_MagnetoMechanical_beam.jl",      title="Magneto-mechanical hMRE beam",                       group=:coupled),
  (number=9,  file="E09_hMRE_vacuum_staggered.jl",       title="hMRE beam in vacuum: staggered solution",            group=:coupled),
  (number=10, file="E10_Stokes.jl",                      title="Stokes flow around a cylinder",                      group=:fluids),
  (number=11, file="E11_NavierStokes_transient.jl",      title="Transient Navier–Stokes flow around a cylinder",     group=:fluids),
  (number=12, file="E12_Aorta_blood_flow.jl",            title="Pulsatile blood flow in a patient aorta",            group=:fluids),
  (number=13, file="E13_Solvers.jl",                     title="Linear solvers for large problems",                  group=:solvers),
  (number=14, file="E14_Multigrid.jl",                   title="Geometric h- and p-multigrid",                       group=:solvers),
  (number=15, file="E15_Elastodynamics.jl",              title="Elastodynamics: free vibration of a cantilever",     group=:dynamics),
  (number=16, file="E16_ViscoElectroMechanics.jl",       title="Visco-electro-mechanics: creep of a dielectric bimorph", group=:dynamics),
  (number=17, file="E17_ThermoMechanics.jl",             title="Thermo-mechanics: transient heat conduction and thermal expansion", group=:coupled),
  (number=18, file="E18_Explicit_wave_propagation.jl",   title="Explicit dynamics: wave propagation in a bar",        group=:dynamics),
  (number=19, file="E19_Constitutive_models.jl",         title="Constitutive models: energies, stresses and tangents", group=:basics),
]

"Absolute path of the script of tutorial `n` (its number) or of the tutorial whose file name starts with `n`."
function tutorial_path(n)
  t = n isa Integer ? only(filter(t -> t.number == n, TUTORIALS)) : only(filter(t -> startswith(t.file, n), TUTORIALS))
  joinpath(@__DIR__, t.file)
end

"""
    run_tutorial(n; fast=false)

Run tutorial `n` (number or file-name prefix such as `"E02"`) in a fresh module. With `fast = true` the reduced
case of the test suite is run (environment variable `HYPERFEM_TUTORIALS_FAST=1`). Returns the module, whose
global variables hold the results of the script.
"""
function run_tutorial(n; fast::Bool=false)
  file = tutorial_path(n)
  m = Module(Symbol(splitext(basename(file))[1]))
  withenv("HYPERFEM_TUTORIALS_FAST" => (fast ? "1" : "0")) do
    Base.include(m, file)
  end
  m
end

end # module
