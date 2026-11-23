<p align="center"><img src="https://github.com/MultiSimOLab/HyperFEM.jl/raw/main/docs/img/logo.png" width="20%"></p>

# HyperFEM tutorials

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://multisimolab.github.io/HyperFEMTutorials.jl/dev/)
[![CI](https://github.com/MultiSimOLab/HyperFEMTutorials.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MultiSimOLab/HyperFEMTutorials.jl/actions/workflows/CI.yml?query=branch%3Amain)

Tutorials of [HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl), a library of the
[Gridap](https://github.com/gridap/Gridap.jl) ecosystem for the finite-element simulation of multiphysics problems with
hyperelastic materials: thermo-, electro- and magneto-mechanics at finite strains, contact, fluids, transient problems
and solvers for large systems.

Every tutorial is a self-contained Julia script in [`src/`](src/). The
[documentation](https://multisimolab.github.io/HyperFEMTutorials.jl/dev/) turns each one into a page that explains the
problem, the equations and every step of the code, with a Jupyter notebook generated from the same script. Each script
ends with quantitative checks of its result (exact solutions, published benchmarks, conservation laws), which are also
the test suite of this repository.

## Getting started

Julia 1.12 or later is needed. Clone the repository and install the dependencies (HyperFEM 0.3, HyperFEMKernels 0.2.6, Gridap 0.20.8, …):

```
git clone https://github.com/MultiSimOLab/HyperFEMTutorials.jl.git
cd HyperFEMTutorials.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run a tutorial from the root of the repository (several tutorials use threads):

```
julia -t 4 --project=. src/E02_Hyperelastic_stretch.jl
```

Results (VTK files for [ParaView](https://www.paraview.org/), CSV tables) are written to `data/sims/<tutorial>/`.
With `HYPERFEM_TUTORIALS_FAST=1` every tutorial runs a reduced case that finishes in a few minutes at most. When several
Julia processes run at the same time, limit the BLAS threads (`OPENBLAS_NUM_THREADS=4`): the sparse LU factorisations
otherwise compete for the cores and become many times slower.

## Tutorials

| Nº | Tutorial | What it shows |
|:--:|:--|:--|
| 01 | [Poisson equation](src/E01_Poisson.jl) | linear problems, Dirichlet and Neumann conditions, convergence under refinement |
| 02 | [Uniaxial stretch of a hyperelastic bar](src/E02_Hyperelastic_stretch.jl) | constitutive models, load stepping, Newton–Raphson, post-processing |
| 03 | [Fibre-reinforced cylinder under internal pressure](src/E03_Fibre_reinforced_cylinder.jl) | anisotropic models, fibre fields, follower pressure |
| 04 | [Electro-mechanics of a dielectric elastomer](src/E04_ElectroMechanics.jl) | multi-field problems, coupled weak forms |
| 05 | [Anisotropic electro-mechanics](src/E05_ElectroMechanics_anisotropic.jl) | composition of models, extra arguments of the constitutive functions |
| 06 | [Third-medium contact](src/E06_ThirdMedium_contact.jl) | contact through a soft medium, injectivity line search, hybrid Newton |
| 07 | Topology optimisation | in preparation ([HyperShape.jl](https://github.com/MultiSimOLab/HyperShape.jl)) |
| 08 | [Magneto-mechanical hMRE beam](src/E08_MagnetoMechanical_beam.jl) | hard-magnetic soft materials, lagged loads |
| 09 | [hMRE beam in vacuum: staggered solution](src/E09_hMRE_vacuum_staggered.jl) | staggered models, coupling between meshes, transfers |
| 10 | [Stokes flow around a cylinder](src/E10_Stokes.jl) | mixed and stabilised formulations |
| 11 | [Transient Navier–Stokes flow around a cylinder](src/E11_NavierStokes_transient.jl) | time integration with `TransientModel`, benchmarks |
| 12 | [Pulsatile blood flow in a patient aorta](src/E12_Aorta_blood_flow.jl) | 3D transient flow, Windkessel outlet conditions |
| 13 | [Linear solvers for large problems](src/E13_Solvers.jl) | direct, Krylov and algebraic multigrid solvers, inexact Newton |
| 14 | [Geometric h- and p-multigrid](src/E14_Multigrid.jl) | mesh hierarchies, matrix-free p-multigrid |
| 15 | [Elastodynamics: free vibration of a cantilever](src/E15_Elastodynamics.jl) | structural dynamics, energy conservation |
| 16 | [Visco-electro-mechanics: creep of a dielectric bimorph](src/E16_ViscoElectroMechanics.jl) | generalised Maxwell viscoelasticity, internal variables, quasi-static time stepping |
| 17 | [Thermo-mechanics: transient heat conduction and thermal expansion](src/E17_ThermoMechanics.jl) | coupled thermal and mechanical fields, weak forms from the energy, adaptive BDF2 |
| 18 | [Explicit dynamics: wave propagation in a bar](src/E18_Explicit_wave_propagation.jl) | central differences, lumped mass, critical time step, wave speed |
| 19 | [Constitutive models: energies, stresses and tangents](src/E19_Constitutive_models.jl) | the constitutive-model API, homogeneous tests, writing your own model |

## Notebooks

Every page of the documentation links to a Jupyter notebook (not executed) generated from the same script with
[Literate.jl](https://github.com/fredrikekre/Literate.jl). To use them, install
[IJulia](https://github.com/JuliaLang/IJulia.jl) and generate the notebooks inside your clone, so that the paths to
`data/` work:

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl notebooks     # writes notebooks/E<NN>_<Name>.ipynb
```

## Tests

```
julia -t 4 --project=. -e 'using Pkg; Pkg.test()'
```

runs every tutorial in its reduced case. `HYPERFEM_TUTORIALS_GROUP` (`basics`, `coupled`, `contact`, `fluids`,
`solvers`, `dynamics`) and `HYPERFEM_TUTORIALS_ONLY` (e.g. `E02,E10`) select tutorials; `HYPERFEM_TUTORIALS_PETSC=0`
skips the parts that use PETSc. See [Running and testing](https://multisimolab.github.io/HyperFEMTutorials.jl/dev/running/).

## How to cite

If you use HyperFEM, please cite

C. Perez-Garcia, R. Ortigosa, J. Martínez-Frutos and D. Garcia-Gonzalez, **Topology and material optimization in
ultra-soft magnetoactive structures: making advantage of residual anisotropies.** Adv. Mater. (2025) e18489.
https://doi.org/10.1002/adma.202518489

along with the citations requested by [Gridap](https://github.com/gridap/Gridap.jl) (see [CITATION.cff](CITATION.cff)).

## Funding

Grants PID2022-141957OA-C22/PID2022-141957OB-C22 funded by MCIN/AEI/10.13039/501100011033 and by "ERDF A way of making
Europe".

<p align="center"><img src="https://github.com/MultiSimOLab/HyperFEM.jl/raw/main/docs/img/aei.png" width="70%"></p>

## Contact

Contact the project administrator [Jesús Martínez-Frutos](mailto:jesus.martinez@upct.es) for questions about licenses and
terms of use.
