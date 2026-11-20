# HyperFEM tutorials

Tutorials of [HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl), a library of the
[Gridap](https://github.com/gridap/Gridap.jl) ecosystem for the finite-element simulation of multiphysics problems with
hyperelastic materials: thermo-, electro- and magneto-mechanics at finite strains, contact, fluids, transient
problems and solvers for large systems.

Every tutorial is a self-contained Julia script in the [`src/`](https://github.com/MultiSimOLab/HyperFEMTutorials.jl/tree/main/src)
folder of the repository. Its page explains the problem, the equations and every step of the code, and links to a
Jupyter notebook generated from the same script. Each script ends with quantitative checks of its result (exact
solutions, published benchmarks, conservation laws), which are also the test suite of the repository.

## Getting started

```
git clone https://github.com/MultiSimOLab/HyperFEMTutorials.jl.git
cd HyperFEMTutorials.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t 4 --project=. src/E02_Hyperelastic_stretch.jl
```

The tutorials need Julia 1.12 or later and use HyperFEM 0.3 and Gridap 0.20.8. See
[Running and testing](running.md) for reduced cases, notebooks and tests.

## Tutorials

| Nº | Tutorial | What it shows |
|:--:|:--|:--|
| 01 | Poisson equation | linear problems, Dirichlet and Neumann conditions, convergence under refinement |
| 02 | Uniaxial stretch of a hyperelastic bar | constitutive models, load stepping, Newton–Raphson, post-processing |
| 03 | Fibre-reinforced cylinder under internal pressure | anisotropic models, fibre fields, follower pressure |
| 04 | Electro-mechanics of a dielectric elastomer | multi-field problems, coupled weak forms |
| 05 | Anisotropic electro-mechanics | composition of models, extra arguments of the constitutive functions |
| 06 | Third-medium contact | contact through a soft medium, injectivity line search, hybrid Newton |
| 07 | Topology optimisation | in preparation (HyperShape) |
| 08 | Magneto-mechanical hMRE beam | hard-magnetic soft materials, lagged loads |
| 09 | hMRE beam in vacuum: staggered solution | staggered models, coupling between meshes, transfers |
| 10 | Stokes flow around a cylinder | mixed and stabilised formulations |
| 11 | Transient Navier–Stokes flow around a cylinder | time integration with `TransientModel`, benchmarks |
| 12 | Pulsatile blood flow in a patient aorta | 3D transient flow, Windkessel outlet conditions |
| 13 | Linear solvers for large problems | direct, Krylov and algebraic multigrid solvers, inexact Newton |
| 14 | Geometric h- and p-multigrid | mesh hierarchies, matrix-free p-multigrid |
| 15 | Elastodynamics: free vibration of a cantilever | structural dynamics, energy conservation |

## How to cite

If you use HyperFEM, please cite

C. Perez-Garcia, R. Ortigosa, J. Martínez-Frutos and D. Garcia-Gonzalez, *Topology and material optimization in
ultra-soft magnetoactive structures: making advantage of residual anisotropies*, Adv. Mater. (2025) e18489,
<https://doi.org/10.1002/adma.202518489>,

together with the citations requested by [Gridap](https://github.com/gridap/Gridap.jl).
