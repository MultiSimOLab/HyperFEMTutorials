# HyperFEM tutorials

Tutorials of [HyperFEM.jl](https://github.com/MultiSimOLab/HyperFEM.jl), a library of the
[Gridap](https://github.com/gridap/Gridap.jl) ecosystem for the finite-element simulation of multiphysics problems with
hyperelastic materials: thermo-, electro- and magneto-mechanics at finite strains, contact, fluids, transient
problems and solvers for large systems.

Every tutorial is a self-contained Julia script in the [`src/`](https://github.com/MultiSimOLab/HyperFEMTutorials/tree/main/src)
folder of the repository. Its page explains the problem, the equations and every step of the code, and links to a
Jupyter notebook generated from the same script. Each script ends with quantitative checks of its result (exact
solutions, published benchmarks, conservation laws), which are also the test suite of the repository.

## Getting started

```
git clone https://github.com/MultiSimOLab/HyperFEMTutorials.git
cd HyperFEMTutorials
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t 4 --project=. src/E02_Hyperelastic_stretch.jl
```

The tutorials need Julia 1.12 or later and use HyperFEM 0.3, HyperFEMKernels 0.2.6 and Gridap 0.20.8. See
[Running and testing](running.md) for reduced cases, notebooks and tests.

## Tutorials

| Nº | Tutorial | What it shows |
|:--:|:--|:--|
| 01 | [Poisson equation](tutorials/E01_Poisson.md) | linear problems, Dirichlet and Neumann conditions, convergence under refinement |
| 02 | [Uniaxial stretch of a hyperelastic bar](tutorials/E02_Hyperelastic_stretch.md) | constitutive models, load stepping, Newton–Raphson, post-processing |
| 03 | [Fibre-reinforced cylinder under internal pressure](tutorials/E03_Fibre_reinforced_cylinder.md) | anisotropic models, fibre fields, follower pressure |
| 04 | [Electro-mechanics of a dielectric elastomer](tutorials/E04_ElectroMechanics.md) | multi-field problems, coupled weak forms |
| 05 | [Anisotropic electro-mechanics](tutorials/E05_ElectroMechanics_anisotropic.md) | composition of models, extra arguments of the constitutive functions |
| 06 | [Third-medium contact](tutorials/E06_ThirdMedium_contact.md) | contact through a soft medium, injectivity line search, hybrid Newton |
| 07 | Topology optimisation | in preparation (HyperShape) |
| 08 | [Magneto-mechanical hMRE beam](tutorials/E08_MagnetoMechanical_beam.md) | hard-magnetic soft materials, lagged loads |
| 09 | [hMRE beam in vacuum: staggered solution](tutorials/E09_hMRE_vacuum_staggered.md) | staggered models, coupling between meshes, transfers |
| 10 | [Stokes flow around a cylinder](tutorials/E10_Stokes.md) | mixed and stabilised formulations |
| 11 | [Transient Navier–Stokes flow around a cylinder](tutorials/E11_NavierStokes_transient.md) | time integration with `TransientModel`, benchmarks |
| 12 | [Pulsatile blood flow in a patient aorta](tutorials/E12_Aorta_blood_flow.md) | 3D transient flow, Windkessel outlet conditions |
| 13 | [Linear solvers for large problems](tutorials/E13_Solvers.md) | direct, Krylov and algebraic multigrid solvers, inexact Newton |
| 14 | [Geometric h- and p-multigrid](tutorials/E14_Multigrid.md) | mesh hierarchies, matrix-free p-multigrid |
| 15 | [Elastodynamics: free vibration of a cantilever](tutorials/E15_Elastodynamics.md) | structural dynamics, energy conservation |
| 16 | [Visco-electro-mechanics: creep of a dielectric bimorph](tutorials/E16_ViscoElectroMechanics.md) | generalised Maxwell viscoelasticity, internal variables, quasi-static time stepping |
| 17 | [Thermo-mechanics: transient heat conduction and thermal expansion](tutorials/E17_ThermoMechanics.md) | coupled thermal and mechanical fields, weak forms from the energy, adaptive BDF2 |
| 18 | [Explicit dynamics: wave propagation in a bar](tutorials/E18_Explicit_wave_propagation.md) | central differences, lumped mass, critical time step, wave speed |
| 19 | [Constitutive models: energies, stresses and tangents](tutorials/E19_Constitutive_models.md) | the constitutive-model API, homogeneous tests, writing your own model |

## How to cite

If you use HyperFEM, please cite

C. Perez-Garcia, R. Ortigosa, J. Martínez-Frutos and D. Garcia-Gonzalez, *Topology and material optimization in
ultra-soft magnetoactive structures: making advantage of residual anisotropies*, Adv. Mater. (2025) e18489,
<https://doi.org/10.1002/adma.202518489>,

together with the citations requested by [Gridap](https://github.com/gridap/Gridap.jl).
