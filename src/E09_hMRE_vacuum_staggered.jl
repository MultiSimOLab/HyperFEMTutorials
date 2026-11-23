# # Tutorial 09: hMRE beam in vacuum: staggered solution
#
# A thin beam of hard-magnetic soft elastomer (hMRE: an ultra-soft silicone filled with magnetised particles)
# bends when an external magnetic field is applied across its remanent magnetisation. Here the beam is
# surrounded by an explicit layer of vacuum (air) that carries the magnetic field and deforms with the beam,
# so that the field, the Maxwell stress on the beam and the beam itself are computed together. The coupled
# problem is split into three problems solved one after another at every load step (a *staggered*, or
# partitioned, solution): this is how the simulations of Perez-Garcia et al. (Adv. Mater. 2025) were run.
#
# **What you will learn**
# - magneto-mechanical models: `HardMagnetic2D` inside a `MagnetoMechModel`, `IdealMagnetic2D` for the vacuum,
#   and a regularised fictitious material (`HessianRegularization`) for the motion of the vacuum mesh;
# - finite element spaces on parts of a mesh (solid, vacuum, their interface) and a field on an interface;
# - `StaggeredModel` and `StaggeredStepping`: three `StaticNonlinearModel`s solved in turn, with the states of
#   the other fields read between their previous and current values (`StageBlend`);
# - prescribing the state of another problem on a boundary, `BoundaryCondition(tag, Coupling(...))`;
# - moving nodal values between meshes with `Transfer`/`transfer!` (the Maxwell traction of the vacuum on the beam);
# - the record of a staggered solve, `get_log`.
#
# Measured on a Mac M5 Max (Julia 1.13.0, 4 threads, 4 BLAS threads; HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8; compiled backend), whole run including compilation:
# FAST 74 s, full 80 s (20 steps of 2 passes, 28 s in `solve!`, of which about 11 s compile the three problems at
# the first pass). 25 375 unknowns: 8 563 magnetic potential, 1 232 solid displacement, 15 580 vacuum displacement.
# One assembly of the tangent, compiled against Gridap: magnetic 3.9 ms / 27 ms, vacuum 6.0 ms / 29 ms.
#
# ## Problem statement
# The mesh is half of the problem (``x \ge 0``; ``x = 0`` is a plane of symmetry): a beam
# ``\Omega_s = (0, 50) \times (-1, 1)`` mm, clamped on its lower face for ``x < 2.5`` mm, inside a box of vacuum
# ``\Omega_v`` of 350 × 450 mm. The magnetic scalar potential ``\varphi`` is prescribed on the bottom and top of the
# box, ``\varphi = \Lambda\,\varphi_b`` and ``\varphi = 0``, which applies a vertical field
# ``\mathcal H_0 \approx \Lambda\,\varphi_b / 0.45`` A/m; the load factor ``\Lambda`` goes from 0 to 1.
#
# All the unknowns are written in the reference configuration: the displacement ``\boldsymbol u`` (of the beam
# and, fictitiously, of the vacuum), ``\boldsymbol F = \boldsymbol I + \nabla_0 \boldsymbol u``,
# ``\boldsymbol H = \operatorname{cof} \boldsymbol F``, ``J = \det \boldsymbol F``, and the Lagrangian magnetic
# field ``\boldsymbol{\mathcal H}_0 = -\nabla_0 \varphi``. In the vacuum the energy is that of free space mapped
# to the reference configuration,
# ```math
# \Psi_v(\boldsymbol F, \boldsymbol{\mathcal H}_0) = -\frac{\mu_0}{2J}\,
#   \lvert \boldsymbol H \boldsymbol{\mathcal H}_0 \rvert^2 ,
# ```
# and in the beam it adds the elastic energy and the energy of the remanent field
# ``\boldsymbol{\mathcal H}_r = \alpha_r \boldsymbol N`` along the magnetisation direction ``\boldsymbol N = (1, 0)``:
# ```math
# \Psi_s(\boldsymbol F, \boldsymbol{\mathcal H}_0, \boldsymbol N) = \Psi_{\mathrm{mec}}(\boldsymbol F, \boldsymbol N)
#   - \frac{\mu_0}{2J}\,\lvert \boldsymbol H \boldsymbol{\mathcal H}_0 \rvert^2
#   + \frac{\mu_0 (1 + \chi_t)}{J}\,(\boldsymbol H \boldsymbol{\mathcal H}_0) \cdot (\boldsymbol H \boldsymbol{\mathcal H}_r) .
# ```
# Here ``\chi_t = 0`` (the keyword `χt` of `HardMagnetic2D` defaults to the susceptibility `χe = 0`); the
# constructor's `χr` only enters a further term weighted by `βmok`, which is zero by default, so it is not set.
# The last term is the torque on the magnetised particles: a field across ``\boldsymbol N`` rotates the beam. In the
# undeformed state the induction is ``\boldsymbol{\mathcal B}_0 = -\partial\Psi_s/\partial\boldsymbol{\mathcal H}_0
# = \mu_0\,(\boldsymbol{\mathcal H}_0 - (1+\chi_t)\,\boldsymbol{\mathcal H}_r)``: with this sign convention the
# remanent magnetisation points along ``-\boldsymbol N``, so an upward field gives a clockwise torque and the free
# end of the beam moves down.
#
# The three problems, solved in this order at every load step, are:
#
# 1. **Magnetic potential** on ``\Omega_s \cup \Omega_v`` (Gauss's law for the magnetic induction
#    ``\boldsymbol{\mathcal B}_0 = -\partial \Psi / \partial \boldsymbol{\mathcal H}_0``) with the displacements fixed:
#    ```math
#    -\int_{\Omega_s} \frac{\partial \Psi_s}{\partial \boldsymbol{\mathcal H}_0} \cdot \nabla_0 \delta\varphi \,\mathrm dV
#    -\int_{\Omega_v} \frac{\partial \Psi_v}{\partial \boldsymbol{\mathcal H}_0} \cdot \nabla_0 \delta\varphi \,\mathrm dV = 0 .
#    ```
# 2. **Mechanics of the beam** with the magnetic field fixed, loaded on the interface ``\Gamma`` by the Maxwell
#    stress ``\boldsymbol P_v = \partial \Psi_v / \partial \boldsymbol F`` of the vacuum:
#    ```math
#    \int_{\Omega_s} \frac{\partial \Psi_s}{\partial \boldsymbol F} : \nabla_0 \delta\boldsymbol u \,\mathrm dV
#    + \int_{\Gamma} \boldsymbol t_M \cdot \delta\boldsymbol u \,\mathrm dA = 0 ,
#    ```
#    where ``\boldsymbol t_M`` reproduces at the nodes of ``\Gamma`` the nodal forces
#    ``\int_{\Omega_v} \boldsymbol P_v : \nabla_0 \delta\boldsymbol u \,\mathrm dV`` of the vacuum (see *Maxwell traction on the interface*).
# 3. **Motion of the vacuum**: a fictitious hyperelastic problem on ``\Omega_v`` with ``\boldsymbol u_v = \boldsymbol u_s``
#    on ``\Gamma``, whose only purpose is to let the vacuum mesh follow the beam without inverting its elements.
#
# Splitting the problem this way keeps three small, well-conditioned systems, at the price of a coupling that
# is only satisfied after enough passes; here every load step makes a fixed number of passes (see *Solve*).
#
# Reference: C. Perez-Garcia, R. Ortigosa, J. Martínez-Frutos, D. Garcia-Gonzalez, Topology and material
# optimization in ultra-soft magnetoactive structures: making advantage of residual anisotropies,
# Adv. Mater. (2025), [doi:10.1002/adma.202518489](https://doi.org/10.1002/adma.202518489).
#
# ## Setup
using HyperFEM
using Gridap, GridapGmsh
using HyperFEMKernels                                          # compiled backend (`backend=:compiled`)
using Gridap.FESpaces: assemble_vector!, zero_free_values
using Gridap.CellData: get_cell_points
using LinearAlgebra, Printf

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E09_hMRE_vacuum_staggered")
setupfolder(outdir)                                            # VTK output (not versioned)

# ## Geometry and mesh
# A Gmsh mesh of linear triangles (lengths in m) with physical groups for the beam (`Beam`), the vacuum
# (`vacuumb`, which includes a region kept for a permanent magnet in other variants of the problem), the interface
# between them (`Interface`) and the boundary conditions.
geomodel = GmshDiscreteModel(joinpath(@__DIR__, "..", "data", "models", "Short_magnet.msh"))

# ## Constitutive models
# SI units (m, Pa, A). The beam: a nonlinear Mooney–Rivlin matrix plus a transversely isotropic term along
# ``\boldsymbol N`` (the residual anisotropy left by the magnetisation process), with the parameters of the
# paper for a Dowsil-based hMRE (shear moduli of about 10 kPa), and the hard-magnetic coupling with a remanent field
# ``\mu_0 \mathcal H_r = 10`` mT. `MagnetoMechModel` adds the two; calling it returns the energy and its
# derivatives with respect to ``\boldsymbol F`` and ``\boldsymbol{\mathcal H}_0``, functions of ``(\boldsymbol F, \boldsymbol{\mathcal H}_0, \boldsymbol N)``.
μ0 = 4π * 1e-7                                                 # H/m
params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
model_mech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
             TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])
model_hard = HardMagnetic2D(μ0=μ0, αr=10e-3 / μ0, χe=0.0)
model_solid = MagnetoMechModel(magneto=model_hard, mechano=model_mech)

# The vacuum: free space for the magnetic problem, and for its motion a soft compressible Mooney–Rivlin material
# with the term ``\lambda (J^\gamma + J^{-\gamma})``, which grows quickly when an element is compressed or
# stretched. Its tangent is not positive definite for large distortions; `HessianRegularization` clamps the
# eigenvalues of ``\partial^2 \Psi / \partial \boldsymbol F \partial \boldsymbol F`` at ``\delta`` (energy and stress
# unchanged), so that Newton always moves downhill.
model_vacuum_mag = IdealMagnetic2D(μ0=μ0, χe=0.0)
model_vacuum_mech = HessianRegularization(mechano=NonlinearMooneyRivlin2D_CV(λ=1.0, μ1=1.0, μ2=0.0, α1=6.0, α2=1.0, γ=6.0), δ=1e-6)

# ## Domains
# The magnetic potential lives on the whole mesh, the displacements on the beam and on the vacuum separately.
# `Γs` is the boundary of the beam on the `Interface` tag, seen from the beam.
Ωdomain = Triangulation(geomodel, tags=["vacuumb", "Beam"])
Ωsolid = Triangulation(geomodel, tags=["Beam"])
Ωvacuum = Triangulation(geomodel, tags=["vacuumb"])
order = 1
dΩsolid = Measure(Ωsolid, 2 * order)
dΩvacuum = Measure(Ωvacuum, 2 * order)
Γs = BoundaryTriangulation(Ωsolid, tags="Interface")
dΓs = Measure(Γs, 4 * order)

# ## Boundary conditions
# The potential ramps with the load factor at the bottom of the box; the beam is clamped on `Fixed_Solid` and
# symmetric on `Fixed_Solid_x` (only ``u_x = 0``). The vacuum is fixed on the outer box and symmetric on ``x = 0``.
# FAST applies half of the potential.
φbot = FAST ? 0.2 * 5200 / 2 : 0.2 * 5200                      # A (magnetic scalar potential)
Dφ = BoundaryConditions(BoundaryCondition("phibot", φbot, Λ -> Λ), BoundaryCondition("phitop", 0.0, Λ -> Λ))
Du_solid = BoundaryConditions(BoundaryCondition("Fixed_Solid", [0.0, 0.0]),
                              BoundaryCondition("Fixed_Solid_x", [0.0, 0.0]; mask=(true, false)))

# ## Finite element spaces and states
# A staggered solve needs, for every field, the current state (⁺) and its value at the previous pass (⁻), each an
# `FEFunction` on its own trial space: the forms of a field read the other fields through them.
reffeφ = ReferenceFE(lagrangian, Float64, order)
reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)

Vφ = TestFESpace(Ωdomain, reffeφ, Dφ, conformity=:H1)
Uφ⁺ = TrialFESpace(Vφ, Dφ); Uφ⁻ = TrialFESpace(Vφ, Dφ)
φh⁺ = FEFunction(Uφ⁺, zero_free_values(Uφ⁺)); φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ⁻))

Vu_solid = TestFESpace(Ωsolid, reffeu, Du_solid, conformity=:H1)
Uu_solid⁺ = TrialFESpace(Vu_solid, Du_solid); Uu_solid⁻ = TrialFESpace(Vu_solid, Du_solid)
uh_solid⁺ = FEFunction(Uu_solid⁺, zero_free_values(Uu_solid⁺)); uh_solid⁻ = FEFunction(Uu_solid⁻, zero_free_values(Uu_solid⁻))

# The vacuum follows the beam on the interface. `StageBlend(uh⁻, uh⁺)` is the beam displacement between its two
# states, ``\boldsymbol u_s(\tau) = \boldsymbol u_s^- + \tau\,(\boldsymbol u_s^+ - \boldsymbol u_s^-)``, with the stage
# ``\tau \in [0, 1]`` of the inner solve; `Coupling` prescribes it as Dirichlet data on the `Interface` nodes of
# the vacuum. The nodes are located once in the mesh of the beam (here they coincide, so the Dirichlet values are
# a copy of dof values) and updated in place at every solve.
uhsolid = StageBlend(uh_solid⁻, uh_solid⁺)
Du_vacuum = BoundaryConditions(BoundaryCondition("Fixed_Air", [0.0, 0.0]),
                               BoundaryCondition("Fixed_Air_x", [0.0, 0.0]; mask=(true, false)),
                               BoundaryCondition("Interface", Coupling(uhsolid)))
Vu_vacuum = TestFESpace(Ωvacuum, reffeu, Du_vacuum, conformity=:H1)
Uu_vacuum⁺ = TrialFESpace(Vu_vacuum, Du_vacuum); Uu_vacuum⁻ = TrialFESpace(Vu_vacuum, Du_vacuum)
uh_vacuum⁺ = FEFunction(Uu_vacuum⁺, zero_free_values(Uu_vacuum⁺)); uh_vacuum⁻ = FEFunction(Uu_vacuum⁻, zero_free_values(Uu_vacuum⁻))

uhvacuum(τ) = uh_vacuum⁻ + (uh_vacuum⁺ - uh_vacuum⁻) * τ        # the same blend written as a closure
φh(τ) = φh⁻ + (φh⁺ - φh⁻) * τ

# The magnetisation direction ``\boldsymbol N = (1, 0)``, as a finite element field on the beam.
Nh = interpolate_everywhere(x -> VectorValue(1.0, 0.0), TestFESpace(Ωsolid, reffeu))
@printf("Unknowns: magnetic potential %d, solid displacement %d, vacuum displacement %d (total %d)\n",
        num_free_dofs(Uφ⁺), num_free_dofs(Uu_solid⁺), num_free_dofs(Uu_vacuum⁺),
        num_free_dofs(Uφ⁺) + num_free_dofs(Uu_solid⁺) + num_free_dofs(Uu_vacuum⁺))

# ## Maxwell traction on the interface
# The magnetic stress of the vacuum acts on the beam. Its nodal forces,
# ``f_a = \int_{\Omega_v} \boldsymbol P_v : \nabla_0 N_a \,\mathrm dV``, are assembled on a vacuum space without
# Dirichlet conditions; `Transfer` takes the values at the interface nodes into a space `Usolid_int` defined only on
# ``\Gamma`` (built once, it is a gather of dof values when the nodes coincide), and the solve with the interface mass
# matrix ``\boldsymbol M`` turns them into a traction field ``\boldsymbol t_M`` with
# ``\int_\Gamma \boldsymbol t_M \cdot \delta\boldsymbol u \,\mathrm dA = \sum_a f_a \cdot \delta\boldsymbol u_a``.
# (`StaticLinearModel(...)(x)` solves ``\boldsymbol M \boldsymbol y = \boldsymbol x`` in place, reusing its factorisation.)
V_maxwell = TestFESpace(Ωvacuum, reffeu)
Pmaxw = FEFunction(V_maxwell, zero_free_values(V_maxwell))
Pmaxw_ = get_free_dof_values(Pmaxw)
Usolid_int = FESpace(Γs, reffeu)
tmaxw = interpolate_everywhere(Pmaxw, Usolid_int)
tmaxw_ = get_free_dof_values(tmaxw)
T_maxwell = Transfer(V_maxwell, Usolid_int)                    # vacuum → interface of the beam
V_traction = TestFESpace(Γs, reffeu)
mass_Γ = StaticLinearModel((u, v) -> ∫(u ⋅ v)dΓs, TrialFESpace(V_traction), V_traction, NothingBC(); backend=:compiled)

# ## Weak forms
# Each problem gets its residual and Jacobian as functions of the stage ``\tau`` of its inner solve. The fields solved
# earlier in the same pass are read between their previous and new values (`φh(τ)`, `uhsolid(τ)`), so that the
# inner load stepping of a problem applies the change of the other fields gradually (and can bisect it); the fields
# solved later are read at their last value (there ``⁻`` and ``⁺`` coincide).
Ψs, ∂Ψs∂F, ∂Ψs∂ℋ, ∂Ψs∂FF, ∂Ψs∂ℋF, ∂Ψs∂ℋℋ = model_solid()
Ψv, ∂Ψv∂F, ∂Ψv∂ℋ, ∂Ψv∂FF, ∂Ψv∂ℋF, ∂Ψv∂ℋℋ = model_vacuum_mag()
Ψvm, ∂Ψvm∂F, ∂Ψvm∂FF = model_vacuum_mech()
F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
ℋ₀ = get_Kinematics(Kinematics(Magneto, Solid))               # ∇φ ↦ -∇φ

# Problem 1, magnetic potential (``\boldsymbol{\mathcal H}_0 = -\nabla_0\varphi`` gives the signs):
res_mag(τ) = (φ, vφ) ->
  -1.0 * ∫(∇(vφ) ⋅ (∂Ψs∂ℋ ∘ (F ∘ (∇(uhsolid(τ))'), ℋ₀ ∘ (∇(φ)), Nh)))dΩsolid -
  ∫(∇(vφ) ⋅ (∂Ψv∂ℋ ∘ (F ∘ (∇(uhvacuum(τ))'), ℋ₀ ∘ (∇(φ)))))dΩvacuum
jac_mag(τ) = (φ, dφ, vφ) ->
  ∫(∇(vφ) ⋅ ((∂Ψs∂ℋℋ ∘ (F ∘ (∇(uhsolid(τ))'), ℋ₀ ∘ (∇(φ)), Nh)) ⋅ ∇(dφ)))dΩsolid +
  ∫(∇(vφ) ⋅ ((∂Ψv∂ℋℋ ∘ (F ∘ (∇(uhvacuum(τ))'), ℋ₀ ∘ (∇(φ)))) ⋅ ∇(dφ)))dΩvacuum

# Problem 2, mechanics of the beam. Before returning the form, `res_mech(τ)` updates the Maxwell traction with the
# current fields (the three steps of the previous section); it is a load, so it is not differentiated in the Jacobian.
res_mech(τ) = begin
  assemble_vector!(v -> ∫(∇(v)' ⊙ (∂Ψv∂F ∘ (F ∘ (∇(uhvacuum(τ))'), ℋ₀ ∘ (∇(φh(τ))))))dΩvacuum, Pmaxw_, V_maxwell)
  transfer!(tmaxw_, T_maxwell, Pmaxw)
  mass_Γ(tmaxw_)
  (u, v) -> ∫(∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φh(τ))), Nh)))dΩsolid + ∫(v ⋅ tmaxw)dΓs
end
jac_mech(τ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φh(τ))), Nh)) ⋅² (∇(du)')))dΩsolid

# Problem 3, motion of the vacuum:
res_vac(τ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψvm∂F ∘ (F ∘ (∇(u)'))))dΩvacuum
jac_vac(τ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψvm∂FF ∘ (F ∘ (∇(u)'))) ⋅² (∇(du)')))dΩvacuum

# ## Solvers and computational models
# One `StaticNonlinearModel` per problem, each solving for its own state (`xh=`), with the tolerances of the paper.
# Two of them need no Newton iteration at all, because the Dirichlet projection of `ProjectDirichlet=true` (see
# *Solve*) is a linear solve with the tangent that already does the work:
# - the magnetic problem is linear in ``\varphi`` once the displacements are fixed, so the projection of the new
#   potential on the box gives its solution and Newton converges at iteration 0;
# - the motion of the vacuum is not solved to equilibrium: `rtol=2` accepts the starting point, the projection that
#   moves the interface to the new position of the beam (one linear solve per pass), which is all the vacuum needs,
#   a valid mesh that follows the beam (`maxiter=1` only bounds what is never used).
# The beam is solved to a 1 % relative residual, as in the paper (the coupling error of a staggered step is larger
# than that). The regression values in *Checks* rest on these loosely converged iterates; they reproduce across
# machines because the number of iterations is fixed by the data: HyperFEM's test suite asserts the FAST values
# below at `rtol = 1e-6` on Linux. All of them, and the interface mass
# matrix above, assemble with the compiled backend (`backend=:compiled`, tutorial 01): at every pass the forms are
# rebound to the new stage and to the current states of the other fields, never compiled again.
nls_mag = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-9, rtol=1e-8, verbose=false)
cm_mag = StaticNonlinearModel(res_mag, jac_mag, Uφ⁺, Vφ, Dφ; nls=nls_mag, xh=φh⁺, backend=:compiled)
nls_mech = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-6, rtol=1e-2, verbose=false)
cm_mech = StaticNonlinearModel(res_mech, jac_mech, Uu_solid⁺, Vu_solid, Du_solid; nls=nls_mech, xh=uh_solid⁺, backend=:compiled)
nls_vac = NewtonRaphson(LUSolver(); maxiter=1, rtol=2, verbose=false)
cm_vac = StaticNonlinearModel(res_vac, jac_vac, Uu_vacuum⁺, Vu_vacuum, Du_vacuum; nls=nls_vac, xh=uh_vacuum⁺, backend=:compiled)

# `StaggeredModel` takes the three models (in the order they are solved), their states and the lagged states.
comp_model = StaggeredModel((cm_mag, cm_mech, cm_vac), (φh⁺, uh_solid⁺, uh_vacuum⁺), (φh⁻, uh_solid⁻, uh_vacuum⁻))

# ## Post-processing
# `on_step` runs after every accepted step: it writes the deformed beam and the vacuum with the field
# ``\boldsymbol{\mathcal H}_0 = -\nabla_0\varphi``.
pvd_solid = createpvd(joinpath(outdir, "solid"))
pvd_vacuum = createpvd(joinpath(outdir, "vacuum"))
nvtk = Ref(0)
function write_step(τ, Δτ, nsub)
  n = (nvtk[] += 1)
  pvd_solid[τ] = createvtk(Ωsolid, joinpath(outdir, "solid_$n"), cellfields=["u" => uh_solid⁺, "H0" => -∇(φh⁺)])
  pvd_vacuum[τ] = createvtk(Ωvacuum, joinpath(outdir, "vacuum_$n"), cellfields=["u" => uh_vacuum⁺, "H0" => -∇(φh⁺)])
end

# ## Solve
# `StaggeredStepping` controls the outer loop: `nsteps` equal load steps, and in each one `max_substeps` passes
# through the three problems. Without `coupling_tol` this is the **fixed loop**: every step makes exactly that number
# of passes, whatever the remaining coupling error (the paper's setting; see *Going further* for the converged loop).
# `kargsolve` holds the keywords of the inner `solve!` of each problem: one load increment per pass, bisected up to
# five times if Newton fails, and `ProjectDirichlet=true` for the two problems with moving Dirichlet data (the
# potential and the interface of the vacuum).
stepping = FAST ? StaggeredStepping(nsteps=4, max_substeps=1, maxbisec=1) :
                  StaggeredStepping(nsteps=20, max_substeps=2, maxbisec=1)
kargsolve = ((stepping=LoadStepping(nsteps=1, maxbisec=5), ProjectDirichlet=true),
             (stepping=LoadStepping(nsteps=1, maxbisec=5),),
             (stepping=LoadStepping(nsteps=1, maxbisec=5), ProjectDirichlet=true))
x, flag = solve!(comp_model; stepping=stepping, kargsolve=kargsolve, verbose=true, on_step=write_step)
savepvd(pvd_solid); savepvd(pvd_vacuum)

# `get_log` returns the record of the solve: per accepted step, the passes, the coupling error of every field
# after the last pass (the relative change of its free values in that pass) and the time spent in each problem.
# With one pass per step that change is just the increment of the step; from the second pass on it measures how
# far the three fields still are from agreeing with each other.
slog = get_log(comp_model)
println(slog)
tf = reduce(+, slog.times)
@printf("time in the inner solves: magnetic %.1f s, beam %.1f s, vacuum %.1f s\n", tf...)
@printf("coupling error after the last pass of the last step: magnetic %.1e, beam %.1e, vacuum %.1e\n", slog.errors[end]...)

# ## Checks
# Displacement of the tip of the beam (``x = 50`` mm), averaged over the tip face.
χtip = CellField(x -> x[1] > 0.05 - 1e-6 ? 1.0 : 0.0, Γs)
u_tip = sum(∫(χtip * uh_solid⁺)dΓs) / sum(∫(χtip)dΓs)
@printf("tip displacement: (%.4f, %.4f) mm\n", 1e3 * u_tip[1], 1e3 * u_tip[2])

# (b) The vacuum follows the beam: on the interface its displacement equals the one of the beam. Both are taken to
# the interface space with `Transfer` (the beam's nodes there are free or clamped, the vacuum's are all Dirichlet).
us_Γ = zeros(num_free_dofs(Usolid_int)); uv_Γ = zeros(num_free_dofs(Usolid_int))
transfer!(us_Γ, Transfer(Vu_solid, Usolid_int), uh_solid⁺)
transfer!(uv_Γ, Transfer(Vu_vacuum, Usolid_int), uh_vacuum⁺)
gap = maximum(abs.(uv_Γ .- us_Γ)) / maximum(abs.(us_Γ))
println("interface gap |u_vacuum - u_solid| / max|u_solid| = $gap")

# (b) Neither mesh is inverted: the smallest ``J`` at the quadrature points.
Jmin(uh, dΩ) = minimum(minimum, (J ∘ (F ∘ ∇(uh)'))(get_cell_points(dΩ)))
Jmin_s, Jmin_v = Jmin(uh_solid⁺, dΩsolid), Jmin(uh_vacuum⁺, dΩvacuum)
@printf("min J: beam %.4f, vacuum %.4f\n", Jmin_s, Jmin_v)

nφ, ns, nv = norm(get_free_dof_values(φh⁺)), norm(get_free_dof_values(uh_solid⁺)), norm(get_free_dof_values(uh_vacuum⁺))
@printf("‖φ‖ = %.10e   ‖u_s‖ = %.10e   ‖u_v‖ = %.10e   u_tip = (%.10e, %.10e)\n", nφ, ns, nv, u_tip[1], u_tip[2])

@assert flag == 1 "an inner solve did not converge"
@assert length(slog.τ) == stepping.nsteps && all(==(stepping.max_substeps), slog.nsub) "unexpected number of steps or passes"
@assert gap < 1e-10 "the vacuum does not follow the beam on the interface (gap $gap)"
@assert Jmin_s > 0 && Jmin_v > 0 "inverted elements"
@assert u_tip[2] < 0 && u_tip[1] < 0 "the tip should move down and towards the clamp (clockwise rotation)"
# (c) Regression. FAST is the reduced case of the test suite of HyperFEM (`test/SimulationsTests/HMREVacuumSimulation.jl`,
# half of the potential, 4 steps of one pass); the same values here show that `StaggeredStepping` reproduces it.
# The full case: our own values (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0, macOS arm64).
if FAST
  @assert isapprox(nφ, 2.0940145799e4; rtol=1e-6) "‖φ‖ = $nφ differs from the HyperFEM test"
  @assert isapprox(ns, 1.0426282896e-1; rtol=1e-6) "‖u_s‖ = $ns differs from the HyperFEM test"
  @assert isapprox(nv, 2.4014448097e-1; rtol=1e-6) "‖u_v‖ = $nv differs from the HyperFEM test"
else
  REF_PHI, REF_US, REF_UV, REF_TIP = 4.2100933777e4, 1.9454136102e-1, 4.5495058874e-1, -1.5853670013e-2
  @assert isapprox(nφ, REF_PHI; rtol=1e-5) "‖φ‖ = $nφ differs from the reference"
  @assert isapprox(ns, REF_US; rtol=1e-5) "‖u_s‖ = $ns differs from the reference"
  @assert isapprox(nv, REF_UV; rtol=1e-5) "‖u_v‖ = $nv differs from the reference"
  @assert isapprox(u_tip[2], REF_TIP; rtol=1e-5) "tip deflection $(u_tip[2]) differs from the reference"
end

# ## Going further
# - **Converged staggered loop.** With a coupling tolerance every step makes passes until the relative change of
#   every field is below it, and a step that does not converge in `max_substeps` passes is rejected and bisected.
#   The plain iteration converges slowly here (the field and the beam pull on each other); a quasi-Newton update of
#   the beam displacement built from the passes of the step, `IQNILS()`, accelerates it (one relaxation per field):
#   ```julia
#   stepping = StaggeredStepping(nsteps=20, max_substeps=30, coupling_tol=1e-6, maxbisec=4,
#                                relaxation=(:none, IQNILS(), :none))
#   ```
#   `get_log(comp_model)` then shows how many passes every step needed. Compare the tip deflection with the fixed loop.
# - `Aitken()` is a cheaper scalar relaxation; `StaggeredStepping(grow=1.5)` lets the steps grow when they converge in
#   few passes.
# - Tutorial 08 solves the same beam without the vacuum (the field applied directly in the beam), in one problem.
# - Paper and original scripts: Perez-Garcia et al., Adv. Mater. (2025), doi:10.1002/adma.202518489; the
#   repository `MultiSimOLab/Adv_Mater_Ultra_Soft_Magnet` (`src/hMRE_Vacuum_2D.jl`).
