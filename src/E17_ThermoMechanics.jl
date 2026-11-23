# # Tutorial 17: Thermo-mechanics: transient heat conduction and thermal expansion
#
# One end of a hyperelastic bar is suddenly heated while the other is kept at the initial temperature. Heat
# diffuses along the bar and the bar expands, freely: it is held only as much as needed to remove its rigid
# motions. The temperature has a classical series solution, and the expansion of a free body has two exact
# properties (the change of length is the integral of the thermal strain, and a linear temperature profile
# leaves the body free of stress), so every part of the coupled computation can be checked.
#
# Measured on a Mac (M5 Max, Julia 1.13.0, 4 threads, 4 BLAS threads; HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8; compiled backend), whole run including compilation: FAST (752 dofs, 143 time steps) 46 s,
# full (4 503 dofs, 210 time steps) 49 s; median time per step after compilation 3.9 ms (FAST) and 26 ms (full).
#
# **What you will learn**
# - the thermo-mechanical models of HyperFEM: `ThermalModel`, a `ThermalLaw` and `ThermoMechModel`, and the
#   derivatives they return as functions of ``(\boldsymbol F, \vartheta)``;
# - the weak form of a coupled displacement–temperature problem written term by term from the derivatives
#   of the free energy: mechanical equilibrium, heat conduction, the coupling and the heat capacity;
# - a first-order transient problem in `TransientModel`: a `mass` form on one field only (heat capacity),
#   quasi-static mechanics, the `BDF(2)` scheme and time steps controlled by the error (`AdaptiveStepping`)
#   with exact output times (`tstops`);
# - how to restrain a free body with symmetry planes and a point, and how to check the compiled backend
#   (`work_summary`, counters) in a time loop.
#
# ## Problem statement
# The unknowns are the displacement ``\boldsymbol u`` and the temperature increment
# ``\vartheta = \theta - \theta_r`` over the reference temperature ``\theta_r``. With the Helmholtz free energy
# ``\Psi(\boldsymbol F, \vartheta)``, the first Piola stress is ``\boldsymbol P = \partial_{\boldsymbol F}\Psi``,
# and the heat flux follows Fourier's law in the reference configuration, ``\boldsymbol Q = -k\nabla_0\vartheta``.
# Neglecting inertia and the heat produced by the deformation (the thermo-elastic heating
# ``-\theta\,\partial^2_{\boldsymbol F\vartheta}\Psi:\dot{\boldsymbol F}``: for a free body it is
# ``9K\alpha_{th}^2\theta_r\,\dot\vartheta``, a fraction ``9K\alpha_{th}^2\theta_r/c_v \approx 5.7\cdot10^{-5}`` of the
# heat-capacity term with the data below, whatever the amplitude; see «Going further»), the problem reads:
# find ``(\boldsymbol u, \vartheta)`` such that
# ```math
# \int_\Omega \boldsymbol P(\boldsymbol F, \vartheta) : \nabla_0 \boldsymbol v \,\mathrm dV = 0 , \qquad
# \int_\Omega c_v\,\dot\vartheta\,\delta\vartheta \,\mathrm dV
# + \int_\Omega k\,\nabla_0\vartheta\cdot\nabla_0\delta\vartheta \,\mathrm dV = 0
# \qquad \forall (\boldsymbol v, \delta\vartheta) ,
# ```
# with ``\boldsymbol F = \boldsymbol I + \nabla_0\boldsymbol u``, the heat capacity per unit volume ``c_v`` and
# the conductivity ``k``. The free energy of `ThermoMechModel` is
# ```math
# \Psi(\boldsymbol F, \vartheta) = f(\theta)\,\Psi_m(\boldsymbol F) + \Psi_t(\vartheta)
# - 3\alpha\,(J^2 - 1)\,\vartheta , \qquad J = \det\boldsymbol F ,
# ```
# with the neo-Hookean ``\Psi_m = \tfrac{\mu}{2}(\boldsymbol F:\boldsymbol F - 3) - \mu\ln J + \tfrac{\lambda}{2}(J-1)^2``,
# the purely thermal ``\Psi_t(\vartheta) = c_v\,[\vartheta - \theta\ln(\theta/\theta_r)]``
# (whose ``-\theta\,\partial^2_{\vartheta\vartheta}\Psi_t = c_v``), the thermal law ``f(\theta)`` (here
# ``f = 1``: the stiffness does not depend on the temperature) and the coupling term. Linearising
# ``\boldsymbol P`` at ``\boldsymbol F = \boldsymbol I``, a free body heated uniformly by ``\vartheta`` expands with
# the strain ``\varepsilon = 6\alpha\vartheta/(3\lambda + 2\mu)``: the coefficient of linear thermal expansion is
# ``\alpha_{th} = 2\alpha/K``, with the bulk modulus ``K = \lambda + 2\mu/3``. We give ``\alpha_{th}`` and set
# ``\alpha = K\alpha_{th}/2``.
#
# The bar ``\Omega = (0,L)\times(0,W)\times(0,H)`` is a quarter of a bar of section ``2W\times 2H``: the planes
# ``y = 0`` and ``z = 0`` are symmetry planes (no normal displacement), and the origin cannot move along ``x``.
# These are the minimal restraints: they remove the rigid motions and nothing else. The bar starts at
# ``\vartheta = 0``; the temperature of the end ``x = 0`` rises linearly to ``\Delta T`` in a short time
# ``t_r``, ``\vartheta(0, t) = \Delta T\,g(t)`` with ``g(t) = \min(t/t_r, 1)``, the end ``x = L`` is kept at
# ``\vartheta = 0`` and the lateral faces are insulated. (A sudden jump of the end temperature, ``t_r = 0``, is
# the textbook case, but neither the mesh nor the error control of the time step can follow it: the step
# collapses to its minimum on fine meshes. A short ramp is also what a real heater does.) The temperature
# does not depend on the displacement, so it is that of a 1D rod; by Duhamel's principle,
# ```math
# \vartheta(x, t) = \Delta T\,g(t)\Big(1 - \frac{x}{L}\Big) - \sum_{n\ge1} \frac{2\Delta T}{n\pi}\,
# \frac{1 - e^{-\lambda_n \min(t, t_r)}}{\lambda_n t_r}\, e^{-\lambda_n \max(t - t_r, 0)} \sin\frac{n\pi x}{L} ,
# \qquad \lambda_n = \frac{n^2\pi^2 D}{L^2} , \quad D = \frac{k}{c_v} ,
# ```
# (for ``t_r \to 0`` the classical series of the sudden jump), which tends to the linear profile
# ``\Delta T(1 - x/L)``. For the mechanics, two exact results of linear
# thermo-elasticity for a free body: (i) by the reciprocal theorem (with a uniform uniaxial stress as the
# second state), the change of length of the bar, averaged over the end faces, is the integral of the thermal
# strain, ``\Delta\bar\ell(t) = \alpha_{th}\int_\Omega \vartheta\,\mathrm dV / A`` (``A = WH``), at every time;
# (ii) a temperature linear in the coordinates gives a compatible thermal strain, so at steady state the
# bar is free of stress and ``\Delta\bar\ell = \alpha_{th}\Delta T L/2``.
#
# The data are dimensionless for the mechanics and the conduction (any consistent set of units), and the
# temperature is in kelvin: ``L = 1``, ``W = H = 0.1``, ``\mu = 1``, ``\lambda = 1.5`` (``\nu = 0.3``),
# ``c_v = 1``, ``k = 1`` (``D = 1``, diffusion time ``L^2/(\pi^2 D) \approx 0.1``), ``t_r = 0.01``,
# ``\theta_r = 293.15``, ``\Delta T = 10``, ``\alpha_{th} = 10^{-4}`` (thermal strains of ``10^{-3}`` at most:
# small deformations).
#
# ## Setup
using HyperFEM
using HyperFEMKernels                                          # compiled backend (`backend=:compiled`)
using Gridap
using Printf

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E17_ThermoMechanics")
setupfolder(outdir)                                            # VTK output (not versioned)

# ## Geometry and mesh
# A Cartesian mesh of hexahedra along the bar. The temperature varies along ``x`` only, so the section needs
# few elements. The restraints of the displacement use four tags: the origin (all components fixed through
# `mask` below), the edge ``y = z = 0`` (on both symmetry planes), and the two symmetry planes. A node that
# belongs to several tags takes the first of them in the list of conditions, so the most restrained come first.
const L, W, H = 1.0, 0.1, 0.1
partition = FAST ? (16, 1, 1) : (32, 2, 2)
geometry = CartesianDiscreteModel((0, L, 0, W, 0, H), partition)
labels = get_face_labeling(geometry)
add_tag_from_tags!(labels, "origin", CartesianTags.corner000)
add_tag_from_tags!(labels, "axis", CartesianTags.edgeX00⁺)
add_tag_from_tags!(labels, "y0", CartesianTags.faceX0Z⁺)
add_tag_from_tags!(labels, "z0", CartesianTags.faceXY0⁺)
add_tag_from_tags!(labels, "hot", CartesianTags.face0YZ⁺)
add_tag_from_tags!(labels, "cold", CartesianTags.face1YZ⁺)

# ## Constitutive model
# `ThermalModel` holds ``c_v``, ``\theta_r``, ``\alpha`` and ``k``; `ConstantEnergyLaw()` is the law ``f = 1``
# (`EntropicElasticityLaw(θr=θr, γ=0.0)`, ``f = \theta/\theta_r``, would make the stiffness proportional to
# the absolute temperature, as in rubber). Calling the model returns the energy and its derivatives as
# functions of ``(\boldsymbol F, \vartheta)``:
# ``(\Psi, \partial_{\boldsymbol F}\Psi, \partial_\vartheta\Psi, \partial^2_{\boldsymbol F\boldsymbol F}\Psi,
# \partial^2_{\vartheta\vartheta}\Psi, \partial^2_{\boldsymbol F\vartheta}\Psi, \eta)``, with the entropy
# ``\eta = -\partial_\vartheta\Psi``. These functions are all the weak form needs; we compose them below with
# the kinematics. The mechanical kinematics gives ``\boldsymbol F = \boldsymbol I + \nabla_0\boldsymbol u``
# from the displacement gradient (Gridap's `∇(u)` is the transpose of ``\nabla_0\boldsymbol u``, hence
# `F ∘ ∇(u)'`); the temperature enters the functions as it is (no kinematic map: `Kinematics(Thermo, Solid)`
# exists but carries no function, so we do not use it).
const μ, λ = 1.0, 1.5
const cv, k = 1.0, 1.0
const θr, ΔT, αth = 293.15, 10.0, 1e-4
const t_r = 0.01                                               # duration of the ramp of the end temperature
const K = λ + 2μ / 3                                           # bulk modulus
const E = μ * (3λ + 2μ) / (λ + μ)                              # Young's modulus
const D = k / cv                                               # thermal diffusivity
physmodel = ThermoMechModel(ThermalModel(Cv=cv, θr=θr, α=K * αth / 2, κ=k), NeoHookean3D(λ=λ, μ=μ),
                            ConstantEnergyLaw())
Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ, η = physmodel()
F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))

# ## Boundary conditions
# One set per field, gathered by `MultiFieldBC`. The temperature of the end ``x = 0`` is ``\Delta T`` times the
# evolution function `EvolutionFunctions.ramp(t_r)`, ``g(t) = \min(t/t_r, 1)``, evaluated by `TransientModel` at
# the time of every stage (it is 0 in the initial state, which is therefore consistent with the conditions).
# The other conditions are constant.
Du = BoundaryConditions(
  BoundaryCondition("origin", [0.0, 0.0, 0.0]),
  BoundaryCondition("axis", [0.0, 0.0, 0.0]; mask=(false, true, true)),
  BoundaryCondition("y0", [0.0, 0.0, 0.0]; mask=(false, true, false)),
  BoundaryCondition("z0", [0.0, 0.0, 0.0]; mask=(false, false, true)))
Dθ = BoundaryConditions(BoundaryCondition("hot", ΔT, EvolutionFunctions.ramp(t_r)), BoundaryCondition("cold", 0.0))
D_bc = MultiFieldBC([Du, Dθ])

# ## Finite element spaces
# Quadratic displacements and linear temperatures (Q2/Q1): the thermal strain ``\alpha_{th}\vartheta`` and the
# strain of the displacement are then polynomials of the same degree, and the stress-free state of a linear
# temperature profile is represented exactly.
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, 2)
reffeθ = ReferenceFE(lagrangian, Float64, 1)
Vu = TestFESpace(geometry, reffeu, D_bc[1], conformity=:H1)
Vθ = TestFESpace(geometry, reffeθ, D_bc[2], conformity=:H1)
Uu = TrialFESpace(Vu, D_bc[1], 0.0)
Uθ = TrialFESpace(Vθ, D_bc[2], 0.0)
V = MultiFieldFESpace([Vu, Vθ])
U = MultiFieldFESpace([Uu, Uθ])
Ω = Triangulation(geometry)
dΩ = Measure(Ω, 4)
println("free dofs: ", num_free_dofs(U), " (u: ", num_free_dofs(Uu), ", ϑ: ", num_free_dofs(Uθ), ")")

# ## Weak form
# We write the residual and its tangent term by term from the derivatives of the free energy; every term
# is a Gridap integral of the model's functions composed (`∘`) with the kinematics at the trial functions.
# `⊙` contracts two second-order tensors (``\boldsymbol A:\boldsymbol B``) and `⋅²` the last two indices of a
# fourth-order tensor with a second-order one, ``(\mathbb C\cdot^2\boldsymbol A)_{ij} = C_{ijkl}A_{kl}``.
#
# **Mechanical equilibrium.** The stress is the derivative of the free energy at the current temperature,
# ```math
# R_u = \int_\Omega \boldsymbol P(\boldsymbol F, \vartheta) : \nabla_0\boldsymbol v \,\mathrm dV , \qquad
# \boldsymbol P = \partial_{\boldsymbol F}\Psi = \partial_{\boldsymbol F}\Psi_m - 6\alpha J\vartheta\,\boldsymbol H ,
# \qquad \boldsymbol H = J\boldsymbol F^{-T} ,
# ```
# where the second term is the thermal stress of the coupling energy (``f = 1``).
r_mech(u, θ, v) = ∫(∇(v)' ⊙ (∂Ψu ∘ (F ∘ ∇(u)', θ)))dΩ

# **Heat conduction.** Fourier's law in the reference configuration, ``\boldsymbol Q = -k\nabla_0\vartheta``:
# ```math
# R_\vartheta = \int_\Omega k\,\nabla_0\vartheta\cdot\nabla_0\delta\vartheta \,\mathrm dV .
# ```
r_cond(θ, vθ) = ∫(k * (∇(θ) ⋅ ∇(vθ)))dΩ

# **Tangent.** The derivative of ``R_u + R_\vartheta`` in the direction ``(\delta\boldsymbol u, \delta\vartheta)``
# (trial functions `du`, `dθ`) has three blocks. The mechanical block, with the elasticity tensor
# ``\mathbb C = \partial^2_{\boldsymbol F\boldsymbol F}\Psi``,
# ```math
# K_{uu} = \int_\Omega \nabla_0\boldsymbol v : \mathbb C : \nabla_0\delta\boldsymbol u \,\mathrm dV ;
# ```
# the coupling block, the stress produced by a temperature change, with
# ``\partial^2_{\boldsymbol F\vartheta}\Psi = -6\alpha J\boldsymbol H``,
# ```math
# K_{u\vartheta} = \int_\Omega \nabla_0\boldsymbol v : \partial^2_{\boldsymbol F\vartheta}\Psi\;\delta\vartheta \,\mathrm dV ;
# ```
# and the conduction block
# ```math
# K_{\vartheta\vartheta} = \int_\Omega k\,\nabla_0\delta\vartheta\cdot\nabla_0\delta\vartheta^{*} \,\mathrm dV
# ```
# (``\delta\vartheta^{*}`` the test function `vθ`). There is no block ``K_{\vartheta u}``: the conduction is
# written in the reference configuration and the thermo-elastic heating is neglected, so the temperature
# equation does not depend on the displacement.
j_mech(u, θ, du, v) = ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F ∘ ∇(u)', θ)) ⋅² ∇(du)'))dΩ
j_coupling(u, θ, dθ, v) = ∫(∇(v)' ⊙ ((∂Ψuθ ∘ (F ∘ ∇(u)', θ)) * dθ))dΩ
j_cond(dθ, vθ) = ∫(k * (∇(dθ) ⋅ ∇(vθ)))dΩ

# **Heat capacity.** The time derivative enters through the bilinear `mass` form,
# ```math
# M(\dot\vartheta, \delta\vartheta) = \int_\Omega c_v\,\dot\vartheta\,\delta\vartheta \,\mathrm dV ,
# \qquad c_v = -\theta\,\partial^2_{\vartheta\vartheta}\Psi ,
# ```
# with the heat capacity of the model (`∂Ψθθ(F, ϑ) = -c_v/θ` for `ThermalModel` and ``f = 1``, so ``c_v`` is
# the constant `Cv` given to it). The form acts on the temperature only: `TransientModel` assembles it once
# into a matrix ``\boldsymbol M`` and adds ``\boldsymbol M\dot{\boldsymbol x}`` (with ``\dot{\boldsymbol x}``
# given by the time scheme) to the residual of every stage, and its multiple to the tangent. The displacement
# has no time derivative: the mechanics is quasi-static, in equilibrium with the temperature of every step.
#
# The static forms are functions of the time `t`, as `TransientModel` expects; here they do not depend on it.
res(t) = ((u, θ), (v, vθ)) -> r_mech(u, θ, v) + r_cond(θ, vθ)
jac(t) = ((u, θ), (du, dθ), (v, vθ)) -> j_mech(u, θ, du, v) + j_coupling(u, θ, dθ, v) + j_cond(dθ, vθ)
mass = ((du, dθ), (v, vθ)) -> ∫(cv * (dθ * vθ))dΩ
@assert ∂Ψθθ(one(TensorValue{3,3,Float64}), 0.0) ≈ -cv / θr "heat capacity of the model differs from cv"

# ## Solver and computational model
# `BDF(2)`, the second-order backward differentiation formula, is a natural scheme for diffusion: implicit,
# stable for any step, and it damps the fast components of the start. The step is chosen by the local error
# estimate of the scheme (`AdaptiveStepping`): small at the start, when the temperature changes quickly near
# the heated end, and growing as the bar approaches the steady state. The estimate uses only the degrees of
# freedom with a time derivative (the temperature); `atol` is in kelvin. The tolerances of the step are tied
# to the mesh, ``\mathrm{tol} = 0.01\,h^2`` (relative to ``\Delta T``), so that the error of the time
# integration stays below that of the mesh, which is of order ``h^2`` for linear elements, when the mesh is
# refined. `tstops` are the end of the ramp and the times at which we compare with the series solution: the
# steps are cut to reach them exactly.
#
# `backend=:compiled` evaluates the forms with the compiled, threaded kernels of HyperFEMKernels; they are
# built once, and at every stage the model only rebinds them to the new state (no recompilation). The
# mechanics is nonlinear (finite strains), so the tangent is updated at every Newton iteration
# (`tangent_update=:iteration`, the default).
t_check = (t_r, 0.05, 0.2)
t_end = 1.5
h = L / partition[1]
tol_t = 0.01 * h^2
stepping = AdaptiveStepping(; Δt=1e-4, t_end, rtol=tol_t, atol=tol_t * ΔT, tstops=t_check)
nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8, verbose=0)
comp_model = TransientModel(res, jac, U, V, D_bc; scheme=BDF(2), mass, nls, backend=:compiled)
xh = get_state(comp_model)                                     # (u, ϑ) after every accepted step
uh, θh = xh

# ## Post-processing
# The exact temperature (series above) is a closure: the parameters are captured when it is built, so the
# function evaluated at the quadrature points reads only its argument and constants. After the ramp the
# terms are summed until the exponential factor is negligible; during the ramp the terms decay as ``n^{-3}``,
# and 4 000 of them leave a truncation error below ``10^{-6}\Delta T``.
function θ_exact(t)
  X -> begin
    x = X[1]
    s = ΔT * min(t / t_r, 1.0) * (1 - x / L)
    for n in 1:4000
      λn = (n * π / L)^2 * D
      e = exp(-λn * max(t - t_r, 0.0))
      e < 1e-16 && break
      s -= 2ΔT / (n * π) * (1 - exp(-λn * min(t, t_r))) / (λn * t_r) * e * sin(n * π * x / L)
    end
    s
  end
end

# After every accepted step we store the time, the change of length (mean ``u_x`` on the end faces) and its
# estimate from the reciprocal theorem; at the check times also the error of the temperature,
# ``\lVert\vartheta_h - \vartheta\rVert_{L^2} / (\Delta T\,|\Omega|^{1/2})``, and the VTK files.
Γ0 = BoundaryTriangulation(geometry, tags="hot")
Γ1 = BoundaryTriangulation(geometry, tags="cold")
dΓ0, dΓ1 = Measure(Γ0, 4), Measure(Γ1, 4)
const A = W * H
volΩ = sum(∫(1.0)dΩ)
ex = VectorValue(1.0, 0.0, 0.0)
elongation(uh) = (sum(∫(uh ⋅ ex)dΓ1) - sum(∫(uh ⋅ ex)dΓ0)) / A
elongation_thermal(θh) = αth * sum(∫(θh)dΩ) / A
θ_error(θh, t) = sqrt(sum(∫(abs2 ∘ (θh - θ_exact(t)))dΩ) / volΩ) / ΔT
stress_norm(uh, θh) = sqrt(sum(∫((∂Ψu ∘ (F ∘ ∇(uh)', θh)) ⊙ (∂Ψu ∘ (F ∘ ∇(uh)', θh)))dΩ) / volΩ)

times, Δℓ, Δℓ_th, wall = Float64[], Float64[], Float64[], Float64[]
checks = Dict{Float64,NamedTuple}()
x_005 = zeros(num_free_dofs(U))                                # state at t = 0.05 (for the check of the tangent)
work_2 = Ref(HyperFEMKernels.counters())                       # kernel counters after the second step
time_post = Ref(0.0)                                           # time spent in the post-processing
function record(post)
  time_post[] += @elapsed begin
    t = post.Λ[post.iter]
    push!(wall, time())                                        # wall clock, for the time per step
    post.iter == 2 && (work_2[] = HyperFEMKernels.counters())
    push!(times, t); push!(Δℓ, elongation(uh)); push!(Δℓ_th, elongation_thermal(θh))
    tc = findfirst(s -> isapprox(t, s; atol=1e-12), (t_check..., t_end))
    if tc !== nothing
      checks[t] = (err=θ_error(θh, t), σ=stress_norm(uh, θh), Δℓ=Δℓ[end], Δℓ_th=Δℓ_th[end])
      t == 0.05 && (x_005 .= get_free_dof_values(xh))
      pvd = get_pvd(post)
      pvd[t] = createvtk(Ω, joinpath(outdir, "step_$(post.iter).vtu"), cellfields=["u" => uh, "theta" => θh], order=2)
    end
  end
end
post = PostProcessor(comp_model, record; is_vtk=true, filepath=outdir)

# ## Solve
# Most of the run time is compilation: of the compiled kernels when the model is built, of the first
# Newton iteration, and of the Gridap integrals and the VTK output of the post-processing the first time
# they run. The median time per step is what a step costs after that: the coupled mechanics and conduction
# solved together, in one or two Newton iterations (the mechanics is almost linear here). The step stays
# small during the ramp and grows to about ``0.1`` near the steady state.
HyperFEMKernels.reset_counters!()
time_solve = @elapsed (x, flag) = solve!(comp_model; stepping, post)
@assert flag == 1 "the time loop did not reach t_end"
work = HyperFEMKernels.counters()
nsteps = length(times)
Δts = diff([0.0; times])
step_times = sort(diff(wall))                                  # wall time between accepted steps
println(@sprintf("%d steps, Δt ∈ [%.2e, %.2e]; solve %.1f s (post-processing %.1f s); median time per step %.1f ms",
                 nsteps, minimum(Δts), maximum(Δts), time_solve, time_post[], 1e3 * step_times[cld(end, 2)]))

# ## Efficiency
# What the compiled operators do (`work_summary`): every group of the stage forms and of the mass form takes
# a fast path (`:linear_maps`) and nothing falls back to Gridap. The kernels are generated when the model is
# built; the counters, reset before `solve!`, show that the time loop rebinds them to the new state at every
# attempt (`update_fast`). The moving Dirichlet values of the ramp need one more kernel (their contribution
# to the inertia term), generated once in the first step (`codegen`, `update_full`); after that nothing is
# generated or rebuilt: the counters after the second step and at the end agree. HyperFEM 0.3.0
# has no accessor for the operators of a `TransientModel`: the two lines below read internal fields, and
# may change.
fallbacks = 0
for (what, o) in (("stage forms", comp_model.caches.op.inner), ("mass form", comp_model.caches.opM))
  ws = HyperFEMKernels.work_summary(o)
  global fallbacks += ws.fallback_residual + ws.fallback_jacobian
  println(what, ": ", [(g.kind, g.domain, g.path) for g in ws.groups], "; Gridap fallbacks: ",
          ws.fallback_residual + ws.fallback_jacobian)
end
println("counters after the second step: ", work_2[])
println("counters of the solve:          ", work)
@assert fallbacks == 0 && work.fallback_assemblies == 0 "a form fell back to Gridap"
@assert work.codegen == work_2[].codegen && work.update_full == work_2[].update_full "kernels rebuilt during the time loop"

# ## Checks
# The values at the check times: error of the temperature, stress, change of length and its estimate.
println("\n   t        error ϑ                ‖P‖/(E αth ΔT)   Δℓ            Δℓ (reciprocal)")
for t in sort(collect(keys(checks)))
  c = checks[t]
  println(@sprintf("%6.3f   %.3e (%.2f h²)   %.3e        %.6e   %.6e", t, c.err, c.err / h^2, c.σ / (E * αth * ΔT), c.Δℓ, c.Δℓ_th))
end
@assert all(t -> haskey(checks, t), (t_check..., t_end)) "a check time was not reached exactly"

# The tangent written above must be the derivative of the residual written above: Newton converges (more
# slowly) even with a wrong tangent, so the solution alone does not test it. We compare the tangent applied
# to a direction ``\boldsymbol d`` with a central difference of the residual,
# ``[\boldsymbol r(\boldsymbol x + \epsilon\boldsymbol d) - \boldsymbol r(\boldsymbol x - \epsilon\boldsymbol d)]/(2\epsilon)``,
# at the state of ``t = 0.05`` (stressed, with a curved temperature profile). The direction moves every
# displacement and temperature dof, in proportion to its value with a varying factor, so that all the blocks,
# the coupling included, take part; the difference is ``O(\epsilon^2)`` plus round-off (``10^{-10}``). The rows of
# the two equations are compared apart: the conduction rows are much larger than the mechanical ones, which
# carry the coupling block (a sign error in ``K_{u\vartheta}`` gives a difference of order one there). The
# vectors and the matrix are assembled by Gridap from the same `res` and `jac`.
r_at(x) = assemble_vector(y -> res(0.05)(FEFunction(U, x), y), V)
d = x_005 .* (1 .+ 0.5 .* sin.(1:length(x_005)))
ϵ = 1e-6
Jd = assemble_matrix((dx, y) -> jac(0.05)(FEFunction(U, x_005), dx, y), U, V) * d
fd = (r_at(x_005 .+ ϵ .* d) .- r_at(x_005 .- ϵ .* d)) ./ (2ϵ)
nu = num_free_dofs(Uu)                                         # the displacement dofs come first
blockdiff(rows) = sqrt(sum(abs2, fd[rows] .- Jd[rows]) / sum(abs2, Jd[rows]))
tangent_error = max(blockdiff(1:nu), blockdiff(nu+1:length(d)))   # mechanical and thermal rows apart
println(@sprintf("tangent against finite differences of the residual: relative difference %.2e (u rows), %.2e (ϑ rows)",
                 blockdiff(1:nu), blockdiff(nu+1:length(d))))
@assert tangent_error < 1e-6 "the tangent is not the derivative of the residual ($tangent_error)"

# (a) The temperature against the series solution. The error comes from the mesh (linear elements,
# ``h = L/n_x``: of order ``h^2``) and from the time integration (whose tolerance is ``0.01h^2``). Measured
# with ``n_x = 16, 32, 64, 128`` (the FAST and full meshes, and two refinements of the full one), the errors
# are ``1.9h^2`` at ``t = t_r = 0.01`` (the mesh dominates: the steep profile near the heated end), and
# ``0.30``–``0.41\,h^2`` and ``0.14``–``0.32\,h^2`` at ``t = 0.05`` and ``0.2``, where the time error, made of
# more steps on finer meshes, adds a part that grows slowly with ``n_x``. The tolerances, ``3h^2``, ``0.8h^2``
# and ``0.6h^2``, hold for those four meshes with a margin of at least 1.5. At ``t = 1.5`` the series has
# decayed to ``10^{-6}\Delta T``, and the linear profile is represented exactly by the linear elements.
for (t, c) in zip(t_check, (3.0, 0.8, 0.6))
  @assert checks[t].err < c * h^2 "temperature at t = $t away from the series solution (error $(checks[t].err))"
end
@assert checks[t_end].err < 1e-5 "temperature at t = $t_end away from the steady linear profile"

# (b) The mechanics. The change of length matches the integral of the thermal strain at every step, and at
# steady state the linear estimate ``\alpha_{th}\Delta T L/2``, with the bar free of stress: the stress is
# ``10^{-5}`` of the thermal stress scale ``E\alpha_{th}\Delta T`` against ``10^{-3}``–``4\cdot10^{-2}`` during the
# transient. The remaining differences (``2\cdot10^{-3}`` relative) are the second-order terms of the
# finite-strain model, of relative size ``\alpha_{th}\Delta T = 10^{-3}``: with ``\Delta T = 1`` they drop
# tenfold (measured), so they are not a discretisation error.
rel_recip = maximum(abs.(Δℓ .- Δℓ_th) ./ Δℓ_th)
rel_steady = abs(Δℓ[end] / (αth * ΔT * L / 2) - 1)
σ_steady = checks[t_end].σ / (E * αth * ΔT)
println(@sprintf("reciprocal theorem: max relative difference %.2e; steady state: Δℓ/(αth ΔT L/2) - 1 = %.2e, ‖P‖/(E αth ΔT) = %.2e",
                 rel_recip, rel_steady, σ_steady))
@assert rel_recip < 5 * αth * ΔT "change of length away from the integral of the thermal strain ($rel_recip)"
@assert rel_steady < 5 * αth * ΔT "steady change of length away from αth ΔT L/2 ($rel_steady)"
@assert σ_steady < 1e-4 "the bar is not free of stress at steady state ($σ_steady)"
@assert checks[t_check[1]].σ / (E * αth * ΔT) > 1e-2 "no thermal stress during the transient"

# (c) Regression values (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0, macOS arm64).
println(@sprintf("regression: err(0.05) = %.10e, σ(0.05) = %.10e, Δℓ(0.2) = %.10e, Δℓ(end) = %.10e",
                 checks[0.05].err, checks[0.05].σ, checks[0.2].Δℓ, Δℓ[end]))
reference = FAST ? (err=1.1898517676e-03, σ=1.9909595440e-05, Δℓ02=4.4218746895e-04, Δℓend=5.0102569875e-04) :
                   (err=3.1409089772e-04, σ=1.9031434554e-05, Δℓ02=4.4183785276e-04, Δℓend=5.0102534280e-04)
@assert isapprox(checks[0.05].err, reference.err; rtol=1e-5) "regression: temperature error at t = 0.05"
@assert isapprox(checks[0.05].σ, reference.σ; rtol=1e-5) "regression: stress at t = 0.05"
@assert isapprox(checks[0.2].Δℓ, reference.Δℓ02; rtol=1e-5) "regression: change of length at t = 0.2"
@assert isapprox(Δℓ[end], reference.Δℓend; rtol=1e-5) "regression: final change of length"

# ## Going further
# - The thermo-elastic heating: the energy balance ``\theta\dot\eta + \nabla_0\cdot\boldsymbol Q = 0`` with
#   ``\eta = -\partial_\vartheta\Psi`` adds ``-\theta\,\partial^2_{\boldsymbol F\vartheta}\Psi:\dot{\boldsymbol F}`` to the
#   heat equation; it needs the general level of `TransientModel` (`∂t(ctx, u)` on the displacement).
# - `EntropicElasticityLaw(θr=θr, γ=0.0)` makes the shear stiffness proportional to ``\theta``, as in rubber;
#   `ThermoMech_Bonet` and `ThermoMech_EntropicPolyconvex` are the other thermo-mechanical models.
# - Clamp the end ``x = 0`` (all components): the bar can no longer expand freely, and the thermal stress
#   appears.
