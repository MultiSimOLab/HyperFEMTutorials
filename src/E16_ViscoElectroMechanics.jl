# # Tutorial 16: Visco-electro-mechanics: creep of a dielectric bimorph
#
# Soft dielectric elastomers are viscoelastic: under a constant voltage an actuator does not stop where it is
# when the voltage is reached, but keeps deforming for seconds or minutes as the polymer network relaxes. We
# model the material with a generalised Maxwell model (an elastic equilibrium branch plus two viscous branches
# with relaxation times of about 1 s and 10 s), apply a voltage ramp to a bimorph cantilever and hold it: the tip
# deflection creeps towards the value given by the equilibrium branch alone, which we compute separately. The
# creep turns out to be much slower than the relaxation times suggest, and we predict by how much.
#
# Measured on a Mac (M5 Max, 4 Julia threads, 4 BLAS threads, Julia 1.13.0, HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8; compiled backend), whole run including compilation: FAST 54 s (1 089 free dofs, 41 time steps,
# 35 ms per step), full 62 s (3 615 free dofs, 87 time steps, 0.10 s per step; the first step takes 7 s, mostly
# compilation). In a step of the full case the LU factorisation (17 ms, once per Newton iteration) dominates; one
# assembly of the tangent takes 4.9 ms compiled and 76 ms with Gridap's lazy evaluation.
#
# **What you will learn**
# - generalised Maxwell viscoelasticity (`GeneralizedMaxwell`, `ViscousPolyconvex`) inside an
#   electro-mechanical model (`ElectroMechModel`);
# - internal variables stored at the quadrature points (`CellState`) and updated once per accepted time step;
# - `TransientModel` with the `QuasiStatic` scheme: a time loop without inertia, where time enters through the
#   material (``\Delta t`` in the viscous update) and through a time-dependent voltage;
# - graded time steps with `TimeStepping(; grow, Δtmax, tstops)` to cover two relaxation times in a few steps;
# - checking a viscous computation against its long-time (equilibrium) limit, and checking that the compiled
#   backend is not compiled again at every step.
#
# ## Problem statement
# The plate ``\Omega = (0,L)\times(0,W)\times(0,H)`` (``L = 50`` mm, ``W = 5`` mm, ``H = 1`` mm) is clamped at
# ``x = 0``. It carries two compliant electrodes: one on the bottom face ``z = 0`` (grounded) and one on the
# mid-plane ``z = H/2``, whose potential follows a ramp of duration ``t_r = 1`` s up to ``V_0`` and is then held.
# The electric field lives in the lower layer only: it squeezes the layer across its thickness and, the material
# being nearly incompressible, the layer elongates. The upper layer resists, and the plate bends upwards (``+z``).
#
# As in tutorial 04, the unknowns are the displacement ``\boldsymbol u`` and the electric potential ``\varphi``,
# with ``\boldsymbol F = \boldsymbol I + \nabla_0\boldsymbol u`` and ``\boldsymbol E_0 = -\nabla_0\varphi``.
# The free energy is now the sum of three mechanical branches and the ideal dielectric,
# ```math
# \Psi(\boldsymbol F, \boldsymbol E_0; \boldsymbol C_{v,1}, \boldsymbol C_{v,2}) = \Psi_\infty(\boldsymbol F)
#   + \sum_{a=1}^{2} \Psi_a(\boldsymbol C, \boldsymbol C_{v,a})
#   - \frac{\varepsilon}{2J}\,|\boldsymbol H \boldsymbol E_0|^2 ,
# \qquad \boldsymbol C = \boldsymbol F^T\boldsymbol F ,
# ```
# with the compressible neo-Hookean equilibrium branch of tutorial 02,
# ``\Psi_\infty = \tfrac{\mu_\infty}{2}(\boldsymbol F:\boldsymbol F - 3) - \mu_\infty \ln J + \tfrac{\lambda}{2}(J-1)^2``,
# and two viscous (Maxwell) branches, each a neo-Hookean spring in series with a dashpot, written in terms of a
# viscous strain ``\boldsymbol C_{v,a}`` (``\det\boldsymbol C_{v,a} = 1``):
# ```math
# \Psi_a = \frac{\mu_a}{2}\left(\boldsymbol C : \boldsymbol C_{v,a}^{-1} - 3 J^{2/3}\right) .
# ```
# The viscous strains are **internal variables**: they are not unknowns of the finite element problem but
# evolve at every quadrature point by a flow rule with relaxation time ``\tau_a``, which drives
# ``\boldsymbol C_{v,a}`` towards the isochoric part of ``\boldsymbol C``. Integrated with a backward-Euler step from
# ``t_n`` to ``t_{n+1} = t_n + \Delta t`` it has a closed form (no local Newton iteration):
# ```math
# \boldsymbol C_{v,a}^{-1}\big|_{n+1} = \frac{\boldsymbol B}{(\det\boldsymbol B)^{1/3}} , \qquad
# \boldsymbol B = \frac{\Delta t}{\tau_a}\,\boldsymbol C_{n+1}^{-1} + \boldsymbol C_{v,a}^{-1}\big|_{n} .
# ```
# For ``\Delta t \ll \tau_a`` the viscous strain does not move and the branch acts as an extra spring of modulus
# ``\mu_a``: the **instantaneous** response is that of all branches together, ``\mu_\infty + \mu_1 + \mu_2``. Under a
# constant ``\boldsymbol C``, ``\boldsymbol C_{v,a}`` tends to ``J^{-2/3}\boldsymbol C``, the energy and the stress
# of the branch vanish, and the **long-time** response is that of the equilibrium branch alone.
#
# **Relaxation and creep.** ``\tau_a`` is the time with which the stress of branch ``a`` relaxes under a *fixed*
# deformation. Under a fixed *load* the deformation creeps with other times, the retardation times
# ``\lambda``: in the small-strain (linear) version of the model they are the roots of
# ```math
# \mu_\infty \lambda^2 - \left(\mu_\infty(\tau_1 + \tau_2) + \mu_1\tau_1 + \mu_2\tau_2\right)\lambda
#   + \tau_1\tau_2\,(\mu_\infty + \mu_1 + \mu_2) = 0
# ```
# (the zeros of the relaxation modulus ``\mu_\infty + \sum_a \mu_a s\tau_a/(1 + s\tau_a)`` in the Laplace variable,
# at ``s = -1/\lambda``). With a soft equilibrium branch they are much longer than the relaxation times: the
# dashpot of a stiff branch must stretch a long way before the soft branch takes over the load. With the
# material below, ``\lambda_1 = 1.67`` s and ``\lambda_2 = 39.1`` s (for ``\tau_1 = 0.82`` s and ``\tau_2 = 10.7`` s).
#
# The problem at ``t_{n+1}`` is the electro-mechanical problem of tutorial 04 with the stress of the step
# (equilibrium branch, viscous branches with ``\boldsymbol C_{v,a}`` given by the update above, and Maxwell stress):
# find ``(\boldsymbol u, \varphi)`` such that for all ``(\boldsymbol v, \delta\varphi)``
# ```math
# \int_\Omega \nabla_0 \boldsymbol v : \partial_{\boldsymbol F}\Psi \,\mathrm dV
#   - \int_\Omega \nabla_0 \delta\varphi \cdot \partial_{\boldsymbol E_0}\Psi \,\mathrm dV = 0 ,
# ```
# with ``\partial_{\boldsymbol F}\Psi`` evaluated at ``(\boldsymbol F_{n+1}, \boldsymbol E_{0,n+1})`` and depending
# on the internal variables of the previous step, ``\boldsymbol C_{v,a}|_n``, through the update. There is no
# inertia: the time only enters through the dashpots and the voltage (a *quasi-static* time loop). Once the step has converged, the
# internal variables are updated, and the next step starts from them.
#
# Material (from the HyperFEM test of this simulation): ``\mu_\infty = 14`` kPa, ``\lambda = 1`` MPa,
# ``\mu_1 = 56`` kPa with ``\tau_1 = 0.82`` s, ``\mu_2 = 34`` kPa with ``\tau_2 = 10.7`` s. The instantaneous shear
# modulus is 7.4 times the equilibrium one. The permittivity is ``\varepsilon = 1`` in scaled units: the
# solution depends on ``\varepsilon`` and the voltage only through ``\varepsilon V_0^2``, so ``V_0 = 0.008`` below is
# ``0.008/\sqrt{\varepsilon}`` volts for a material of permittivity ``\varepsilon`` (about 1.2 kV for an acrylic
# elastomer with ``\varepsilon \approx 4.2\cdot 10^{-11}`` F/m).
#
# ## Setup
using HyperFEM
using Gridap
using HyperFEMKernels                                          # compiled backend (`backend=:compiled`)
using LinearAlgebra: norm
using Printf

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E16_ViscoElectroMechanics")
setupfolder(outdir)                                            # VTK output (not versioned)

# ## Geometry and mesh
# A Cartesian mesh of the plate with two layers of elements across the thickness, so that the mid-plane is a
# mesh plane and can carry the electrode. `add_tag_from_vertex_filter!` tags the faces, edges and vertices whose
# vertices all satisfy the filter (the filter captures the height of the plane in a `let`, rather than reading a
# global variable). The bimorph bends, and trilinear hexahedra are much too stiff in bending when they are
# this thin (shear locking): we use triquadratic ones. Even so, the meshes are coarse: the equilibrium tip
# deflection is 5.45 mm with the FAST mesh and 5.77 mm with the full one (6 % apart; 5.9 mm, from a static run
# not shown here, with 40×2×4 elements). The checks compare results on the same mesh, so they hold for any of them.
L, W, H = 0.050, 0.005, 0.001                                  # m
partition = FAST ? (10, 1, 2) : (20, 2, 2)
model = CartesianDiscreteModel((0, L, 0, W, 0, H), partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
add_tag_from_tags!(labels, "bottom", CartesianTags.faceXY0⁺)
add_tag_from_vertex_filter!(labels, "mid", model, let zm = H / 2; x -> abs(x[3] - zm) < 1e-12 end)
add_tag_from_tags!(labels, "tip", CartesianTags.face1YZ)

# ## Constitutive model
# `GeneralizedMaxwell(equilibrium, branches...)` adds viscous branches to an elastic model; `ViscousPolyconvex`
# is the branch above. Calling the electro-mechanical model returns the energy and its derivatives as functions
# of ``(\boldsymbol F, \boldsymbol E_0, \boldsymbol F_n, \boldsymbol A_1, \boldsymbol A_2)``: the deformation
# gradient of the previous step and one internal variable per branch (here ``\boldsymbol A_a =
# \boldsymbol C_{v,a}|_n``) come after the arguments of tutorial 04. ``\boldsymbol F_n`` is part of the interface of
# the viscous models (it is passed to every branch); the update of `ViscousPolyconvex` above does not use it.
# The stress is the *algorithmic* stress, with the viscous strains given by the update of the step, and the
# tangent is its consistent derivative.
μ∞, λ = 1.4e4, 1.0e6                                           # Pa
μ₁, τ₁ = 5.6e4, 0.82                                           # Pa, s
μ₂, τ₂ = 3.4e4, 10.7                                           # Pa, s
ε = 1.0                                                        # scaled permittivity
viscoelastic = GeneralizedMaxwell(NeoHookean3D(λ=λ, μ=μ∞), ViscousPolyconvex(μ=μ₁, τ=τ₁), ViscousPolyconvex(μ=μ₂, τ=τ₂))
physmodel = ElectroMechModel(mechano=viscoelastic, electro=IdealDielectric(ε=ε))
km = Kinematics(Mechano, Solid)
ke = Kinematics(Electro, Solid)
k = (km, ke)
F, _, _ = get_Kinematics(km)
E = get_Kinematics(ke)

# ## Boundary conditions
# The clamp is constant. The potential of the mid-plane electrode follows `EvolutionFunctions.ramp(t_r)`,
# ``\min(t/t_r, 1)``: in a `TransientModel` the evolution functions receive the time.
V0 = 0.008                                                     # voltage of the mid-plane electrode (scaled)
t_r = 1.0                                                      # s, duration of the ramp
Du = BoundaryConditions(BoundaryCondition("fixed", [0.0, 0.0, 0.0]))
Dφ = BoundaryConditions(BoundaryCondition("bottom", 0.0), BoundaryCondition("mid", V0, EvolutionFunctions.ramp(t_r)))
D_bc = MultiFieldBC([Du, Dφ])

# ## Finite element spaces
order = 2
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)
Vu = TestFESpace(model, reffeu, D_bc[1], conformity=:H1)
Vφ = TestFESpace(model, reffeφ, D_bc[2], conformity=:H1)
V = MultiFieldFESpace([Vu, Vφ])
U = MultiFieldFESpace([TrialFESpace(Vu, D_bc[1], 0.0), TrialFESpace(Vφ, D_bc[2], 0.0)])
Ω = Triangulation(model)
dΩ = Measure(Ω, 2 * order)
println("free dofs: ", num_free_dofs(U), " (u: ", num_free_dofs(U[1]), ", φ: ", num_free_dofs(U[2]), ")")

# ## Internal variables
# `CellState(physmodel, dΩ)` creates one internal variable per viscous branch at every quadrature point,
# initialised to ``\boldsymbol C_{v,a} = \boldsymbol I`` (the plate starts relaxed). With `backend=:compiled` they
# are stored as dense arrays that the compiled kernels read directly, and their update runs as a compiled kernel
# too. `A` is a tuple with one state per branch.
A = CellState(physmodel, dΩ; backend=:compiled)

# ## Weak form
# A `TransientModel` passes to the forms a time context `ctx`: the time ``t_{n+1}`` (`ctx.t`), the step
# (`ctx.Δt`) and the state of the previous step (`ctx.u⁻`; `ctx[1].u⁻` is its first field, the displacement).
# With the `QuasiStatic` scheme the unknown of the step is the field itself, so the forms are those of a static
# problem with ``\boldsymbol F_n`` and the internal variables as extra arguments. HyperFEM's helpers build the
# terms of tutorial 04 with those extra arguments: the mechanical and the electrical residual, and the
# mechanical, electrical and coupling blocks of the tangent (`residual(physmodel, Mechano, k, ...)` etc.).
# The slot after `dΩ` is the load factor ``\Lambda`` of the static helpers, which the kinematics of this problem
# ignore; we pass `ctx.t` there only because the extra arguments ``\boldsymbol F_n, \boldsymbol A_1, \boldsymbol A_2``
# come after it.
# The time step ``\Delta t`` is not an argument: the model reads it from a reference that the driver sets before
# every attempt (`physics=physmodel` below), so a change of ``\Delta t`` changes no form.
res(ctx) = ((u, φ), (v, vφ)) -> begin
  Fn = F ∘ ∇(ctx[1].u⁻)'
  residual(physmodel, Mechano, k, (u, φ), v, dΩ, ctx.t, Fn, A...) +
  residual(physmodel, Electro, k, (u, φ), vφ, dΩ, ctx.t, Fn, A...)
end
jac(ctx) = ((u, φ), (du, dφ), (v, vφ)) -> begin
  Fn = F ∘ ∇(ctx[1].u⁻)'
  jacobian(physmodel, Mechano, k, (u, φ), du, v, dΩ, ctx.t, Fn, A...) +
  jacobian(physmodel, Electro, k, (u, φ), dφ, vφ, dΩ, ctx.t, Fn, A...) +
  jacobian(physmodel, ElectroMechano, k, (u, φ), (du, dφ), (v, vφ), dΩ, ctx.t, Fn, A...)
end

# The update of the internal variables, called by the driver once per **accepted** step with the new state
# `xh` and the state of the previous step `xh⁻` (a failed attempt leaves them untouched). The arguments are
# those of the energy: ``\boldsymbol F_{n+1}``, ``\boldsymbol E_{0,n+1}`` and ``\boldsymbol F_n`` (the electric field is
# not used by the update, but it is part of the call: without it, HyperFEM 0.3.0 does nothing and does not complain,
# and the internal variables keep their initial value, which turns the viscous branches into elastic springs).
internal(xh, xh⁻) = update_state!(physmodel, A, F ∘ ∇(xh[1])', E ∘ ∇(xh[2]), F ∘ ∇(xh⁻[1])'; backend=:compiled)

# ## Solver and computational model
# `TransientModel` with `scheme=QuasiStatic()` is the time loop of a problem without inertia. `physics` receives
# `update_time_step!(physics, Δt)` before every attempt; `internal` is the update above. With `backend=:compiled`
# the forms are compiled once; at every step they are only rebound to the new previous state and internal
# variables (we check it at the end).
nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-9, verbose=1)
creep = TransientModel(res, jac, U, V, D_bc; scheme=QuasiStatic(), nls=nls, physics=physmodel, internal=internal,
                       backend=:compiled)

# ## Post-processing
# After every accepted step: the time, the voltage, the mean deflection ``u_z`` of the free end and the wall-clock
# time (to measure the cost of a step); the VTK files every few steps. The post-processing runs before the update
# of the internal variables.
Γtip = BoundaryTriangulation(model, tags="tip")
dΓtip = Measure(Γtip, 2 * order)
ez = VectorValue(0.0, 0.0, 1.0)
ramp_voltage = EvolutionFunctions.ramp(t_r)
times, voltage, tip_uz, wall = Float64[], Float64[], Float64[], Float64[]
function record(post)
  t = post.Λ[post.iter]
  xh = get_state(creep)
  push!(times, t)
  push!(voltage, V0 * ramp_voltage(t))
  push!(tip_uz, sum(∫(xh[1] ⋅ ez)dΓtip) / (W * H))
  push!(wall, time())
  if post.iter % 5 == 1
    get_pvd(post)[t] = createvtk(Ω, joinpath(outdir, "creep_$(post.iter).vtu"), cellfields=["u" => xh[1], "φ" => xh[2]])
  end
end
post = PostProcessor(creep, record; is_vtk=true, filepath=outdir)

# ## Solve
# The time scales go from the ramp and ``\tau_1`` (about a second) to ``\lambda_2`` (40 s), and the hold must last
# several times ``\lambda_2``. Equal steps small enough for the ramp would need thousands of steps;
# `TimeStepping` grows the step by the factor `grow` after every step that converged easily, up to `Δtmax`, and
# `tstops` makes a step end exactly at the end of the ramp. The voltage increments of the ramp are small and the
# response is nearly linear in ``V^2``, so Newton's method converges without the Dirichlet projection of tutorial
# 02 (`ProjectDirichlet=true` gives the same results with fewer iterations, but compiles and factorises a
# second operator: here it costs more than it saves).
λ₁, λ₂ = let b = μ∞ * (τ₁ + τ₂) + μ₁ * τ₁ + μ₂ * τ₂, c = τ₁ * τ₂ * (μ∞ + μ₁ + μ₂)   # retardation times (above)
  sort([(b - sqrt(b^2 - 4μ∞ * c)) / (2μ∞), (b + sqrt(b^2 - 4μ∞ * c)) / (2μ∞)])
end
@printf("relaxation times τ = %.2f s, %.2f s; retardation (creep) times λ = %.2f s, %.2f s\n", τ₁, τ₂, λ₁, λ₂)
t_end = FAST ? 250.0 : 400.0                                   # s: about 6 λ₂ (FAST) and 10 λ₂
Δtmax = 10.0
stepping = FAST ? TimeStepping(; Δt=0.1, t_end, grow=1.3, Δtmax, tstops=(t_r,), maxbisec=4) :
                  TimeStepping(; Δt=0.05, t_end, grow=1.1, Δtmax, tstops=(t_r,), maxbisec=4)
HyperFEMKernels.reset_counters!()
t_solve = @elapsed x, flag = solve!(creep; stepping, post=post)
work = HyperFEMKernels.counters()
nsteps = length(times)
t_step = sort(diff(wall))[cld(nsteps - 1, 2)]                  # median wall-clock time of a step (the first compiles)
@printf("%d steps in %.1f s (%.1f ms per step, median)\n", nsteps, t_solve, 1e3 * t_step)

# ## Long-time limit: the equilibrium branch alone
# A static electro-mechanical problem (tutorial 04) with only the neo-Hookean branch, at the final voltage, on the
# same mesh: the deflection that the creep tends to. It has its own conditions (the voltage follows the load
# factor) and trial spaces; the test spaces are the same.
equilibrium = ElectroMechModel(mechano=NeoHookean3D(λ=λ, μ=μ∞), electro=IdealDielectric(ε=ε))
Dφ_eq = BoundaryConditions(BoundaryCondition("bottom", 0.0), BoundaryCondition("mid", V0, Λ -> Λ))
D_eq = MultiFieldBC([Du, Dφ_eq])
U_eq = MultiFieldFESpace([TrialFESpace(Vu, D_eq[1], 0.0), TrialFESpace(Vφ, D_eq[2], 0.0)])
res_eq(Λ) = (x, y) -> residual(equilibrium, k, x, y, dΩ)
jac_eq(Λ) = (x, dx, y) -> jacobian(equilibrium, k, x, dx, y, dΩ)
static = StaticNonlinearModel(res_eq, jac_eq, U_eq, V, D_eq; nls=NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-9),
                              backend=:compiled)
_, flag_eq = solve!(static; stepping=LoadStepping(nsteps=10, maxbisec=4), ProjectDirichlet=true)
tip_eq = sum(∫(get_state(static)[1] ⋅ ez)dΓtip) / (W * H)

# ## Results
# The deflection at the end of the ramp, and then along the hold, against the long-time limit.
i_r = findfirst(t -> t ≥ t_r - 1e-12, times)                  # the step that ends the ramp
for i in unique([1:min(nsteps, 3); i_r; i_r+3:5:nsteps; nsteps])
  @printf("t = %8.3f s   V = %.4f   tip u_z = %.4f mm   (%.1f %% of the long-time value)\n",
          times[i], voltage[i], 1e3 * tip_uz[i], 100 * tip_uz[i] / tip_eq)
end
@printf("equilibrium branch alone at V0: tip u_z = %.4f mm\n", 1e3 * tip_eq)

# ## Checks
# **Long-time limit.** After the hold, the deflection is that of the equilibrium branch alone. What is left of
# the viscous stresses decays like ``e^{-t/\lambda_2}`` (a little more slowly with the backward-Euler update and
# large steps): after ``t_{end}`` it is below 0.5 % (FAST) and 0.05 %.
rel_limit = abs(tip_uz[end] - tip_eq) / abs(tip_eq)
println("long-time limit: |u_z(t_end) − u_z,eq| / |u_z,eq| = $(rel_limit)")
@assert flag == 1 && flag_eq == 1 "a time loop or the load stepping did not finish"
@assert rel_limit < (FAST ? 5e-3 : 5e-4) "the deflection after the hold is not the equilibrium deflection ($(rel_limit))"

# **Creep time.** During the hold the distance to the limit, ``r = u_{z,eq} - u_z``, decays with the longest
# retardation time. With steps of constant size ``\Delta t`` the backward-Euler update multiplies it by
# ``1/(1 + \Delta t/\lambda)`` per step, so ten steps of size `Δtmax` give a measured ``\lambda``, which we compare
# with ``\lambda_2`` of the small-strain model (5 %: the deflection is about a tenth of the length, and the
# electric load changes a little with the deformation).
r = tip_eq .- tip_uz
iconst = findall(d -> abs(d - Δtmax) < 1e-9, diff(times)) .+ 1   # steps of size Δtmax
ia = first(iconst)
ib = min(ia + 10, last(iconst))
λ_measured = Δtmax / ((r[ia] / r[ib])^(1 / (ib - ia)) - 1)
@printf("creep time measured between t = %.1f s and %.1f s: %.2f s (small-strain λ₂ = %.2f s)\n",
        times[ia], times[ib], λ_measured, λ₂)
@assert ib - ia ≥ 5 && times[ib] - times[ia] ≥ 50 "the hold is too short to measure the creep time"
@assert isapprox(λ_measured, λ₂; rtol=0.05) "creep time $(λ_measured) s away from λ₂ = $(λ₂) s"

# **Creep and the instantaneous response.** After the ramp the voltage is constant and the deflection grows
# monotonically towards the limit, from below. The compliance ``u_z/V^2``, relative to that of the equilibrium
# branch, grows during the whole run, ramp included: the plate is stiffest at the start, where the viscous
# branches have not relaxed and the response is close to the instantaneous one, ``\mu_\infty/(\mu_\infty +
# \mu_1 + \mu_2) = 0.135`` in the small-strain model (the first step already lets them relax a little).
hold = findfirst(t -> t ≥ t_r - 1e-12, times):nsteps
compliance = (tip_uz ./ voltage .^ 2) ./ (tip_eq / V0^2)
c_inst = μ∞ / (μ∞ + μ₁ + μ₂)
@printf("relative compliance: %.3f after the first step, %.3f at the end of the ramp (instantaneous %.3f)\n",
        compliance[1], compliance[first(hold)], c_inst)
@assert tip_uz[end] > 0 "the plate does not bend upwards"
@assert all(diff(tip_uz[hold]) .> 0) "the deflection does not grow monotonically during the hold"
@assert all(tip_uz[hold] .< tip_eq) "the deflection goes beyond the long-time limit"
@assert all(diff(compliance) .> 0) "the compliance does not grow in time"
@assert compliance[1] < 1.25 * c_inst "the response of the first step is not close to the instantaneous one"

# **Efficiency.** `HyperFEMKernels.work_summary` lists the groups of integrals of the compiled operator, the path
# each takes (`:linear_maps`: the coefficients come from linear maps of the state inputs, the fast path) and whether
# a part falls back to Gridap. The counters since `reset_counters!()` show that the forms were only rebound to the
# new previous state and internal variables at every step (`update_fast`, one per step) and never compiled again
# (`update_full`). The forms were generated when the model was built; during the loop the functions generated
# (`codegen`) are two per step: the update of the internal variables of each branch is generated again at every
# call from the same expression, which finds the compiled code already there (about 1 ms per step, against tens
# of milliseconds for the Newton iterations). HyperFEM 0.3.0 has no
# accessor for the operator of a `TransientModel`: `creep.caches.op.inner` is an internal field and may change.
ws = HyperFEMKernels.work_summary(creep.caches.op.inner)
foreach(g -> println("  ", g.kind, " group ", g.group, ": ", g.cells, " cells × ", g.points, " points, path ", g.path), ws.groups)
println("  Gridap fallbacks: residual $(ws.fallback_residual), Jacobian $(ws.fallback_jacobian); ",
        "updates: $(work.update_fast) fast, $(work.update_full) full; functions generated: $(work.codegen)")
@assert ws.fallback_residual == 0 && ws.fallback_jacobian == 0 "part of the forms is assembled by Gridap"
@assert work.update_full == 0 "the compiled forms were compiled again during the time loop"
@assert work.update_fast == nsteps "the compiled forms were not rebound once per step"

# **Regression** (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0, macOS arm64): deflection at
# the end of the ramp and at the end of the hold, and the equilibrium deflection.
println("regression: tip u_z(t_r) = $(tip_uz[i_r]), u_z(t_end) = $(tip_uz[end]), u_z,eq = $(tip_eq), steps = $(nsteps)")
tipr_ref, tipend_ref, tipeq_ref, nsteps_ref = FAST ?
  (0.0009741162955541829, 0.00543470810431621, 0.005448188282398284, 41) :
  (0.000979340936078761, 0.005767556235774467, 0.00576801014451426, 87)
@assert nsteps == nsteps_ref "$(nsteps) time steps instead of $(nsteps_ref)"
@assert isapprox(tip_uz[i_r], tipr_ref; rtol=1e-5) "tip deflection at the end of the ramp $(tip_uz[i_r]) differs from $(tipr_ref)"
@assert isapprox(tip_uz[end], tipend_ref; rtol=1e-5) "tip deflection at t_end $(tip_uz[end]) differs from $(tipend_ref)"
@assert isapprox(tip_eq, tipeq_ref; rtol=1e-5) "equilibrium tip deflection $(tip_eq) differs from $(tipeq_ref)"

# ## Going further
# - Hold the voltage for a short time and switch it off (`BoundaryCondition("mid", V0, t -> …)` with any function of
#   time): the plate does not come back at once, it recovers with the same relaxation times.
# - Raise ``V_0``: the equilibrium deflection grows as ``V_0^2`` up to ``V_0 \approx 0.012`` (about 13 mm), and beyond
#   that the plate curls up as it creeps (with ``V_0 = 0.02`` the tip of the equilibrium plate rises by about 30 mm,
#   60 % of its length). More voltage steps (a smaller `Δt` or `grow`) may be needed for the large rotations.
# - Add a third branch with a long relaxation time (the HyperFEM test of a viscoelastic bar uses
#   ``\tau_3 = 500`` s), or use `ViscousIncompressible(NeoHookean3D(...); τ)` branches.
# - Solve the two fields one after the other (`StaggeredModel`, tutorial 09) instead of together.
# - References: Reese & Govindjee, *Int. J. Solids Struct.* 35 (1998) 3455–3482, doi:10.1016/S0020-7683(97)00217-5
#   (finite viscoelasticity with internal variables); Hossain, Vu & Steinmann, *Comput. Mater. Sci.* 59 (2012) 65–74,
#   doi:10.1016/j.commatsci.2012.02.027 (experiments and a generalised Maxwell model of the acrylic elastomer VHB 4910).
