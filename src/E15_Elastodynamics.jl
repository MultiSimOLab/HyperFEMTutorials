# # Tutorial 15: Elastodynamics: free vibration of a cantilever
#
# A slender neo-Hookean cantilever is set in motion with an initial velocity that has the shape of its first
# bending mode, and vibrates freely. With a small amplitude the motion is a single harmonic whose period is
# given by beam theory, and the time schemes can be told apart by what they do with the energy: the
# trapezoidal rule and the midpoint rule keep it, the generalised-α scheme removes it in a controlled way.
# With a large amplitude the response is geometrically nonlinear, and an adaptive time step follows it.
#
# Measured on a Mac (M5 Max, 4 Julia threads, 4 BLAS threads, Julia 1.13.0, HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8): FAST (540 dofs,
# four runs of 50–100 steps) 38 s, full (3 000 dofs, 100 steps per small-amplitude run) 51 s, including
# compilation; after the first run, 7 ms (FAST) and 30 ms (full) per time step with two Newton iterations.
#
# **What you will learn**
# - `TransientModel` at its structured level: the static forms of `StaticNonlinearModel` plus a `mass` form,
#   an initial velocity `vh` and a consistent initial acceleration (`initial_acceleration=:solve`);
# - the second-order schemes `Newmark()`, `Midpoint()` and `GeneralizedAlpha(ρ∞)`, and what each does with
#   the energy;
# - fixed steps with `TimeStepping` and error-controlled steps with `AdaptiveStepping`;
# - the energies of the run: `energy=`, `get_energies`, `get_velocity`, `get_previous_state`;
# - the compiled backend in a time loop, and how to check which path its kernels take.
#
# ## Problem statement
# Find the displacement ``\boldsymbol u(\boldsymbol X, t)`` of the beam
# ``\Omega = (0,L)\times(0,W)\times(0,H)`` such that
# ```math
# \int_\Omega \rho\,\ddot{\boldsymbol u}\cdot\boldsymbol v \,\mathrm dV
# + \int_\Omega \boldsymbol P(\boldsymbol F) : \nabla_0 \boldsymbol v \,\mathrm dV = 0
# \quad \forall \boldsymbol v , \qquad \boldsymbol F = \boldsymbol I + \nabla_0 \boldsymbol u ,
# ```
# with the compressible neo-Hookean energy of tutorial 02,
# ``\Psi = \tfrac{\mu}{2}(\boldsymbol F:\boldsymbol F - 3) - \mu \ln J + \tfrac{\lambda}{2}(J-1)^2``,
# the face ``x = 0`` clamped and the rest free of traction. The beam starts undeformed,
# ``\boldsymbol u(\cdot, 0) = \boldsymbol 0``, with the velocity
# ``\dot{\boldsymbol u}(\boldsymbol X, 0) = V_0\,\phi_1(X)/\phi_1(L)\,\boldsymbol e_z``, where
# ```math
# \phi_1(x) = \cosh\beta_1 x - \cos\beta_1 x - \sigma_1(\sinh\beta_1 x - \sin\beta_1 x) , \qquad
# \sigma_1 = \frac{\cosh\beta_1 L + \cos\beta_1 L}{\sinh\beta_1 L + \sin\beta_1 L} , \qquad \beta_1 L = 1.8751\ldots
# ```
# is the first bending mode of an Euler–Bernoulli cantilever. Its angular frequency is
# ```math
# \omega_1 = (\beta_1 L)^2 \sqrt{\frac{E I}{\rho A L^4}} , \qquad I = \frac{W H^3}{12} , \quad A = W H ,
# \quad E = \frac{\mu(3\lambda + 2\mu)}{\lambda + \mu} ,
# ```
# so with a small ``V_0`` the tip moves as ``u_z(L, t) \approx (V_0/\omega_1)\sin\omega_1 t`` and the period
# is ``T_1 = 2\pi/\omega_1``. The total energy ``K + \Pi``, with ``K = \tfrac12\int_\Omega\rho\,\dot{\boldsymbol u}
# \cdot\dot{\boldsymbol u}\,\mathrm dV`` and ``\Pi = \int_\Omega\Psi\,\mathrm dV``, is constant in time.
#
# The data are dimensionless (any consistent set of units): ``L = 10``, ``W = H = 1``, ``\mu = 1``,
# ``\lambda = 1.5`` (``\nu = 0.3``, ``E = 2.6``), ``\rho = 1``.
#
# ## Setup
using HyperFEM
using HyperFEMKernels                                          # compiled backend (`backend=:compiled`)
using Gridap
using Printf

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E15_Elastodynamics")
setupfolder(outdir)                                            # VTK output (not versioned)

# ## Geometry and mesh
# A Cartesian mesh of hexahedra, ten times longer than thick. The clamped face includes its edges and
# vertices (`⁺`); the tip face is used to measure the deflection.
const L, W, H = 10.0, 1.0, 1.0
partition = FAST ? (10, 1, 1) : (20, 2, 2)
geometry = CartesianDiscreteModel((0, L, 0, W, 0, H), partition)
labels = get_face_labeling(geometry)
add_tag_from_tags!(labels, "clamped", CartesianTags.face0YZ⁺)
add_tag_from_tags!(labels, "tip", CartesianTags.face1YZ)

# ## Constitutive model
const λ, μ, ρ = 1.5, 1.0, 1.0
physmodel = NeoHookean3D(λ=λ, μ=μ)
kinematics = Kinematics(Mechano, Solid)

# Beam theory: the first bending frequency and the mode shape. The mode shape is evaluated at the nodes
# (it becomes the initial velocity), so it reads only `const` globals and its argument.
const E = μ * (3λ + 2μ) / (λ + μ)
const β₁L = 1.8751040687119611
const ω₁ = β₁L^2 * sqrt(E * (W * H^3 / 12) / (ρ * W * H * L^4))
const T₁ = 2π / ω₁
const σ₁ = (cosh(β₁L) + cos(β₁L)) / (sinh(β₁L) + sin(β₁L))
φ₁(x) = (b = β₁L * x / L; cosh(b) - cos(b) - σ₁ * (sinh(b) - sin(b)))
mode_velocity(V0) = X -> VectorValue(0.0, 0.0, V0 * φ₁(X[1]) / φ₁(L))   # closure: V0 is captured
println(@sprintf("Euler–Bernoulli: ω₁ = %.6e, T₁ = %.4f", ω₁, T₁))

# ## Boundary conditions, finite element spaces and forms
# Quadratic hexahedra (Q2): trilinear ones lock in bending and give a beam far too stiff. The static forms
# are those of tutorial 02; the inertia comes from the bilinear `mass` form, which the model assembles once
# into a mass matrix. `energy(t)` is the stored energy ``\Pi``: the model reports it with the kinetic energy.
bc = BoundaryConditions(BoundaryCondition("clamped", [0.0, 0.0, 0.0]))
order = 2
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
Ω = Triangulation(geometry)
dΩ = Measure(Ω, 2 * order)
Γtip = BoundaryTriangulation(geometry, tags="tip")
dΓtip = Measure(Γtip, 2 * order)
ez = VectorValue(0.0, 0.0, 1.0)

res(t) = (u, v) -> residual(physmodel, kinematics, u, v, dΩ)
jac(t) = (u, du, v) -> jacobian(physmodel, kinematics, u, du, v, dΩ)
stored(t) = u -> energy(physmodel, kinematics, u, dΩ)
mass = (a, v) -> ∫(ρ * (a ⋅ v))dΩ

println("free dofs: ", num_free_dofs(TestFESpace(Ω, reffe, bc, conformity=:H1)))

tip(uh) = sum(∫(uh ⋅ ez)dΓtip) / (W * H)                      # mean vertical displacement of the tip face
kinetic(vh) = 0.5 * sum(mass(vh, vh))

# ## One free-vibration run
# The function below builds the model for a time `scheme`, an initial tip velocity `V0` and a `stepping`,
# and solves it. `vh` is the initial velocity; `initial_acceleration=:solve` takes the initial acceleration
# from ``\boldsymbol M \boldsymbol a_0 = -\boldsymbol r(\boldsymbol u_0)`` (zero here, as the beam starts
# undeformed, but it is the consistent choice and costs one solve).
#
# The post-processing runs after every accepted step, before the scheme commits it: the state is already
# ``\boldsymbol u_{n+1}``, while `get_previous_state` and `get_velocity` still give ``\boldsymbol u_n`` and
# ``\boldsymbol v_n``. The energies are therefore recorded one step behind, and the last ones are taken after
# the solve. With `verbose=1` the solver prints one line per step with ``K``, ``\Pi`` and ``K + \Pi``, and
# `get_energies` returns the last values (they are tracked only when the solver is verbose). The three
# integrals of the post-processing are evaluated by Gridap: about 1 ms per step, a few per cent of a step.
#
# `backend=:compiled` evaluates the static forms and the mass form with the compiled, threaded kernels of
# HyperFEMKernels. The forms are built once; at every stage the model only rebinds them to the new time
# and state (no recompilation), and the mass matrix is assembled once. The problem is nonlinear, so the
# tangent is updated at every Newton iteration (`tangent_update=:iteration`, the default); a linear
# problem would use `:Δt` (factorise only when the step changes).
function free_vibration(scheme, V0, stepping; verbose=0, vtk=false, name="run")
  V = TestFESpace(Ω, reffe, bc, conformity=:H1)
  U = TrialFESpace(V, bc, 0.0)
  vh = interpolate(mode_velocity(V0), U)
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-12, rtol=1e-10, verbose=verbose)
  m = TransientModel(res, jac, U, V, bc; scheme, mass, vh, initial_acceleration=:solve, energy=stored, nls,
                     backend=:compiled)
  uh = get_state(m)
  t, z = [0.0], [0.0]                                          # tip history
  te, Ke, Πe = Float64[], Float64[], Float64[]                 # energies, one step behind
  function record(post)
    push!(t, post.Λ[post.iter]); push!(z, tip(uh))
    push!(te, post.iter == 1 ? 0.0 : post.Λ[post.iter-1])
    push!(Ke, kinetic(get_velocity(m))); push!(Πe, sum(stored(0.0)(get_previous_state(m))))
    if vtk
      pvd = get_pvd(post)
      pvd[t[end]] = createvtk(Ω, joinpath(outdir, "$(name)_$(post.iter).vtu"), cellfields=["u" => uh], order=2)
    end
  end
  post = PostProcessor(m, record; is_vtk=vtk, filepath=outdir)
  time = @elapsed (_, flag) = solve!(m; stepping, post)
  push!(te, t[end]); push!(Ke, kinetic(get_velocity(m))); push!(Πe, sum(stored(0.0)(uh)))
  (; flag, t, z, te, Ke, Πe, Etot=Ke .+ Πe, m, time, nsteps=length(t) - 1)
end

# The period from the tip history: the tip starts at zero going up, crosses zero downwards near ``T_1/2``
# and upwards near ``T_1``; the crossings are located by linear interpolation.
function period(t, z)
  k = findfirst(i -> z[i] > 0 && z[i+1] <= 0, 2:length(z)-1) + 1           # downward crossing
  j = findfirst(i -> z[i] < 0 && z[i+1] >= 0, k:length(z)-1) + k - 1     # next upward crossing
  t[j] - z[j] * (t[j+1] - t[j]) / (z[j+1] - z[j])
end

# ## Solve
# Small amplitude: tip amplitude ``V_0/\omega_1 = 10^{-3} L``, so the response is linear in practice. The
# runs cover 1.25 periods (the period check needs the second zero crossing) with ``T_1/40`` (FAST) or
# ``T_1/80`` steps.
nper = FAST ? 40 : 80
Δt = T₁ / nper
t_end = 1.25 * T₁
V0small = 1e-3 * L * ω₁
println("\n── Newmark (trapezoidal rule), small amplitude ──")
nm = free_vibration(Newmark(), V0small, TimeStepping(; Δt, t_end); verbose=1)
println("\n── Midpoint rule, small amplitude ──")
HyperFEMKernels.reset_counters!()
mp = free_vibration(Midpoint(), V0small, TimeStepping(; Δt, t_end))
work_mp = HyperFEMKernels.counters()
println("── Generalised-α (ρ∞ = 0.5), small amplitude ──")
ga = free_vibration(GeneralizedAlpha(0.5), V0small, TimeStepping(; Δt, t_end))

# Large amplitude: tip amplitude ``\approx 0.25 L`` (linear estimate), one period, generalised-α with a
# little dissipation (ρ∞ = 0.8), and the step controlled by the local error estimate. The weighted error
# is measured against ``\mathrm{atol} + \mathrm{rtol}\,|u_i|`` dof by dof: `atol` must be set to the scale of
# the displacements that matter (here ``10^{-3}``, 0.04 % of the tip amplitude); with the default
# ``10^{-6}`` the dofs that barely move (the axial ones) would force steps of ``T_1/5000``.
println("── Generalised-α (ρ∞ = 0.8), large amplitude, adaptive steps ──")
V0large = 0.25 * L * ω₁
lg = free_vibration(GeneralizedAlpha(0.8), V0large, AdaptiveStepping(; Δt=T₁ / 40, t_end=T₁, rtol=1e-3, atol=1e-3);
                    vtk=true, name="large")
Δts = diff(lg.t)
println(@sprintf("adaptive run: %d steps, Δt ∈ [%.3f, %.3f] (T₁/%.0f … T₁/%.0f), max tip deflection %.3f",
                 length(Δts), minimum(Δts), maximum(Δts), T₁ / minimum(Δts), T₁ / maximum(Δts), maximum(abs.(lg.z))))

# ## Efficiency
# What the compiled operators do (`work_summary`): every group of the stage forms and of the mass form
# should take a fast path (`:linear_maps` or `:factorised`) and nothing should fall back to Gridap. The
# counters of the midpoint run (after the first run compiled everything) show that the forms are only
# rebound at every stage (`update_fast`: one per step, plus one for the initial acceleration), never compiled
# again (`update_full`). HyperFEM 0.3.0 has no
# accessor for the operators of a `TransientModel`: the two lines below read internal fields, and may change.
op = nm.m.caches.op.inner                                      # static forms (stiffness), internal field
opM = nm.m.caches.opM                                          # mass form, internal field
for (what, o) in (("stage forms", op), ("mass form", opM))
  ws = HyperFEMKernels.work_summary(o)
  println(what, ": ", [(g.kind, g.domain, g.path) for g in ws.groups], "; Gridap fallbacks: ",
          ws.fallback_residual + ws.fallback_jacobian)
end
println("counters of the midpoint run: ", work_mp)
for o in (op, opM)
  ws = HyperFEMKernels.work_summary(o)
  @assert ws.fallback_residual == 0 && ws.fallback_jacobian == 0 "part of the forms is assembled by Gridap"
  @assert all(g.path !== :duals for g in ws.groups) "a group is evaluated with dual numbers"
end
@assert work_mp.update_full == 0 && work_mp.fallback_assemblies == 0 "the compiled forms were compiled again or assembled by Gridap"
@assert work_mp.update_fast == mp.nsteps + 1 "the compiled forms were not rebound once per step (plus the initial acceleration)"
for (name, r) in (("Newmark", nm), ("midpoint", mp), ("generalised-α(0.5)", ga), ("large amplitude", lg))
  println(@sprintf("%-20s %4d steps  %.3f s per step", name, r.nsteps, r.time / r.nsteps))
end

# ## Checks
# (a) The period of the small-amplitude motion against the Euler–Bernoulli value, with a 3 % tolerance.
# The measured difference (+0.01 % FAST, +0.22 % full) adds up three effects of opposite signs: the
# model (shear deformation and rotary inertia, which Euler–Bernoulli neglects, lengthen the period by a
# relative amount of order ``(H/L)^2``, as Timoshenko beam theory shows: Han, Benaroya & Wei 1999,
# doi:10.1006/jsvi.1999.2257; the 3D clamped face, which also blocks the lateral contraction, stiffens the
# beam), the Q2 mesh (stiffer than the continuum) and the period elongation of the trapezoidal rule,
# ``(\omega_1\Delta t)^2/12``: ``2\cdot10^{-3}`` with 40 steps per period, ``5\cdot10^{-4}`` with 80.
Tnm = period(nm.t, nm.z)
Tmp = period(mp.t, mp.z)
println(@sprintf("\nperiod: Newmark %.4f, midpoint %.4f, Euler–Bernoulli %.4f (difference %.2f %%)",
                 Tnm, Tmp, T₁, 100 * (Tnm - T₁) / T₁))
@assert all(r.flag == 1 for r in (nm, mp, ga, lg)) "a run did not reach its final time"
@assert abs(Tnm - T₁) < 0.03 * T₁ "period away from beam theory (Newmark: $Tnm, T₁ = $T₁)"
@assert abs(Tmp - T₁) < 0.03 * T₁ "period away from beam theory (midpoint: $Tmp, T₁ = $T₁)"
@assert abs(maximum(nm.z) - V0small / ω₁) < 0.05 * V0small / ω₁ "tip amplitude away from V₀/ω₁"

# (b) Energy. For a linear problem the trapezoidal rule and the midpoint rule conserve the discrete energy
# exactly; at this amplitude the problem is linear up to terms of relative size ``10^{-4}``. The
# generalised-α scheme with ρ∞ < 1 dissipates. Its energy at the step points is not monotone (it
# oscillates by about ``10^{-4}`` at twice the frequency of the motion: the quantity that decreases
# monotonically is a modified energy of the scheme), but it never exceeds the initial one, and most of what
# it removes in the first steps is the high-frequency content of the initial velocity (the Euler–Bernoulli
# mode is not exactly a mode of the 3D mesh), which the trapezoidal rule keeps forever. The last energies of
# the verbose run are also those of `get_energies`.
drift(r) = maximum(abs.(r.Etot .- r.Etot[1])) / r.Etot[1]
loss(r) = 1 - r.Etot[end] / r.Etot[1]
println(@sprintf("energy: initial %.6e; relative drift Newmark %.2e, midpoint %.2e; loss generalised-α(0.5) %.2e, large amplitude %.2e",
                 nm.Etot[1], drift(nm), drift(mp), loss(ga), loss(lg)))
e = get_energies(nm.m)
@assert abs(e.K + e.Π - nm.Etot[end]) < 1e-10 * nm.Etot[1] "get_energies differs from the recorded energies"
@assert abs(nm.Etot[1] - kinetic(interpolate(mode_velocity(V0small), get_trial_space(nm.m)))) < 1e-12 * nm.Etot[1]
@assert drift(nm) < 1e-6 "Newmark: energy not conserved (drift $(drift(nm)))"
@assert drift(mp) < 1e-6 "midpoint: energy not conserved (drift $(drift(mp)))"
excess(r) = maximum(r.Etot) / r.Etot[1] - 1
@assert excess(ga) < 1e-10 && loss(ga) > 1e-3 "generalised-α(0.5): no dissipation (loss $(loss(ga)), excess $(excess(ga)))"
@assert excess(lg) < 1e-10 && loss(lg) > 0 "large amplitude: energy above the initial one (excess $(excess(lg)))"

# (c) Regression values (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0, macOS arm64).
println(@sprintf("regression: Tnm = %.10e, z_end = %.10e, loss_ga = %.10e, zmax(large) = %.10e",
                 Tnm, nm.z[end], loss(ga), maximum(lg.z)))
reference = FAST ? (Tnm=3.8396008005e+02, z_end=9.9400476022e-03, loss_ga=6.9292171059e-03, z_large=2.4515873351e+00) :
                   (Tnm=3.8476605127e+02, z_end=9.9605466481e-03, loss_ga=4.4195200638e-03, z_large=2.4580846981e+00)
@assert isapprox(Tnm, reference.Tnm; rtol=1e-5) "regression: period $Tnm"
@assert isapprox(nm.z[end], reference.z_end; rtol=1e-5) "regression: final tip deflection $(nm.z[end])"
@assert isapprox(loss(ga), reference.loss_ga; rtol=1e-5) "regression: generalised-α energy loss $(loss(ga))"
@assert isapprox(maximum(lg.z), reference.z_large; rtol=1e-5) "regression: large-amplitude maximum tip deflection $(maximum(lg.z))"

# ## Going further
# - `HHT(α)` and `Newmark(β=…, γ=…)` are other members of the generalised-α family; `Newmark(β=0.3, γ=0.6)`
#   dissipates as well, at first order.
# - `CentralDifference()` with `mass_lumping=true` is the explicit scheme of wave propagation and impact
#   problems: no Newton iterations, but a stable step below ``2/\omega_{\max}`` (the highest frequency of the
#   mesh), far smaller than the ones used here.
# - A `damping` form, e.g. Rayleigh damping `(v, w) -> ∫(c * ρ * (v ⋅ w))dΩ`, adds physical dissipation.
# - Increase the amplitude further and compare the period with the small-amplitude one, and the axial
#   displacement of the tip (it moves inwards: geometric nonlinearity). Run the large-amplitude case with
#   `Newmark()` and compare its energy with that of `GeneralizedAlpha(0.8)`.
# - Reference: J. Chung, G. M. Hulbert (1993), *A time integration algorithm for structural dynamics with
#   improved numerical dissipation: the generalized-α method*, J. Appl. Mech. 60(2), 371–375,
#   doi:10.1115/1.2900803.
