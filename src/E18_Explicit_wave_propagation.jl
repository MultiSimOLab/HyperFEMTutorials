# # Tutorial 18: Explicit dynamics: wave propagation in a bar
#
# A short pressure pulse hits one end of a long bar. A compression wave runs along the bar at the speed of
# sound of the material, reflects at the free far end as a tension wave and throws the end forward. Waves
# like this one are the domain of explicit time integration: every step costs one evaluation of the internal
# forces and a division by a diagonal mass, with no tangent and no Newton iteration, but the step must stay
# below the time a wave needs to cross one element. The tutorial measures that limit, shows what happens
# above it, and compares the explicit scheme with an implicit one on the same problem.
#
# Measured on a Mac (M5 Max, 4 Julia threads, 4 BLAS threads, Julia 1.13.0, HyperFEM 0.3.0, HyperFEMKernels
# 0.2.6): FAST (404 dofs, 223 explicit steps) 36 s, full (1 604 dofs, 889 explicit steps) 40–42 s, including
# compilation; after it, 1.3 ms per explicit step (of which 0.03–0.09 ms are the internal forces) and 2–3 ms
# (FAST) or 3.9 ms (full) per implicit step.
#
# **What you will learn**
# - `TransientModel` with the explicit `CentralDifference()` scheme and a lumped (diagonal) mass
#   (`mass_lumping=true`);
# - the critical time step ``\Delta t_{\mathrm{crit}} = 2/\omega_{\max}``: its estimate ``h/c`` and its value from
#   the largest eigenvalue of the mesh, and what a step above it does;
# - a time-dependent traction (`NeumannConditions` with a load function of time) and the energy balance of
#   the run: work of the load = kinetic + stored energy;
# - explicit against implicit (`Newmark()`): accuracy (dispersion) and cost per step;
# - the compiled backend in an explicit loop, and how to check that no matrix is assembled at each step.
#
# ## Problem statement
# Find the displacement ``\boldsymbol u(\boldsymbol X, t)`` of the bar ``\Omega = (0,L)\times(0,W)\times(0,H)``
# such that
# ```math
# \int_\Omega \rho\,\ddot{\boldsymbol u}\cdot\boldsymbol v \,\mathrm dV
# + \int_\Omega \boldsymbol P(\boldsymbol F) : \nabla_0 \boldsymbol v \,\mathrm dV
# = \int_{\Gamma_0} \boldsymbol t(t)\cdot\boldsymbol v \,\mathrm dA \quad \forall \boldsymbol v ,
# ```
# with the compressible neo-Hookean energy of tutorial 02 and the bar at rest at ``t = 0``. The face
# ``\Gamma_0`` (``X = 0``) receives the pressure pulse
# ```math
# \boldsymbol t(t) = \sigma_0\, p(t)\, \boldsymbol e_x , \qquad
# p(t) = \sin^2(\pi t / t_p) \ \text{ for } 0 \le t \le t_p , \quad p(t) = 0 \ \text{ afterwards} ,
# ```
# the face ``X = L`` is free, and the four lateral faces are rollers (``u_y = u_z = 0``). The rollers make
# the motion one-dimensional (uniaxial strain), so the wave travels at the **dilatational** (P-wave) speed
# ```math
# c = \sqrt{\frac{\lambda + 2\mu}{\rho}} ,
# ```
# not at the bar speed ``\sqrt{E/\rho}`` of a bar with free lateral faces (which is reached only for pulses
# much longer than the thickness, and disperses otherwise). For small strains the stress of the wave is
# ``\sigma = \rho c\, v``: the pulse travels with particle velocity ``v = \sigma_0 p/(\rho c)``, reaches the
# far end at ``t = L/c`` and reflects there as a tension pulse; the free end moves with twice the particle
# velocity, so its displacement grows by
# ```math
# u_L^{\infty} = \frac{2}{\rho c}\int_0^{t_p}\sigma_0 p(t)\,\mathrm dt = \frac{\sigma_0 t_p}{\rho c} ,
# ```
# and reaches half of it at ``t_{1/2} = L/c + t_p/2`` (the pulse is symmetric). The work of the load,
# ``W = \int_0^t\int_{\Gamma_0}\boldsymbol t\cdot\dot{\boldsymbol u}\,\mathrm dA\,\mathrm dt``, becomes kinetic
# energy ``K`` plus stored energy ``\Pi``; while the pulse acts the loaded face moves with ``v = \sigma/(\rho c)``,
# so the pulse puts in
# ```math
# W_\infty = \frac{A}{\rho c}\int_0^{t_p}\sigma_0^2 p^2\,\mathrm dt = \frac{3\sigma_0^2 t_p A}{8\rho c} , \qquad A = WH .
# ```
# The momentum of the bar is the impulse of the load, ``\sigma_0 t_p A/2``.
#
# The data are dimensionless (any consistent set of units): ``L = 10``, ``W = H = 1``, ``\mu = 1``,
# ``\lambda = 1.5`` (``\nu = 0.3``), ``\rho = 1``, so ``c = \sqrt{3.5} = 1.871``; the pulse lasts
# ``t_p = 0.2\,L/c`` (it is ``0.2\,L = 2`` long) and its amplitude ``\sigma_0 = 10^{-3}(\lambda + 2\mu)`` gives
# strains of ``10^{-3}``: the material is nonlinear, the response is linear in practice.
#
# ## Setup
using HyperFEM
using HyperFEMKernels                                          # compiled backend (`backend=:compiled`)
using Gridap
using LinearAlgebra
using Printf

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E18_Explicit_wave_propagation")
setupfolder(outdir)                                            # VTK output (not versioned)

# ## Geometry and mesh
# One element across the section: with the rollers on the lateral faces the problem is one-dimensional,
# and the 3D mesh only carries it. ``h = L/N`` is the element length along the bar.
const L, W, H = 10.0, 1.0, 1.0
N = FAST ? 100 : 400
h = L / N
geometry = CartesianDiscreteModel((0, L, 0, W, 0, H), (N, 1, 1))
labels = get_face_labeling(geometry)
add_tag_from_tags!(labels, "lateral", [CartesianTags.faceX0Z⁺; CartesianTags.faceX1Z⁺;
                                       CartesianTags.faceXY0⁺; CartesianTags.faceXY1⁺])
add_tag_from_tags!(labels, "loaded", CartesianTags.face0YZ)
add_tag_from_tags!(labels, "far", CartesianTags.face1YZ)

# ## Constitutive model and wave data
const λ, μ, ρ = 1.5, 1.0, 1.0
physmodel = NeoHookean3D(λ=λ, μ=μ)
kinematics = Kinematics(Mechano, Solid)
const c = sqrt((λ + 2μ) / ρ)                                   # dilatational wave speed
const σ0 = 1e-3 * (λ + 2μ)                                     # pulse amplitude (strain 10⁻³)
const tp = 0.2 * L / c                                         # pulse duration
const uL∞ = σ0 * tp / (ρ * c)                                  # final displacement of the far end
const W∞ = 3 * σ0^2 * tp * W * H / (8 * ρ * c)                 # work of the whole pulse
println(@sprintf("wave speed c = %.6f, pulse duration tp = %.4f, arrival L/c = %.4f, h/c = %.5f", c, tp, L / c, h / c))

# ## Boundary conditions
# The rollers are a Dirichlet condition on the components ``y, z`` (`mask`). The pulse is a Neumann
# condition: a constant vector times a function of the load parameter, which in a `TransientModel` is the
# time (`load = identity`). The function is evaluated once per stage, not at the quadrature points; it
# is built by a function so that it captures `tp` instead of reading a global.
pulse(tp) = t -> 0.0 <= t <= tp ? sin(π * t / tp)^2 : 0.0
dirichlet = BoundaryConditions(BoundaryCondition("lateral", [0.0, 0.0, 0.0]; mask=(false, true, true)))
traction = BoundaryConditions(BoundaryCondition("loaded", [σ0, 0.0, 0.0], pulse(tp)))
neumann = NeumannConditions(geometry, traction, 2)

# ## Finite element spaces and forms
# Trilinear hexahedra (Q1), the usual partner of a lumped mass in explicit dynamics: one dof per node and
# the largest stable step for a given node spacing. (Row-sum lumping also works for the tensor-product Q2
# elements of Gridap, whose row sums are all positive; it fails, with zero or negative vertex masses, for
# serendipity hexahedra and quadratic simplices.) The static forms are those of tutorial 02; `mass` is the bilinear
# inertia form, which `TransientModel` assembles once and, with `mass_lumping=true`, replaces by the
# diagonal of its row sums.
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)
Ω = Triangulation(geometry)
dΩ = Measure(Ω, 2)
V = TestFESpace(Ω, reffe, dirichlet, conformity=:H1)
U = TrialFESpace(V, dirichlet, 0.0)
println("free dofs: ", num_free_dofs(V))

res(t) = (u, v) -> residual(physmodel, kinematics, u, v, dΩ)
jac(t) = (u, du, v) -> jacobian(physmodel, kinematics, u, du, v, dΩ)
mass = (a, v) -> ∫(ρ * (a ⋅ v))dΩ
stored(u) = sum(energy(physmodel, kinematics, u, dΩ))

# The mean axial displacement of a face is a linear functional of the dofs, ``\ell^{\mathsf T}\boldsymbol x / A``:
# its vector is assembled once, and every step only takes a dot product.
ex = VectorValue(1.0, 0.0, 0.0)
face_functional(tag) = (Γ = BoundaryTriangulation(geometry, tags=tag); dΓ = Measure(Γ, 2);
                        assemble_vector(v -> ∫(v ⋅ ex)dΓ, V) ./ (W * H))
ℓ0, ℓL = face_functional("loaded"), face_functional("far")

# ## The critical time step
# The central-difference scheme is stable when ``\Delta t \le 2/\omega_{\max}``, with ``\omega_{\max}^2`` the
# largest eigenvalue of ``\boldsymbol M_L^{-1}\boldsymbol K``. For a chain of linear elements with lumped mass
# ``\omega_{\max} \approx 2c/h``, which gives the familiar Courant condition ``\Delta t \le h/c``: a wave may not
# cross more than one element per step. Here the mesh is small, so ``\omega_{\max}`` is computed exactly from
# the stiffness at the undeformed state and the lumped mass (a dense eigenvalue problem; on a large mesh one
# takes the estimate ``h/c`` with the smallest element, or a few power iterations). For this mesh the
# estimate is exact: the highest mode is the one in which neighbouring sections move in opposite directions,
# and its frequency is ``2c/h``.
uh0 = FEFunction(U, zeros(num_free_dofs(U)))
K = assemble_matrix((du, v) -> jac(0.0)(uh0, du, v), U, V)
mL = vec(sum(assemble_matrix(mass, U, V); dims=2))             # lumped mass: row sums
d = 1 ./ sqrt.(mL)
ωmax = sqrt(eigmax(Symmetric(Matrix(d .* K .* d'))))
Δtcrit = 2 / ωmax
println(@sprintf("ω_max = %.6f, Δt_crit = 2/ω_max = %.6f = %.4f h/c", ωmax, Δtcrit, Δtcrit / (h / c)))

# ## One run
# The function below builds the model for a `scheme`, a step `Δt` and a final time, and solves it. For the
# explicit scheme the stage unknown is the acceleration, ``\boldsymbol M_L\boldsymbol a_{n+1} =
# \boldsymbol f_{\mathrm{ext}}(t_{n+1}) - \boldsymbol r(\boldsymbol u_{n+1})``, with ``\boldsymbol u_{n+1}`` known from
# the previous step: the model evaluates the internal forces once and solves with the mass, which it
# factorises once for the whole run (it changes only with Δt); the Newton parameters are not used. The
# implicit run uses the consistent mass and Newton with the full tangent at every iteration.
#
# The post-processing runs after every accepted step: it records the mean displacement of the two end
# faces (two dot products) and, optionally, the stored and kinetic energies (`energies=true`, one Gridap
# integral each, for the short unstable run: the stored energy of the new state plus the kinetic energy of
# the previous step, enough to see it grow) and a VTK frame every `vtk_every` steps. Its time is measured
# apart, so that the time per step is that of the solver. The bar starts at rest and unloaded, so the
# initial acceleration is zero (`initial_acceleration=:zero`, the default).
function wave_run(scheme, Δt, t_end; lumped, energies=false, vtk_every=0, name="run")
  U = TrialFESpace(V, dirichlet, 0.0)
  nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-13, rtol=1e-10, verbose=0)
  m = TransientModel(res, jac, U, V, dirichlet; scheme, mass, mass_lumping=lumped, neumann, nls, backend=:compiled)
  uh = get_state(m)
  x = get_free_dof_values(uh)
  t, s0, sL, E = [0.0], [0.0], [0.0], [0.0]
  tpost = Ref(0.0)
  function record(post)
    tpost[] += @elapsed begin
      push!(t, post.Λ[post.iter]); push!(s0, dot(ℓ0, x)); push!(sL, dot(ℓL, x))
      energies && push!(E, stored(uh) + 0.5 * sum(mass(get_velocity(m), get_velocity(m))))   # Π(uₙ₊₁) + K(vₙ)
      if vtk_every > 0 && post.iter % vtk_every == 0
        pvd = get_pvd(post)
        pvd[t[end]] = createvtk(Ω, joinpath(outdir, "$(name)_$(post.iter).vtu"), cellfields=["u" => uh])
      end
    end
  end
  post = PostProcessor(m, record; is_vtk=vtk_every > 0, filepath=outdir)
  time = @elapsed (_, flag) = solve!(m; stepping=TimeStepping(; Δt, t_end), post)
  nsteps = length(t) - 1
  (; flag, t, s0, sL, E, m, lumped, nsteps, time, time_per_step=(time - tpost[]) / nsteps)
end

# Work of the load over the run, by the trapezoidal rule in time on the recorded displacement of the loaded
# face (the face moves rigidly: one element across), the energies and the momentum at the end, and the
# impulse of the load as both schemes see it: the trapezoidal rule on the load at the step times.
function balance(r)
  P = pulse(tp)
  Wext = sum(0.5 * σ0 * (P(r.t[k]) + P(r.t[k+1])) * W * H * (r.s0[k+1] - r.s0[k]) for k in 1:r.nsteps)
  vh = get_velocity(r.m)
  v = get_free_dof_values(vh)
  Kin = r.lumped ? 0.5 * dot(v, mL .* v) : 0.5 * sum(mass(vh, vh))
  Π = stored(get_state(r.m))
  momentum = sum(∫(ρ * (vh ⋅ ex))dΩ)
  impulse = sum(0.5 * σ0 * (P(r.t[k]) + P(r.t[k+1])) * W * H * (r.t[k+1] - r.t[k]) for k in 1:r.nsteps)
  (; Wext, Kin, Π, error=(Wext - Kin - Π) / Wext, momentum, impulse)
end

# Arrival time: the far end reaches half of its final displacement at ``t_{1/2} = L/c + t_p/2`` (linear
# interpolation between steps).
function half_time(r)
  k = findfirst(i -> r.sL[i] < uL∞ / 2 <= r.sL[i+1], 1:length(r.sL)-1)
  r.t[k] + (uL∞ / 2 - r.sL[k]) * (r.t[k+1] - r.t[k]) / (r.sL[k+1] - r.sL[k])
end

# ## Above the stability limit
# A step 10 % above the limit. Nothing fails at once: the scheme is explicit, there is no iteration to
# diverge. But the highest mode of the mesh, which the sudden start of the load excites a little, is
# amplified at every step: by 2.43 in amplitude while the motion is small (the root of
# ``\lambda^2 - (2 - \omega_{\max}^2\Delta t^2)\lambda + 1 = 0`` with ``\omega_{\max}\Delta t = 2.2``), and much
# faster once the strains are large and the neo-Hookean material stiffens. Within twenty steps the energy is
# many orders of magnitude above the energy of the whole pulse, and soon it overflows to `Inf` (nothing
# throws: the run simply produces garbage, which is why the limit must be respected by construction).
println("\n── Central differences, Δt = 1.1 Δt_crit ──")
unstable = wave_run(CentralDifference(), 1.1 * Δtcrit, 60 * 1.1 * Δtcrit; lumped=true, energies=true)
Eu = unstable.E ./ W∞                                          # energy / energy of the whole pulse
for k in 11:10:length(Eu)
  println(@sprintf("  step %2d: energy / energy of the whole pulse = %.2e", k - 1, Eu[k]))
end

# ## Explicit run
# Central differences at 90 % of the limit, until the wave has come back to the loaded end (``t = 2L/c``).
println("\n── Central differences, Δt = 0.9 Δt_crit, lumped mass ──")
HyperFEMKernels.reset_counters!()
Δtex = 0.9 * Δtcrit
ex_run = wave_run(CentralDifference(), Δtex, 2L / c; lumped=true, vtk_every=FAST ? 0 : 80, name="explicit")
work_ex = HyperFEMKernels.counters()

# ## Implicit run
# The trapezoidal rule (`Newmark()`) is unconditionally stable: here it takes a step five times larger than
# the explicit one (4.5 times the Courant limit). A first, two-step run compiles the implicit path, so that
# the time per step of the second one is that of the solver.
println("── Newmark (trapezoidal rule), Δt = 5 Δt_explicit, consistent mass ──")
Δtim = 5 * Δtex
wave_run(Newmark(), Δtim, 2Δtim; lumped=false)
im_run = wave_run(Newmark(), Δtim, 2L / c; lumped=false)

# ## Efficiency
# `work_summary` of the compiled operators (internal fields of the model: HyperFEM 0.3.0 has no accessor
# for them, so these lines may change): the volume groups of the stage forms and the mass form take the
# fast `:linear_maps` path and nothing falls back to Gridap (the traction is a small boundary group on the
# general `:duals` path). The counters of the explicit run show that at every step the forms are only
# rebound to the new time of the traction (`update_fast`, one per step), never compiled again
# (`update_full = 0`).
#
# No matrix is assembled or factorised inside the explicit loop: the explicit stage has no stiffness term
# (its coefficient ``s_K`` is zero), so its tangent is the lumped mass alone, assembled and factorised once
# for the run (again only if Δt changes), and every step is one evaluation of the internal forces plus one
# solve with the factors of that diagonal matrix. HyperFEM 0.3.0 does this with the linear solver of `nls`
# (here `LUSolver`) on a matrix with the sparsity pattern of the stiffness: negligible here, but on a large
# mesh the factorisation costs as much as many steps (0.6 s for a 3D mesh of 26 000 dofs, about a hundred
# steps, measured on the same machine), so keep Δt fixed in an explicit run.
#
# The time of one evaluation of the internal forces is measured below for comparison: on a mesh this
# small most of a step is the fixed cost of rebinding the forms to the new time (about a millisecond: the
# forms are captured again, the Jacobian included, although the explicit stage does not use it); on larger
# meshes the internal forces take the larger share.
op = ex_run.m.caches.op.inner                                  # stage forms, internal field
opM = ex_run.m.caches.opM                                      # mass form, internal field
fallbacks = 0
for (what, o) in (("stage forms", op), ("mass form", opM))
  ws = HyperFEMKernels.work_summary(o)
  global fallbacks += ws.fallback_residual + ws.fallback_jacobian
  println(what, ": ", [(g.kind, g.domain, g.path) for g in ws.groups], "; Gridap fallbacks: ",
          ws.fallback_residual + ws.fallback_jacobian)
end
ML = ex_run.m.caches.op.M                                      # the mass matrix of the model, internal field
println("counters of the explicit run: ", work_ex)
println("lumped mass diagonal: ", isdiag(ML))
rint = zeros(num_free_dofs(V))
xend = get_free_dof_values(get_state(ex_run.m))
Gridap.Algebra.residual!(rint, op, xend)
t_res = @elapsed (for _ in 1:100; Gridap.Algebra.residual!(rint, op, xend); end)
println(@sprintf("one evaluation of the internal forces: %.3f ms", 10 * t_res))
for (name, r) in (("central differences", ex_run), ("Newmark", im_run))
  println(@sprintf("%-20s %5d steps  %.3f ms per step  %.2f s in all", name, r.nsteps, 1e3 * r.time_per_step,
                   r.time_per_step * r.nsteps))
end

# On this problem the explicit run costs more in all than the implicit one (about 0.3 s against 0.1 s in the
# FAST case, 1.2 s against 0.7 s in the full one), although it is the more accurate of the two. The reason is
# the time per step: about 1.3 ms, almost all of it fixed overhead of HyperFEM 0.3.0 (the stage forms,
# residual and Jacobian, are rebound at every step, and the diagonal mass is solved with the LU factors of a
# matrix with the pattern of the stiffness), against 0.03–0.09 ms for the internal forces that an explicit
# step really needs. A 1D bar of a few hundred dofs is also the best case for the implicit scheme: its
# factorisations are tiny. Explicit integration pays off on large 3D meshes, where a factorisation of the
# stiffness costs far more than an evaluation of the internal forces, on stiff materials and short events
# (impacts, waves) that need small steps anyway for accuracy, and on strongly nonlinear problems where
# Newton struggles; and its advantage grows once the driver divides by the lumped diagonal and skips the
# unused Jacobian.

# ## Checks
# (a) The wave speed from the arrival at the far end, and the final displacement of the far end, against
# the closed forms. The explicit scheme with a lumped mass is almost free of dispersion near the stability
# limit (in one dimension the two errors cancel exactly at ``\Delta t = h/c``); the implicit run with a
# step of 4.5 ``h/c`` and a consistent mass delays and spreads the pulse.
cex = L / (half_time(ex_run) - tp / 2)
cim = L / (half_time(im_run) - tp / 2)
println(@sprintf("\nwave speed: exact %.5f, explicit %.5f (%+.3f %%), implicit %.5f (%+.3f %%)",
                 c, cex, 100 * (cex - c) / c, cim, 100 * (cim - c) / c))
println(@sprintf("far-end displacement: exact %.6e, explicit %.6e, implicit %.6e", uL∞, ex_run.sL[end], im_run.sL[end]))
@assert ex_run.flag == 1 && im_run.flag == 1 "a run did not reach its final time"
@assert abs(Δtcrit - h / c) < 1e-8 * h / c "critical step away from h/c"
@assert abs(cex - c) < 0.005 * c "explicit: wave speed away from √((λ+2μ)/ρ) ($cex)"
@assert abs(ex_run.sL[end] - uL∞) < 0.01 * uL∞ "explicit: far-end displacement away from σ₀tₚ/(ρc)"
@assert abs(cim - c) < 0.05 * c "implicit: wave speed away from √((λ+2μ)/ρ) ($cim)"

# (b) Invariants. Momentum: the internal forces add up to zero, so both schemes (whose velocity update is
# the trapezoidal rule on the accelerations) give the bar exactly the impulse of the sampled load, and that
# impulse approaches ``\sigma_0 t_p A/2`` as the pulse is resolved by more steps (the implicit run samples it
# with about 4 steps in the FAST case, 18 in the full one). Energy: the work of the load is the kinetic plus
# the stored energy, to the accuracy of the time integration: the trapezoidal rule conserves the energy of a
# linear problem exactly; central differences only up to ``O(\Delta t^2)`` (``3\cdot10^{-3}`` in the FAST run,
# ``2\cdot10^{-4}`` in the full one, whose step is four times smaller). And a step above the limit makes the energy explode.
bex, bim = balance(ex_run), balance(im_run)
impulse = σ0 * tp / 2 * W * H
println(@sprintf("energy: closed form W∞ = %.6e; explicit W = %.6e, K + Π = %.6e (error %.2e); implicit W = %.6e, K + Π = %.6e (error %.2e)",
                 W∞, bex.Wext, bex.Kin + bex.Π, bex.error, bim.Wext, bim.Kin + bim.Π, bim.error))
println(@sprintf("momentum: impulse %.6e; explicit %.6e (sampled impulse %.6e), implicit %.6e (sampled %.6e)",
                 impulse, bex.momentum, bex.impulse, bim.momentum, bim.impulse))
@assert abs(bex.Wext - W∞) < 1e-2 * W∞ "explicit: work of the load away from 3σ₀²tₚA/(8ρc)"
@assert abs(bex.error) < 1e-2 "explicit: energy balance error $(bex.error)"
@assert abs(bim.error) < 1e-3 "implicit: energy balance error $(bim.error)"
@assert abs(bex.momentum - bex.impulse) < 1e-8 * impulse "explicit: momentum ≠ impulse of the load"
@assert abs(bim.momentum - bim.impulse) < 1e-8 * impulse "implicit: momentum ≠ impulse of the load"
@assert abs(bex.momentum - impulse) < 1e-3 * impulse "explicit: momentum away from σ₀tₚA/2"
@assert unstable.flag == 1 "the unstable run did not reach its final time"
@assert Eu[21] > 1e6 && Eu[21] > 1e3 * Eu[11] "Δt above the limit did not blow up (step 10: $(Eu[11]), step 20: $(Eu[21]))"
@assert fallbacks == 0 "a part of the forms fell back to Gridap"
@assert isdiag(ML) "the lumped mass is not diagonal"
@assert work_ex.update_full == 0 && work_ex.update_fast == ex_run.nsteps "the explicit loop compiled forms again"

# (c) Regression values (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0, macOS arm64).
println(@sprintf("regression: Δtcrit = %.10e, t½ ex = %.10e, t½ im = %.10e, uL ex = %.10e",
                 Δtcrit, half_time(ex_run), half_time(im_run), ex_run.sL[end]))
reference = FAST ? (Δtcrit=5.3452248382e-02, t_ex=5.8812024050e+00, t_im=6.0252310222e+00, uL_ex=1.9999358513e-03) :
                   (Δtcrit=1.3363062096e-02, t_ex=5.8791509873e+00, t_im=5.9081377630e+00, uL_ex=2.0000005438e-03)
@assert isapprox(Δtcrit, reference.Δtcrit; rtol=1e-5) "regression: critical step $Δtcrit"
@assert isapprox(half_time(ex_run), reference.t_ex; rtol=1e-5) "regression: explicit arrival time $(half_time(ex_run))"
@assert isapprox(half_time(im_run), reference.t_im; rtol=1e-5) "regression: implicit arrival time $(half_time(im_run))"
@assert isapprox(ex_run.sL[end], reference.uL_ex; rtol=1e-5) "regression: explicit far-end displacement $(ex_run.sL[end])"

# ## Going further
# - Run the explicit scheme at exactly ``\Delta t = h/c``: in one dimension the lumped mass and the central
#   differences then propagate the pulse without any dispersion (their phase errors cancel), and a hair above
#   it the run blows up.
# - `mass_lumping=false` keeps the consistent mass: every step solves with it, and the limit drops to
#   ``h/(\sqrt3\,c)`` (compute ``\omega_{\max}`` with the consistent mass to see it).
# - Free the lateral faces (remove the rollers) and use a few elements across the section: the pulse now
#   travels at about the bar speed ``\sqrt{E/\rho}`` and disperses (Pochhammer–Chree); the critical step is
#   set by the smallest element and the dilatational speed.
# - Raise the amplitude (``\sigma_0 = 0.1(\lambda + 2\mu)``): the speed of the neo-Hookean material depends on
#   the strain, and the pulse changes shape as it travels; the explicit step needs no change, the implicit
#   one needs more Newton iterations.
# - Tutorial 15 treats the implicit schemes (`GeneralizedAlpha`, `AdaptiveStepping`) and the energy of a
#   free vibration.
# - References: R. Courant, K. Friedrichs, H. Lewy (1928), *Über die partiellen Differenzengleichungen der
#   mathematischen Physik*, Math. Ann. 100, 32–74, doi:10.1007/BF01448839; T. Belytschko, W. K. Liu,
#   B. Moran, K. Elkhodary (2014), *Nonlinear Finite Elements for Continua and Structures*, 2nd ed., Wiley,
#   ch. 6 (explicit time integration and its stability).
