# # Tutorial 11: Transient Navier–Stokes flow around a cylinder
#
# The flow around a cylinder in a channel is the classical benchmark of incompressible flow solvers
# (Schäfer & Turek 1996). At a Reynolds number of 20 the flow is steady, with two attached vortices behind
# the cylinder; at 100 the wake becomes unstable and sheds vortices periodically (the von Kármán street).
# We march the Navier–Stokes equations in time with a second-order BDF scheme on the mesh of tutorial 10,
# reach the steady state of the case 2D-1 and compare the drag coefficient and the pressure drop with the
# published values; the full run then computes the periodic case 2D-2.
#
# Measured on a Mac (M5 Max, 4 Julia threads, 4 BLAS threads, Julia 1.13.0, HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8): 10 972 dofs; FAST (case 2D-1, 12 time steps) 67 s and full (2D-1 with both backends, then
# 2D-2 with 800 steps) 145 s, including compilation and package loading. A step of 2D-2 (assembly and LU)
# takes 0.037 s; one Jacobian assembly takes 5.6 ms with the compiled backend and 58 ms with Gridap's.
#
# **What you will learn**
# - `TransientModel` at its general level: the residual and the tangent are written with the time
#   context of the step (`∂t`, `δ∂t`, the previous states `u⁻`, `u⁻⁻`), here with `BDF(2)`;
# - `tangent_update=:step`: a problem linear in the unknown of the step (Oseen linearisation), assembled
#   and solved once per step;
# - residual-based stabilisation of equal-order elements (SUPG, PSPG, LSIC);
# - `TimeStepping`, and recording forces at every step with a `PostProcessor`;
# - assembling with the compiled backend of HyperFEMKernels (`backend=:compiled`), and checking that it is
#   faster than Gridap's assembly;
# - validation against a published benchmark.
#
# ## Problem statement
# Find the velocity ``\boldsymbol u`` and the pressure ``p`` such that
# ```math
# \rho\left(\partial_t\boldsymbol u + (\nabla\boldsymbol u)\,\boldsymbol u\right) - \nabla\cdot\boldsymbol\sigma = \boldsymbol 0 ,
# \qquad \nabla\cdot\boldsymbol u = 0 , \qquad
# \boldsymbol\sigma = -p\boldsymbol I + 2\mu\,\boldsymbol\varepsilon(\boldsymbol u) ,
# ```
# in the channel of tutorial 10, starting from rest, with the parabolic inflow
# ``\boldsymbol u = (4U_m y(H-y)/H^2, 0)`` (``H = 0.41`` m) imposed from ``t = 0``, no slip on the walls and on
# the cylinder, and ``\boldsymbol\sigma\boldsymbol n = \boldsymbol 0`` on the outlet (the natural condition
# of the symmetric form below). ``\rho = 1`` kg/m³, ``\mu = 10^{-3}`` Pa s, and the Reynolds number is
# ``Re = \bar U D/\nu`` with the mean inflow velocity ``\bar U = 2U_m/3`` and the cylinder diameter
# ``D = 0.1`` m: ``U_m = 0.3`` m/s gives ``Re = 20`` (case 2D-1) and ``U_m = 1.5`` m/s gives ``Re = 100`` (2D-2).
#
# **Time discretisation.** BDF2 replaces the time derivative at ``t_{n+1}`` by
# ``\partial_t\boldsymbol u \approx (3\boldsymbol u - 4\boldsymbol u^n + \boldsymbol u^{n-1})/(2\Delta t)``
# (backward Euler, ``(\boldsymbol u - \boldsymbol u^n)/\Delta t``, in the first step, which has no
# ``\boldsymbol u^{n-1}``). **Linearisation.** The advecting velocity of the convective term is extrapolated
# from the history, ``\boldsymbol w = 2\boldsymbol u^n - \boldsymbol u^{n-1}`` (``\boldsymbol u^n`` in the
# first step): the convective term ``(\nabla\boldsymbol u)\boldsymbol w`` becomes linear in the unknown
# (an Oseen problem), the extrapolation error is ``O(\Delta t^2)`` like that of BDF2, and every step is one
# linear solve.
#
# **Stabilisation.** Linear velocity and pressure need PSPG (tutorial 10), and convection-dominated flow
# needs SUPG (streamline upwind Petrov–Galerkin). With the residual of the momentum equation in each
# element, ``\boldsymbol R = \rho\,\partial_t\boldsymbol u + \rho(\nabla\boldsymbol u)\boldsymbol w + \nabla p``
# (the viscous term vanishes inside linear elements), the weak form of a step is
# ```math
# \begin{aligned}
# &\int_\Omega \rho\,\partial_t\boldsymbol u\cdot\boldsymbol v
# + 2\mu\,\boldsymbol\varepsilon(\boldsymbol u):\boldsymbol\varepsilon(\boldsymbol v)
# + \rho\,(\nabla\boldsymbol u)\boldsymbol w\cdot\boldsymbol v
# - p\,\nabla\cdot\boldsymbol v - q\,\nabla\cdot\boldsymbol u \,\mathrm d\Omega \\
# &\quad + \sum_K\int_K \tau_{\mathrm{SUPG}}\,(\nabla\boldsymbol v)\boldsymbol w\cdot\boldsymbol R
# - \tau_{\mathrm{PSPG}}\,\nabla q\cdot\boldsymbol R
# + \tau_{\mathrm{LSIC}}\,(\nabla\cdot\boldsymbol v)(\nabla\cdot\boldsymbol u) \,\mathrm d\Omega = 0 ,
# \end{aligned}
# ```
# with ``\tau_{\mathrm{SUPG}} = \left((2|\boldsymbol u^n|/h)^2 + 9(4\mu/(\rho h^2))^2\right)^{-1/2}``,
# ``\tau_{\mathrm{PSPG}} = \tau_{\mathrm{SUPG}}/\rho`` and ``\tau_{\mathrm{LSIC}} = \rho|\boldsymbol u^n| h/2``
# (Tezduyar 1992, DOI 10.1016/S0065-2156(08)70153-4), evaluated with the velocity of the previous step and
# the element size ``h = \sqrt{2|K|}``.
#
# ## Setup
using HyperFEM
using Gridap
using GridapGmsh
using LinearAlgebra
using HyperFEMKernels                                          # enables `backend=:compiled`
using Gridap.Algebra: allocate_jacobian

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E11_NavierStokes_transient")
setupfolder(outdir)                                            # VTK output (not versioned)

# The material constants are `const` because the stabilisation parameters below read them at every
# quadrature point (tutorial 10 explains why a function evaluated at points must read only its arguments and
# `const` globals).
const ρ = 1.0                                                  # kg/m³
const μ = 1e-3                                                 # Pa s (ν = μ/ρ = 1e-3 m²/s)
H, Dc = 0.41, 0.1                                              # m: channel height, cylinder diameter

# ## Geometry and mesh
# The DFG channel of tutorial 10 (7 402 linear triangles, generator `data/models/turek.jl`).
model = GmshDiscreteModel(joinpath(@__DIR__, "..", "data", "models", "turek2d.msh"))
Ω = Triangulation(model)
dΩ = Measure(Ω, 3)
he = CellField(map(a -> sqrt(2a), collect(get_cell_measure(Ω))), Ω)   # element size (triangles)

# ## Boundary conditions and finite element spaces
# Equal-order P1–P1. The inflow profile is a function of the position; it has no time function, so it is
# applied in full from the first step (an impulsive start). The third argument of `TrialFESpace` is the
# time at which the Dirichlet values are first evaluated: the initial time, 0 (with a time function on the
# inflow, as in tutorial 12, the value matters). The two cases of this tutorial have different
# inflow velocities, so the profile is not a function of constants as in tutorial 10: `inflow(Um, H)`
# builds it as a closure, which stores the values of `Um` and `H` it was built with (the other way to keep
# a function evaluated at points free of non-constant globals).
Um = 0.3                                                       # m/s: Re = 20
inflow(Um, H) = x -> VectorValue(4Um * x[2] * (H - x[2]) / H^2, 0.0)
D_bc = MultiFieldBC([
  BoundaryConditions(
    BoundaryCondition("inlet", inflow(Um, H)),
    BoundaryCondition("wall", [0.0, 0.0]),
    BoundaryCondition("cylinder", [0.0, 0.0])),
  NothingBC()])
reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
reffep = ReferenceFE(lagrangian, Float64, 1)
Vu = TestFESpace(model, reffeu, D_bc[1], conformity=:H1)
Vp = TestFESpace(model, reffep, D_bc[2], conformity=:H1)
V = MultiFieldFESpace([Vu, Vp])
U = MultiFieldFESpace([TrialFESpace(Vu, D_bc[1], 0.0), TrialFESpace(Vp, D_bc[2], 0.0)])
println("P1–P1: $(num_free_dofs(U)) dofs")

# ## Weak form
# At the general level of `TransientModel` the residual and the tangent are functions of the *time
# context* `ctx` of the step, which returns the forms. `ctx[1]` is the context of the velocity field:
# `∂t(cu, u)` is the BDF time derivative ``c_0\boldsymbol u - (c_1\boldsymbol u^n - c_2\boldsymbol u^{n-1})``
# of the unknown `u` (backward Euler in the first step), `δ∂t(cu, du)` its derivative in the direction
# `du` (``c_0\,\delta\boldsymbol u``), `cu.u⁻` and `cu.u⁻⁻` are ``\boldsymbol u^n`` and ``\boldsymbol u^{n-1}``,
# and `cu.step` is the number of the step. The time derivative also enters the SUPG and PSPG terms
# through the residual ``\boldsymbol R``.
τsupg(u, h) = ((2 * norm(u) / h)^2 + 9 * (4μ / (ρ * h^2))^2)^(-1 / 2)
τpspg(u, h) = τsupg(u, h) / ρ
τlsic(u, h) = ρ * norm(u) * h * 0.5
ε(∇u) = 0.5 * (∇u + ∇u')

# The advecting velocity ``\boldsymbol w`` is written as one expression for every step (the operator of the
# model is rebuilt every step and must keep its type).
function advecting(cu)
  a, b = cu.step == 1 ? (1.0, 0.0) : (2.0, 1.0)
  a * cu.u⁻ - b * cu.u⁻⁻
end
function res(ctx)
  cu = ctx[1]
  un = cu.u⁻; w = advecting(cu)
  τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he); τl = τlsic ∘ (un, he)
  ((u, p), (v, q)) -> begin
    ut = ∂t(cu, u)
    Ru = ρ * ut + ρ * (∇(u)' ⋅ w)
    ∫(ρ * (ut ⋅ v))dΩ + ∫(2μ * (ε ∘ (∇(u))) ⊙ (ε ∘ (∇(v))))dΩ + ∫(ρ * ((∇(u)' ⋅ w) ⋅ v))dΩ -
    ∫(p * (∇ ⋅ v))dΩ - ∫(q * (∇ ⋅ u))dΩ +
    ∫(τs * ((∇(v)' ⋅ w) ⋅ Ru))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ ∇(p)))dΩ -
    ∫(τp * (∇(q) ⋅ Ru))dΩ - ∫(τp * (∇(q) ⋅ ∇(p)))dΩ +
    ∫(τl * (∇ ⋅ v) * (∇ ⋅ u))dΩ
  end
end
function jac(ctx)
  cu = ctx[1]
  un = cu.u⁻; w = advecting(cu)
  τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he); τl = τlsic ∘ (un, he)
  ((u, p), (du, dp), (v, q)) -> begin
    dut = δ∂t(cu, du)
    dRu = ρ * dut + ρ * (∇(du)' ⋅ w)
    ∫(ρ * (dut ⋅ v))dΩ + ∫(2μ * (ε ∘ (∇(du))) ⊙ (ε ∘ (∇(v))))dΩ + ∫(ρ * ((∇(du)' ⋅ w) ⋅ v))dΩ -
    ∫(dp * (∇ ⋅ v))dΩ - ∫(q * (∇ ⋅ du))dΩ +
    ∫(τs * ((∇(v)' ⋅ w) ⋅ dRu))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ ∇(dp)))dΩ -
    ∫(τp * (∇(q) ⋅ dRu))dΩ - ∫(τp * (∇(q) ⋅ ∇(dp)))dΩ +
    ∫(τl * (∇ ⋅ v) * (∇ ⋅ du))dΩ
  end
end

# ## Solver and computational model
# The step problem is linear in the unknown, with a tangent that depends on the history (through ``\boldsymbol w``
# and the ``\tau``'s): `tangent_update=:step` assembles it once per step and solves once from the zero guess,
# with no Newton iterations. With 11 000 dofs a sparse LU is the right linear solver (tutorial 13 shows
# the iterative solvers for larger problems).
#
# `backend=:compiled` assembles the forms with HyperFEMKernels instead of Gridap: the weak form is compiled
# once into a kernel per element, and at every later step the model only re-captures the forms of the new
# step (the history fields and the BDF coefficients) without compiling again. The forms are written exactly
# as for Gridap; `backend=:lazy` (the default) gives the same solution, assembled by Gridap.
nls = NewtonRaphson(LUSolver(); verbose=0)
flow = TransientModel(res, jac, U, V, D_bc; scheme=BDF(2), tangent_update=:step, nls=nls, backend=:compiled)

# ## Post-processing
# The force on the cylinder, ``\boldsymbol F = -\int_{\Gamma_c}\boldsymbol\sigma\boldsymbol n\,\mathrm d\Gamma``
# (``\boldsymbol n`` the outward normal of the fluid), gives the drag and lift coefficients
# ``C_{D,L} = 2F_{x,y}/(\rho\bar U^2 D)``; ``\Delta p = p(0.15, 0.2) - p(0.25, 0.2)`` is the pressure difference
# between the front and the back of the cylinder.
Γc = BoundaryTriangulation(model, tags="cylinder")
dΓc = Measure(Γc, 3)
nc = get_normal_vector(Γc)
function coefficients(xh, Ū)
  uh, ph = xh
  σ = 2μ * (ε ∘ (∇(uh))) - ph * I2D
  F = -sum(∫(σ ⋅ nc)dΓc)
  cD, cL = 2 * F[1] / (ρ * Ū^2 * Dc), 2 * F[2] / (ρ * Ū^2 * Dc)
  Δp = ph(Point(0.15, 0.2)) - ph(Point(0.25, 0.2))
  (; cD, cL, Δp)
end

# After every step the drag coefficient is stored (`post.Λ` holds the times of the accepted steps) and the
# last state is written to VTK.
Ū = 2Um / 3
Δt, nsteps = 0.5, 12
cD_history = Float64[]
function record(post)
  xh = get_state(flow)
  push!(cD_history, coefficients(xh, Ū).cD)
  t = post.Λ[end]
  println("step $(post.iter)  t = $t s  C_D = $(cD_history[end])")
  if post.iter == nsteps
    get_pvd(post)[t] = createvtk(Ω, joinpath(outdir, "re20"), cellfields=["u" => xh[1], "p" => xh[2]])
  end
end
post = PostProcessor(flow, record; is_vtk=true, filepath=outdir)

# ## Solve: case 2D-1 (Re = 20)
# The flow starts from rest and reaches the steady state in a few seconds; the time step is large
# (``\Delta t = 0.5`` s) because only the final state matters here.
HyperFEMKernels.reset_counters!()
t2d1 = @elapsed x, flag = solve!(flow; stepping=TimeStepping(; Δt=Δt, t_end=nsteps * Δt), post=post)
re20 = coefficients(get_state(flow), Ū)
println("2D-1: C_D = $(re20.cD), C_L = $(re20.cL), Δp = $(re20.Δp)  (reference 5.5795, 0.0106, 0.1175)")
println("2D-1: $nsteps steps in $(round(t2d1, digits=2)) s (the first ones include compilation)")

# What did the compiled backend do? `HyperFEMKernels.work_summary` lists the groups of integrals of the operator
# of the last step and the path each one takes: the Jacobian of this form is `:factorised` (test factor times
# trial factor at every point, the fastest path); `:duals` would mean automatic differentiation with nested dual
# numbers. `HyperFEMKernels.counters()` counts the work done since `reset_counters!()`: `update_full` the steps
# whose forms had to be compiled again (one, during the start-up of the scheme: the first step, backward
# Euler with a single previous state, differs from the BDF2 steps that follow) and `update_fast` the steps
# that only re-captured the new history.
ws = HyperFEMKernels.work_summary(flow.caches.op.inner)
foreach(g -> println("  ", g.kind, " group ", g.group, ": ", g.cells, " cells × ", g.points, " points, path ", g.path), ws.groups)
counts = HyperFEMKernels.counters()
println("  Gridap fallbacks: residual $(ws.fallback_residual), Jacobian $(ws.fallback_jacobian); updates: $(counts.update_fast) fast, $(counts.update_full) full")

# ## Checks (2D-1)
# Reference values: ``C_D = 5.5795``, ``C_L = 0.0106``, ``\Delta p = 0.1175`` (rounded from the high-order
# computation of Nabh 1998, Heidelberg preprint 42/98, listed with the FEATFLOW benchmark results; all within
# the intervals of Schäfer & Turek 1996). The tolerances, 3 % and 2 %, are those of the test of this
# simulation in HyperFEM; the regression value ``\|\boldsymbol x\|`` is that of the same test (HyperFEM 0.3.0,
# Gridap 0.20.8, Julia 1.13.0, macOS arm64). The lift is not checked: at Re = 20 it is tiny (0.2 % of the drag),
# and the computed ``C_L = -0.037`` differs from 0.0106 by about 1 % of the force on the cylinder, the size of
# the drag error (−1.8 %); P1–P1 on this mesh does not resolve it.
#
# The compiled backend is checked here in both modes without timings: the Jacobian takes the `:factorised`
# path, nothing falls back to Gridap and the forms are not compiled again in the loop. The timing check
# (next section) runs only in the full run.
@assert flag == 1 "the time loop did not complete"
@assert length(cD_history) == nsteps "a time step was bisected"
@assert isapprox(re20.cD, 5.5795, rtol=0.03) "C_D = $(re20.cD) away from the benchmark value 5.5795"
@assert isapprox(re20.Δp, 0.1175, rtol=0.02) "Δp = $(re20.Δp) away from the benchmark value 0.1175"
@assert abs(cD_history[end] - cD_history[end-1]) < 1e-3 * cD_history[end] "the flow is not steady at t = $(nsteps * Δt) s"
@assert isapprox(norm(x), 13.67687986714, rtol=1e-5) "regression: ‖x‖ = $(norm(x))"
@assert all(g.path === :factorised for g in ws.groups if g.kind === :jacobian) "the compiled Jacobian is not factorised"
@assert ws.fallback_residual == 0 && ws.fallback_jacobian == 0 "part of the forms is assembled by Gridap"
@assert counts.update_full <= 1 "the compiled forms were compiled again during the time loop"

# ## Compiled or Gridap assembly?
# The same model with `backend=:lazy` gives the same solution. We time one assembly of the Jacobian of the last
# step with each backend, on the same state, after a warm-up call (`flow.caches.op` is the operator of the step
# that the solver assembles, `flow.caches.x` the state). The compiled Jacobian must be at least twice as fast
# (we measured 5 to 10 times): this check catches a slow pattern in the forms, such as a function evaluated at
# points that reads a non-constant global (tutorial 10), which turns every evaluation at a point into a dynamic
# call. With a non-constant ``\mu`` inside `τsupg` we measured 42 ms for the compiled Jacobian instead of 6 ms,
# against 67 ms for Gridap's.
function jacobian_seconds(m)
  op, xm = m.caches.op, m.caches.x
  A = allocate_jacobian(op, xm)
  Gridap.Algebra.jacobian!(A, op, xm)                          # warm-up (compilation)
  minimum(@elapsed(Gridap.Algebra.jacobian!(A, op, xm)) for _ in 1:5)
end
if !FAST
  Ul = MultiFieldFESpace([TrialFESpace(Vu, D_bc[1], 0.0), TrialFESpace(Vp, D_bc[2], 0.0)])
  flow_lazy = TransientModel(res, jac, Ul, V, D_bc; scheme=BDF(2), tangent_update=:step, nls=nls, backend=:lazy)
  tlazy = @elapsed xl, _ = solve!(flow_lazy; stepping=TimeStepping(; Δt=Δt, t_end=nsteps * Δt))
  tj_compiled, tj_lazy = jacobian_seconds(flow), jacobian_seconds(flow_lazy)
  println("2D-1 with Gridap's assembly: $nsteps steps in $(round(tlazy, digits=2)) s, ‖x - x_lazy‖/‖x‖ = $(norm(x - xl) / norm(x))")
  println("one Jacobian: compiled $(round(1e3 * tj_compiled, digits=2)) ms, Gridap $(round(1e3 * tj_lazy, digits=2)) ms")
  @assert norm(x - xl) < 1e-8 * norm(x) "the two backends give different solutions"
  @assert tj_compiled < tj_lazy / 2 "the compiled Jacobian ($tj_compiled s) is not twice as fast as Gridap's ($tj_lazy s)"
end

# ## Case 2D-2 (Re = 100): vortex shedding
# With ``U_m = 1.5`` m/s the wake is unstable. The cylinder is slightly below the axis of the channel, so
# the symmetric start is disturbed from the beginning and the shedding develops by itself: the lift
# oscillates with its final amplitude after some 3 s. Only the inflow changes: new boundary conditions,
# spaces and model, with the same `res` and `jac`.
# The time step resolves the shedding period (about 0.33 s) with some 30 steps. We record the force
# coefficients and the pressure difference at every step, and write the flow every few steps during the
# last period.
if !FAST
  Um2 = 1.5                                                    # m/s: Re = 100
  Ū2 = 2Um2 / 3
  D2 = MultiFieldBC([
    BoundaryConditions(
      BoundaryCondition("inlet", inflow(Um2, H)),
      BoundaryCondition("wall", [0.0, 0.0]),
      BoundaryCondition("cylinder", [0.0, 0.0])),
    NothingBC()])
  Vu2 = TestFESpace(model, reffeu, D2[1], conformity=:H1)
  Vp2 = TestFESpace(model, reffep, D2[2], conformity=:H1)
  V2 = MultiFieldFESpace([Vu2, Vp2])
  U2 = MultiFieldFESpace([TrialFESpace(Vu2, D2[1], 0.0), TrialFESpace(Vp2, D2[2], 0.0)])
  flow2 = TransientModel(res, jac, U2, V2, D2; scheme=BDF(2), tangent_update=:step, nls=nls, backend=:compiled)

  Δt2, t_end2 = 0.01, 8.0
  ts, cDs, cLs, Δps = Float64[], Float64[], Float64[], Float64[]
  function record2(post)
    xh = get_state(flow2)
    c = coefficients(xh, Ū2)
    t = post.Λ[end]
    push!(ts, t); push!(cDs, c.cD); push!(cLs, c.cL); push!(Δps, c.Δp)
    post.iter % 50 == 0 && println("t = $(round(t, digits=2)) s  C_D = $(c.cD)  C_L = $(c.cL)")
    if t > t_end2 - 0.34 && post.iter % 3 == 0
      get_pvd(post)[t] = createvtk(Ω, joinpath(outdir, "re100_$(post.iter)"), cellfields=["u" => xh[1], "p" => xh[2]])
    end
  end
  post2 = PostProcessor(flow2, record2; is_vtk=true, filepath=outdir)
  t2d2 = @elapsed x2, flag2 = solve!(flow2; stepping=TimeStepping(; Δt=Δt2, t_end=t_end2), post=post2)
  println("2D-2: $(length(ts)) steps in $(round(t2d2, digits=1)) s, $(round(t2d2 / length(ts), digits=3)) s per step")
end

# ### Shedding frequency
# The Strouhal number ``St = f D/\bar U`` from the period ``1/f`` between the upward zero crossings of
# ``C_L - \overline{C_L}`` in the last 3 s, and the maxima of ``C_D`` and ``C_L`` there.
# ``\Delta p`` is reported, as in the benchmark, half a period after the maximum of ``C_L``.
if !FAST
  win = ts .>= t_end2 - 3.0
  tw, cLw, cDw, Δpw = ts[win], cLs[win], cDs[win], Δps[win]
  c = cLw .- sum(cLw) / length(cLw)
  crossings = [tw[i] - c[i] * (tw[i+1] - tw[i]) / (c[i+1] - c[i]) for i in 1:length(c)-1 if c[i] < 0 <= c[i+1]]
  periods = diff(crossings)
  T = sum(periods) / length(periods)
  St = Dc / (Ū2 * T)
  i0 = findlast(t -> t <= tw[end] - T, tw)                     # maximum of C_L in the last full period
  i0 = findmax(cLw[i0-round(Int, T / Δt2):i0])[2] + i0 - round(Int, T / Δt2) - 1
  thalf = tw[i0] + T / 2
  j = findlast(t -> t <= thalf, tw)
  Δp_half = Δpw[j] + (Δpw[j+1] - Δpw[j]) * (thalf - tw[j]) / (tw[j+1] - tw[j])
  println("2D-2 (Re = 100, Δt = $Δt2 s, P1–P1, 7 402 triangles)   reference (Schäfer & Turek 1996)")
  println("  St             = $(round(St, digits=4))       0.2950 – 0.3050")
  println("  max C_D        = $(round(maximum(cDw), digits=4))       3.2200 – 3.2400")
  println("  max C_L        = $(round(maximum(cLw), digits=4))       0.9900 – 1.0100")
  println("  Δp(t₀ + T/2)   = $(round(Δp_half, digits=4))       2.4600 – 2.5000")
  spread = (maximum(periods) - minimum(periods)) / T
  println("  $(length(periods)) periods in the last 3 s, relative spread $(spread), C_L amplitude $((maximum(cLw) - minimum(cLw)) / 2)")
end

# ### Checks (2D-2)
# The shedding is periodic (several periods of equal length, a lift that oscillates with a finite
# amplitude) and its frequency is that of the benchmark: we measured ``St = 0.3007`` and assert it within 3 %
# of 0.30. On this mesh with linear elements the force amplitudes are further from the reference values
# (we measured ``\max C_D = 3.187`` and ``\max C_L = 0.952``, 1 % and 4 % below the reference intervals;
# ``\Delta p = 2.475`` is inside its interval): they are only reported.
if !FAST
  @assert flag2 == 1 "the time loop of 2D-2 did not complete"
  @assert length(periods) >= 5 && spread < 0.02 "no periodic vortex shedding (periods $periods)"
  @assert (maximum(cLw) - minimum(cLw)) / 2 > 0.3 "lift amplitude too small for vortex shedding"
  @assert isapprox(St, 0.30, rtol=0.03) "Strouhal number $St away from the benchmark value 0.30"
end

# ## Going further
# - `scheme=BDF(1)` is the backward Euler method: first order, more numerical damping. Try it on 2D-2
#   with the same time step: the shedding is weaker or delayed. A time-dependent inflow, such as the case
#   2D-3 of the benchmark (``U_m = 1.5\sin(\pi t/8)``), is a time function of the inlet condition:
#   `BoundaryCondition("inlet", inflow(1.5, H), t -> sin(π * t / 8))`.
# - The Oseen linearisation solves one linear system per step. A full Newton method on the convective term
#   (`tangent_update=:iteration` with the consistent tangent of ``\rho(\nabla\boldsymbol u)\boldsymbol u``)
#   converges to the fully implicit BDF2 solution at the cost of several solves per step.
# - Add a non-constant global to `τsupg` (write `μ2 = 1e-3` without `const` and use it instead of `μ`): the
#   solution does not change, but the compiled Jacobian becomes about seven times slower and the check above
#   fails.
# - Taylor–Hood P2–P1 (tutorial 10) needs no PSPG and gets closer to the reference values on the same mesh;
#   with quadratic velocities the SUPG residual also contains the viscous term ``-\mu\Delta\boldsymbol u``.
# - Tutorial 12 uses the same formulation for the pulsatile blood flow in an aorta, with resistance–capacitance
#   outlet conditions.
# - Reference: M. Schäfer and S. Turek, *Benchmark computations of laminar flow around a cylinder*,
#   Notes on Numerical Fluid Mechanics 52 (1996) 547–566, DOI 10.1007/978-3-322-89849-4_39.
