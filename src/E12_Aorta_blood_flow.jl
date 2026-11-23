# # Tutorial 12: Pulsatile blood flow in a patient aorta
#
# Blood is pumped by the heart into the aorta in short pulses and leaves it through the descending aorta and
# the three supra-aortic branches. We compute one cardiac cycle of the incompressible Navier–Stokes flow
# in a patient-specific (rigid) aorta, with the formulation of tutorial 11, a pulsatile inflow and
# three-element Windkessel models at the four outlets: each outlet sees the resistance and the compliance of
# the arterial tree downstream, which gives physiological pressures. The 3D flow and the four 0D models are
# coupled at every time step.
#
# Measured on a Mac (M5 Max, 4 Julia threads, 4 BLAS threads, Julia 1.13.0, HyperFEM 0.3.0, HyperFEMKernels
# 0.2.6, Gridap 0.20.8): 16 659 free dofs (P1–P1 on 33 643 tetrahedra). FAST (10 time steps) 92 s and full (one
# cycle, 200 time steps) 170 s, including compilation (the first step takes about 24 s); 0.21 s per time step
# after that. One assembly of the step Jacobian (the check at the end): 0.06 s with the compiled backend,
# 0.65 s with Gridap.
#
# **What you will learn**
# - boundary terms and lumped (0D) models in a `TransientModel`: Robin outlet conditions and backflow
#   stabilisation in the weak form, and an explicit 3D–0D coupling through the `presolver` and `internal` hooks;
# - a time-dependent Dirichlet condition, `BoundaryCondition(tag, profile, t -> Q(t))`;
# - the compiled backend of HyperFEMKernels (`backend=:compiled`) for a 3D stabilised flow problem, and how to
#   check that it takes its fast paths (`HyperFEMKernels.work_summary`);
# - mass-conservation checks of a 3D–0D model and a comparison with a pure 0D model.
#
# ## Problem statement
# Find the velocity ``\boldsymbol u`` and the pressure ``p`` in the lumen ``\Omega`` such that
# ```math
# \rho\left(\partial_t\boldsymbol u + (\nabla\boldsymbol u)\,\boldsymbol u\right) - \nabla\cdot\boldsymbol\sigma = \boldsymbol 0 ,
# \qquad \nabla\cdot\boldsymbol u = 0 , \qquad
# \boldsymbol\sigma = -p\boldsymbol I + 2\mu\,\boldsymbol\varepsilon(\boldsymbol u) ,
# ```
# with blood as a Newtonian fluid, ``\rho = 1050`` kg/m³ and ``\mu = 3.5`` mPa s. We work in mm, s and g: the
# density is ``1.05\cdot 10^{-3}`` g/mm³, the viscosity ``3.5\cdot 10^{-3}`` g/(mm s) = Pa s, and pressures come out in Pa.
#
# **Inlet.** The flow rate is a half sine during systole (period ``T = 1`` s, systole ``T_s = 0.34`` s, stroke
# volume ``V_s = 83.3`` ml) and zero in diastole:
# ```math
# Q(t) = \frac{\pi V_s}{2T_s}\sin\frac{\pi t}{T_s} \quad (0 \le t \le T_s), \qquad Q(t) = 0 \quad (T_s < t < T) .
# ```
# The velocity at the inlet is ``\boldsymbol u = Q(t)\,\boldsymbol g(\boldsymbol x)``, with ``\boldsymbol g`` a
# plug profile along the mean inward normal that vanishes on the rim of the inlet (no slip) and carries a unit
# flow rate. **Wall**: no slip, ``\boldsymbol u = \boldsymbol 0`` (rigid wall).
#
# **Outlets.** At each outlet ``\Gamma_i`` (descending aorta, brachiocephalic trunk, left common carotid and left
# subclavian arteries) a three-element Windkessel (RCR) model relates the flow rate
# ``Q_i = \int_{\Gamma_i}\boldsymbol u\cdot\boldsymbol n\,\mathrm d\Gamma`` and the pressure: a proximal
# resistance ``Z_i`` in series with a capacitor ``C_i`` (the compliance of the arteries downstream) that discharges
# through a distal resistance ``R_i``,
# ```math
# \boldsymbol\sigma\boldsymbol n = -\left(P_{c,i} + Z_i Q_i\right)\boldsymbol n ,
# \qquad C_i \frac{\mathrm dP_{c,i}}{\mathrm dt} = Q_i - \frac{P_{c,i}}{R_i} .
# ```
# The compliance pressure ``P_{c,i}`` is advanced with backward Euler using the flow rate of the previous step
# (explicit coupling), and the proximal term enters the weak form implicitly as a Robin term,
# ``\int_{\Gamma_i} Z_i A_i\,(\boldsymbol u\cdot\boldsymbol n)(\boldsymbol v\cdot\boldsymbol n)\,\mathrm d\Gamma``,
# which equals ``Z_i Q_i\int_{\Gamma_i}\boldsymbol v\cdot\boldsymbol n\,\mathrm d\Gamma`` when the normal velocity is
# uniform on the outlet (area ``A_i``). The values of ``Z_i``, ``C_i``, ``R_i`` are those of the aorta example of
# HyperFEMKernels.jl (`examples/aorta_cfd/aorta_flow.jl`), taken from published patient data.
#
# **Backflow.** During deceleration and diastole blood flows back into the domain through parts of the outlets,
# and the kinetic energy that enters there can make the computation blow up. The backflow stabilisation
# ``-\rho\int_{\Gamma_i}(\boldsymbol w\cdot\boldsymbol n)_-\,\boldsymbol u\cdot\boldsymbol v\,\mathrm d\Gamma``, with
# ``(a)_- = \min(a, 0)``, acts only where the flow enters (Bazilevs et al. 2009, DOI 10.1016/j.cma.2009.04.015;
# Esmaily-Moghadam et al. 2011, DOI 10.1007/s00466-011-0599-0).
#
# **Discretisation.** As in tutorial 11: linear velocity and pressure (P1–P1) with SUPG, PSPG and LSIC
# stabilisation, BDF2 in time with an extrapolated advecting velocity ``\boldsymbol w = 2\boldsymbol u^n -
# \boldsymbol u^{n-1}`` (an Oseen problem: one linear system per step), ``\Delta t = 5`` ms. The element size is the
# edge of the regular tetrahedron with the volume of the cell, ``h = (6\sqrt2\,|K|)^{1/3}``.
#
# ## Setup
using HyperFEM
using HyperFEMKernels      # the compiled backend (`backend=:compiled`) and its introspection
using Gridap
using GridapGmsh
using LinearAlgebra
using Printf

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E12_Aorta_blood_flow")
setupfolder(outdir)                                            # VTK output (not versioned)

# The constants read by functions evaluated at quadrature points (the stabilisation parameters) or by the
# inflow ``Q(t)`` are `const`: a non-constant global read inside such a function is looked up dynamically at
# every evaluation, which makes the compiled Jacobian several times slower.
const ρ = 1050e-6                                              # g/mm³
const μ = 3.5e-3                                               # Pa s
const Tc = 1.0                                                 # s: cardiac period
const Ts = 0.34                                                # s: systole
const Vs = 83.3e3                                              # mm³: stroke volume (83.3 ml)
const Qmax = π * Vs / (2Ts)                                    # mm³/s: peak flow rate
Qin(t) = (τ = mod(t, Tc); τ <= Ts ? Qmax * sin(π * τ / Ts) : 0.0)
const mmHg = 133.322                                           # Pa

# ## Geometry and mesh
# A patient aorta (ascending aorta, arch with three branches and descending aorta), 33 643 linear tetrahedra,
# with the physical surfaces `inlet`, `wall`, `outlet` (descending aorta), `B1`, `B2`, `B3` (branches).
# The mesh is coarse: most of its nodes lie on the wall, so the boundary layers are not resolved; it is meant
# to show the method, not to give converged wall shear stresses.
model = GmshDiscreteModel(joinpath(@__DIR__, "..", "data", "models", "aorta_fluido_h1p5_3p0.msh"))

# GridapGmsh gives each physical surface its triangles, but an edge or a vertex shared by two surfaces belongs
# to only one of them. A Dirichlet condition needs all the nodes of its faces, so we add the edges and vertices
# of the faces of `inlet` and then of `wall` to their tags; the shared ones (the rim of the inlet) end up in
# the last tag, `wall`: the inflow profile vanishes on the rim.
function close_boundary_tags!(model, names)
  labels = get_face_labeling(model)
  topo = Gridap.Geometry.get_grid_topology(model)
  D = num_cell_dims(model)
  for name in names
    tag = findfirst(==(name), labels.tag_to_name)
    faces = findall(Gridap.Geometry.get_face_mask(labels, name, D - 1))
    for d in 0:D-2
      f_to_d = Gridap.Geometry.get_faces(topo, D - 1, d)
      ents = labels.d_to_dface_to_entity[d+1]
      newent = maximum(maximum(e) for e in labels.d_to_dface_to_entity) + 1
      for f in faces, e in f_to_d[f]
        ents[e] = newent
      end
      push!(labels.tag_to_entities[tag], newent)
    end
  end
  model
end
close_boundary_tags!(model, ["inlet", "wall"])

Ω = Triangulation(model)
dΩ = Measure(Ω, 3)
he = CellField(map(v -> (6sqrt(2) * v)^(1 / 3), collect(get_cell_measure(Ω))), Ω)   # element size
Γin = BoundaryTriangulation(model, tags="inlet")
dΓin = Measure(Γin, 2)
nin = get_normal_vector(Γin)                                   # outward normal of the fluid
Ain = sum(∫(1.0)dΓin)
n̄ = sum(∫(nin)dΓin) / Ain                                      # mean outward normal of the inlet

# ## Windkessel parameters
# The values are given in SI units (Pa s/m³ and m³/Pa) and converted to mm–s–g. `ZA` is ``Z_i A_i``, the
# coefficient of the Robin term.
outs = ["outlet", "B1", "B2", "B3"]
SIR, SIC = 1e-9, 1e9                                           # Pa s/m³ → Pa s/mm³, m³/Pa → mm³/Pa
Z = [1.5e7, 1.36e8, 2.46e8, 1.74e8] .* SIR
C = [1.05e-8, 4.83e-10, 2.92e-10, 3.93e-10] .* SIC
R = [2.5e8, 9.23e8, 1.53e9, 1.13e9] .* SIR
Γo = [BoundaryTriangulation(model, tags=o) for o in outs]
dΓo = [Measure(Γ, 2) for Γ in Γo]
no = [get_normal_vector(Γ) for Γ in Γo]
ZA = [Z[i] * sum(∫(1.0)dΓo[i]) for i in eachindex(outs)]
flux(uh, i) = sum(∫(uh ⋅ no[i])dΓo[i])                         # outflow rate through outlet i

# The same four Windkessel models attached to a rigid, inviscid aorta (a single node with pressure ``P``) give a
# 0D model of the whole circulation downstream: ``Q(t) = \sum_i (P - P_{c,i})/Z_i`` and
# ``C_i\,\dot P_{c,i} = (P - P_{c,i})/Z_i - P_{c,i}/R_i``. Integrated over 30 cycles (RK4) it reaches its periodic
# state, which gives the compliance pressures at ``t = 0`` (the initial values of the 3D computation: the flow
# starts at rest, at the end of diastole) and a reference for the pressures and the split of the flow.
function windkessel_0d(Qin, Z, C, R; T, h=1e-4, ncycles=30)
  P0(t, Pc) = (Qin(t) + sum(Pc ./ Z)) / sum(1 ./ Z)
  f(t, Pc) = ((P0(t, Pc) .- Pc) ./ Z .- Pc ./ R) ./ C
  rk4(t, y) = (k1 = f(t, y); k2 = f(t + h / 2, y + h / 2 * k1); k3 = f(t + h / 2, y + h / 2 * k2);
               k4 = f(t + h, y + h * k3); y + h / 6 * (k1 + 2k2 + 2k3 + k4))
  nper = round(Int, T / h)
  Qm = sum(Qin(k * h) for k in 0:nper-1) / nper
  Pc = R ./ (Z .+ R) .* (Qm / sum(1 ./ (Z .+ R)))             # mean state as the starting point
  for k in 0:ncycles*nper-1
    Pc = rk4(k * h, Pc)
  end
  Pc0 = copy(Pc)
  ts, P, Q = Float64[], Float64[], Vector{Float64}[]           # one period of the periodic state
  for k in 0:nper
    t = k * h; p = P0(t, Pc)
    push!(ts, t); push!(P, p); push!(Q, (p .- Pc) ./ Z)
    Pc = rk4(t, Pc)
  end
  (; Pc0, ts, P, Q)
end
ref0d = windkessel_0d(Qin, Z, C, R; T=Tc)
@printf("0D model: P from %.1f to %.1f mmHg, Pc(0) = %s mmHg\n", minimum(ref0d.P) / mmHg, maximum(ref0d.P) / mmHg,
        string(round.(ref0d.Pc0 ./ mmHg, digits=1)))

# ## Boundary conditions and finite element spaces
# The plug profile is the interpolation of ``-\bar{\boldsymbol n}`` on the inlet with zero on the wall (and so
# on the rim); its discrete flow rate ``q_1`` normalises it, so that the imposed flow rate is exactly ``Q(t)``.
# The profile and the time function are given separately: `BoundaryCondition("inlet", g, Qin)` imposes
# ``\boldsymbol g(\boldsymbol x)\,Q(t)``. The profile is built by a function that returns a closure, so that the
# pointwise function reads its own captured value and not a global variable.
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)
reffep = ReferenceFE(lagrangian, Float64, 1)
uniform(a) = x -> a
Vaux = TestFESpace(model, reffeu; conformity=:H1, dirichlet_tags=["inlet", "wall"])
u1 = FEFunction(TrialFESpace(Vaux, [uniform(-n̄), uniform(zero(n̄))]), zeros(num_free_dofs(Vaux)))
q1 = -sum(∫(u1 ⋅ nin)dΓin)
D_bc = MultiFieldBC([
  BoundaryConditions(
    BoundaryCondition("inlet", uniform(-n̄ / q1), Qin),
    BoundaryCondition("wall", [0.0, 0.0, 0.0])),
  NothingBC()])
Vu = TestFESpace(model, reffeu, D_bc[1], conformity=:H1)
Vp = TestFESpace(model, reffep, D_bc[2], conformity=:H1)
V = MultiFieldFESpace([Vu, Vp])
U = MultiFieldFESpace([TrialFESpace(Vu, D_bc[1], 0.0), TrialFESpace(Vp, D_bc[2], 0.0)])

# ## Weak form
# The volume terms are those of tutorial 11 (with ``h`` of tetrahedra); the outlets add the Robin term of the
# proximal resistances and the backflow stabilisation, which are part of the operator, and the compliance
# pressures ``P_{c,i}``, which are known numbers of the step and only enter the residual.
# `Pc` holds the compliance pressures of the step being computed and `Pcn` those of the last accepted step.
Pcn = copy(ref0d.Pc0)
Pc = copy(Pcn)
τsupg(u, h) = ((2 * norm(u) / h)^2 + 9 * (4μ / (ρ * h^2))^2)^(-1 / 2)
τpspg(u, h) = τsupg(u, h) / ρ
τlsic(u, h) = ρ * norm(u) * h * 0.5
ε(∇u) = 0.5 * (∇u + ∇u')
negpart(a) = 0.5 * (a - abs(a))
function advecting(cu)
  a, b = cu.step == 1 ? (1.0, 0.0) : (2.0, 1.0)
  a * cu.u⁻ - b * cu.u⁻⁻
end
# The bilinear part of a step, with the advecting velocity `w`, the velocity `un` of the previous step (for the
# ``\tau``'s) and the time derivative `ut` of the unknown:
function step_form(w, un, (u, p), ut, (v, q))
  τs = τsupg ∘ (un, he); τp = τpspg ∘ (un, he); τl = τlsic ∘ (un, he)
  Ru = ρ * ut + ρ * (∇(u)' ⋅ w)
  a = ∫(ρ * (ut ⋅ v))dΩ + ∫(2μ * (ε ∘ (∇(u))) ⊙ (ε ∘ (∇(v))))dΩ + ∫(ρ * ((∇(u)' ⋅ w) ⋅ v))dΩ -
      ∫(p * (∇ ⋅ v))dΩ - ∫(q * (∇ ⋅ u))dΩ +
      ∫(τs * ((∇(v)' ⋅ w) ⋅ Ru))dΩ + ∫(τs * ((∇(v)' ⋅ w) ⋅ ∇(p)))dΩ -
      ∫(τp * (∇(q) ⋅ Ru))dΩ - ∫(τp * (∇(q) ⋅ ∇(p)))dΩ +
      ∫(τl * (∇ ⋅ v) * (∇ ⋅ u))dΩ
  for i in eachindex(outs)
    a = a + ∫(ZA[i] * (u ⋅ no[i]) * (v ⋅ no[i]))dΓo[i] - ∫(ρ * (negpart ∘ (w ⋅ no[i])) * (u ⋅ v))dΓo[i]
  end
  a
end
function res(ctx)
  cu = ctx[1]
  un = cu.u⁻; w = advecting(cu)
  ((u, p), (v, q)) -> begin
    r = step_form(w, un, (u, p), ∂t(cu, u), (v, q))
    for i in eachindex(outs)
      r = r + ∫(Pc[i] * (v ⋅ no[i]))dΓo[i]
    end
    r
  end
end
function jac(ctx)
  cu = ctx[1]
  un = cu.u⁻; w = advecting(cu)
  ((u, p), (du, dp), (v, q)) -> step_form(w, un, (du, dp), δ∂t(cu, du), (v, q))
end

# ## Solver and computational model
# `backend=:compiled` assembles with the compiled, threaded kernels of HyperFEMKernels. The operator is
# compiled in the first step; in the next steps only the numbers that change (the BDF coefficients, the
# compliance pressures, the Dirichlet values) are read again from the weak form, and the compiled code, the
# sparsity pattern and the tables are reused. The step system has 16 659 unknowns: a sparse LU factorisation
# costs about 0.06 s, as much as one assembly of the Jacobian, so a direct solver is the right choice at this
# size (tutorial 13 shows the iterative solvers for larger problems).
#
# The 0D models are advanced with two hooks. `presolver(t, Δt)` runs before every attempt of a step: it
# computes the compliance pressures of the attempt from the flow rates of the last accepted state.
# `internal(xh, xh⁻)` runs when the step is accepted: it makes them the new accepted values. With a fixed time
# step there are no rejected attempts, but the pair `Pc`/`Pcn` keeps the coupling right if a step were halved.
nls = NewtonRaphson(LUSolver(); verbose=0)
accept_windkessel(xh, xh⁻) = (Pcn .= Pc)
flow = TransientModel(res, jac, U, V, D_bc; scheme=BDF(2), tangent_update=:step, nls=nls,
                      backend=:compiled, internal=accept_windkessel)
# The flow rates of the last accepted state are computed once, by the post-processing function below (which
# runs after every accepted step), and kept in `Qn` (zero at rest, before the first step).
Qn = zeros(length(outs))
function windkessel!(t, Δt)
  for i in eachindex(outs)
    Pc[i] = (Pcn[i] + Δt * Qn[i] / C[i]) / (1 + Δt / (R[i] * C[i]))
  end
end

# ## Post-processing
# After every step we store the inflow rate (measured on the solution), the four outflow rates, the
# compliance pressures, the outlet pressures ``P_{c,i} + Z_iQ_i``, the mean pressure on the inlet and the
# largest velocity component; the flow is written to VTK a few times per cycle.
Δt = 0.005
nsteps = FAST ? 10 : round(Int, Tc / Δt)
vtk_every = FAST ? nsteps : 20
hist = (t=Float64[], qin=Float64[], qout=Vector{Float64}[], Pc=Vector{Float64}[], Pout=Vector{Float64}[],
        pin=Float64[], umax=Float64[], wall=Float64[])
tclock = Ref(time())
function record(post)
  uh, ph = get_state(flow)
  t = post.Λ[end]
  q = [flux(uh, i) for i in eachindex(outs)]
  Qn .= q
  push!(hist.t, t); push!(hist.qin, -sum(∫(uh ⋅ nin)dΓin)); push!(hist.qout, q); push!(hist.Pc, copy(Pc))
  push!(hist.Pout, Pc .+ Z .* q); push!(hist.pin, sum(∫(ph)dΓin) / Ain)
  push!(hist.umax, maximum(abs, get_free_dof_values(uh))); push!(hist.wall, time() - tclock[])
  tclock[] = time()
  if FAST || post.iter % 10 == 0
    @printf("step %3d  t = %.3f s  Q_in = %6.1f ml/s  Q_out = %s ml/s  P_in = %5.1f mmHg  %.2f s\n", post.iter, t,
            hist.qin[end] / 1e3, string(round.(q ./ 1e3, digits=1)), hist.pin[end] / mmHg, hist.wall[end])
  end
  if post.iter % vtk_every == 0
    get_pvd(post)[t] = createvtk(Ω, joinpath(outdir, "aorta_$(post.iter)"), cellfields=["u" => uh, "p" => ph])
  end
end
post = PostProcessor(flow, record; is_vtk=true, filepath=outdir)

# ## Solve
println("$(num_cells(Ω)) tetrahedra, $(num_free_dofs(U)) free dofs, $nsteps steps of $Δt s")
tclock[] = time()
x, flag = solve!(flow; stepping=TimeStepping(; Δt=Δt, t_end=nsteps * Δt), post=post, presolver=windkessel!)
steady = hist.wall[3:end]                                       # the first two steps compile
@printf("mean time per step after the first two: %.3f s\n", sum(steady) / length(steady))

# ## Checks
# **Mass conservation.** The pressure test functions include the constants, and with ``q = 1`` the weak form
# says ``\int_\Omega\nabla\cdot\boldsymbol u\,\mathrm d\Omega = 0`` (the stabilisation terms contain ``\nabla q``):
# the inflow equals the sum of the outflows at every step, to the precision of the linear solver. The
# measured inflow is the imposed ``Q(t)``. The tolerance of these checks and of the volume budget below is
# ``10^{-10}`` relative to the peak flow: the LU solution is accurate to about ``10^{-12}`` (we measured
# ``9\cdot 10^{-13}`` for the mass balance), and the budget inherits that error.
qout_sum = [sum(q) for q in hist.qout]
mass_err = maximum(abs.(hist.qin .- qout_sum)) / Qmax
inflow_err = maximum(abs.(hist.qin .- Qin.(hist.t))) / Qmax
println("max |Q_in - ΣQ_out| / Q_max = $mass_err,  max |Q_in - Q(t)| / Q_max = $inflow_err")
@assert flag == 1 "the time loop did not complete"
@assert length(hist.t) == nsteps "a time step was bisected"
@assert mass_err < 1e-10 "mass is not conserved: max |Q_in - ΣQ_out| / Q_max = $mass_err"
@assert inflow_err < 1e-10 "the inflow rate differs from Q(t): $inflow_err"

# **Volume budget of the 3D–0D system.** The volume that entered up to the previous step is stored in the
# compliances or has left through the distal resistances: summing the update of ``P_{c,i}`` over the steps,
# ``\Delta t\sum_{k<N}Q_{in}^k = \sum_i C_i\,(P_{c,i}^N - P_{c,i}^0) + \Delta t\sum_{k\le N}\sum_i P_{c,i}^k/R_i``
# (the flow rate is lagged by one step, and ``Q^0 = 0``: the flow starts at rest).
Vin = Δt * sum(hist.qin[1:end-1])
Vstored = sum(C .* (hist.Pc[end] .- ref0d.Pc0))
Vdrained = Δt * sum(sum(Pk ./ R) for Pk in hist.Pc)
budget_err = abs(Vin - Vstored - Vdrained) / (Δt * nsteps * Qmax)
@printf("volume budget: in %.1f ml = stored %.1f ml + drained %.1f ml  (relative error %.1e)\n",
        Vin / 1e3, Vstored / 1e3, Vdrained / 1e3, budget_err)
@assert budget_err < 1e-10 "the volume budget of the Windkessel models does not close: $budget_err"

# **Physiological range.** The pressures at the outlets and at the inlet stay in the physiological range of the
# systemic arteries, and the velocity stays bounded (the backflow stabilisation does its job).
Pall = reduce(vcat, hist.Pout)
@printf("outlet pressures %.1f–%.1f mmHg, inlet %.1f–%.1f mmHg, max velocity component %.0f mm/s\n",
        minimum(Pall) / mmHg, maximum(Pall) / mmHg, minimum(hist.pin) / mmHg, maximum(hist.pin) / mmHg, maximum(hist.umax))
@assert all(p -> 50mmHg < p < 180mmHg, Pall) "outlet pressure out of the physiological range"
@assert all(p -> 50mmHg < p < 180mmHg, hist.pin) "inlet pressure out of the physiological range"
@assert maximum(hist.umax) < 3000 "velocity above 3 m/s"

# **Regression values** of the final state (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0,
# macOS arm64): the norm of the vector of free values (dominated by the pressures, in Pa) and the mean inlet
# pressure (Pa), after 10 steps (FAST) and after one cycle.
println("regression: ‖x‖ = $(norm(x)), Pc = $(hist.Pc[end]), P_in = $(hist.pin[end])")
if FAST
  @assert isapprox(norm(x), 1.5565908708897454e6, rtol=1e-5) "regression: ‖x‖ = $(norm(x))"
  @assert isapprox(hist.pin[end], 16656.492743907616, rtol=1e-5) "regression: P_in = $(hist.pin[end])"
else
  @assert isapprox(norm(x), 1.0728389663079288e6, rtol=1e-5) "regression: ‖x‖ = $(norm(x))"
  @assert isapprox(hist.pin[end], 10492.936373140392, rtol=1e-5) "regression: P_in = $(hist.pin[end])"
end

# ## Comparison with the 0D model (full run)
# Over one cycle the 3D–0D model and the 0D model see the same inflow and the same Windkessel models; they
# differ by the resistance and the inertia of the aorta itself and by the spatial distribution of the pressure
# at the outlets. We compare the volume that leaves through each outlet in the cycle and the extreme pressures
# of the descending aorta. We measured differences of 0.2–2.7 % in the volumes (the branches, whose flow
# depends more on the 3D pressure distribution, differ most) and of about 1 % in the extreme pressures; the
# tolerances are 5 % and 3 %.
if !FAST
  vol3d = Δt .* reduce(+, hist.qout)
  w0 = [k == 1 || k == length(ref0d.ts) ? 0.5 : 1.0 for k in eachindex(ref0d.ts)]
  vol0d = (ref0d.ts[2] - ref0d.ts[1]) .* reduce(+, w0 .* ref0d.Q)
  Pdesc = [P[1] for P in hist.Pout]
  println("outlet                volume 3D (ml)   volume 0D (ml)")
  for i in eachindex(outs)
    @printf("  %-8s           %10.2f       %10.2f\n", outs[i], vol3d[i] / 1e3, vol0d[i] / 1e3)
  end
  @printf("descending aorta: P 3D %.1f–%.1f mmHg, 0D %.1f–%.1f mmHg\n", minimum(Pdesc) / mmHg, maximum(Pdesc) / mmHg,
          minimum(ref0d.P) / mmHg, maximum(ref0d.P) / mmHg)
  @assert all(isapprox.(vol3d ./ sum(vol3d), vol0d ./ sum(vol0d), rtol=0.05)) "flow split away from the 0D model"
  @assert isapprox(maximum(Pdesc), maximum(ref0d.P), rtol=0.03) "systolic pressure away from the 0D model"
  @assert isapprox(minimum(Pdesc), minimum(ref0d.P), rtol=0.03) "diastolic pressure away from the 0D model"
end

# ## Efficiency of the compiled backend (full run)
# With Gridap's assembly (`backend=:lazy`) the Jacobian alone takes 0.65 s and dominates the step; with the
# compiled backend a step of 0.21 s splits into Jacobian 0.06 s, residual 0.04 s, LU 0.06 s and the rest (see
# below). Running this tutorial with `backend=:lazy` gives the same solution (‖x‖ equal to 1e-13 in FAST) at
# about 1.0 s per step. We build, on the final state, the Jacobian
# of a BDF2 step with the same forms, assemble it with Gridap and with a `KernelOperator` of HyperFEMKernels
# (what `backend=:compiled` does inside the model), and check that both give the same matrix and that the
# compiled one is at least four times faster: we measured eleven times, and a non-constant global read by a
# stabilisation parameter (see the setup) makes the compiled Jacobian about eight times slower, which this
# margin catches. `HyperFEMKernels.work_summary` lists the path each group of integrals takes:
# `:factorised` or `:linear_maps` are the fast paths; `:duals` (automatic differentiation at every point)
# would point at a weak form or a pointwise function the compiler could not analyse. We measured (4 threads)
# 0.06 s for the compiled Jacobian and 0.65 s for Gridap's, with the volume group and the four outlet groups
# all `:factorised` and nothing left to Gridap. Inside the model a step costs about 0.21 s: Jacobian 0.06 s,
# residual 0.04 s, LU factorisation 0.06 s, and the rest is the update of the operator with the numbers of the
# step and the post-processing.
if !FAST
  uN = get_state(flow)[1]
  c0 = 1.5 / Δt
  kres(x, y) = step_form(uN, uN, x, c0 * x[1], y)
  kjac(x, dx, y) = step_form(uN, uN, dx, c0 * dx[1], y)
  x0 = zeros(num_free_dofs(U))
  gop = Gridap.FESpaces.get_algebraic_operator(FEOperator(kres, kjac, U, V))
  kop = KernelOperator(kres, kjac, U, V)
  Ag = Gridap.Algebra.allocate_jacobian(gop, x0)
  Ak = Gridap.Algebra.allocate_jacobian(kop, x0)
  Gridap.Algebra.jacobian!(Ag, gop, x0); Gridap.Algebra.jacobian!(Ak, kop, x0)   # warm-up (compilation)
  tg = minimum(@elapsed(Gridap.Algebra.jacobian!(Ag, gop, x0)) for _ in 1:3)
  tk = minimum(@elapsed(Gridap.Algebra.jacobian!(Ak, kop, x0)) for _ in 1:3)
  ws = HyperFEMKernels.work_summary(kop)
  for g in ws.groups
    g.kind === :jacobian && println("  Jacobian group $(g.group): $(g.domain), $(g.cells) cells, path $(g.path)")
  end
  @printf("Jacobian assembly: compiled %.3f s, Gridap %.3f s (%d threads); fallback groups: %d\n", tk, tg,
          Threads.nthreads(), ws.fallback_jacobian)
  @assert norm(Ak - Ag) < 1e-10 * norm(Ag) "the compiled Jacobian differs from Gridap's"
  @assert ws.fallback_jacobian == 0 "part of the Jacobian falls back to Gridap"
  @assert all(g -> g.path !== :duals, filter(g -> g.kind === :jacobian, ws.groups)) "a Jacobian group takes the slow :duals path"
  @assert tk < tg / 4 "the compiled Jacobian ($tk s) is not four times faster than Gridap's ($tg s)"
end

# ## Going further
# - Two or more cycles: `nsteps = 2 * round(Int, Tc / Δt)`. The compliance pressures start from the periodic
#   state of the 0D model, so the second cycle differs little from the first.
# - Taylor–Hood P2–P1 (99 141 dofs on this mesh) needs no PSPG and resolves the flow better; with quadratic
#   velocities the SUPG residual also contains the viscous term ``-\mu\Delta\boldsymbol u^n``, which must be
#   computed (for instance by projecting ``\nabla\boldsymbol u^n`` onto discontinuous P1 functions).
# - A realistic inflow profile, from the solution of a Poisson problem on the inlet plus an eccentric valve jet,
#   is in `examples/aorta_cfd/aorta_flow.jl` of HyperFEMKernels.jl, which also compares the assembly of that
#   P2–P1 problem with Gridap and with HyperFEMKernels.
# - `backend=:lazy` assembles with Gridap: same solution, slower steps.
# - Tutorial 11 validates the same formulation against the cylinder benchmark.
# - References: I. E. Vignon-Clementel et al., *Outflow boundary conditions for three-dimensional finite element
#   modeling of blood flow and pressure in arteries*, Comput. Methods Appl. Mech. Eng. 195 (2006) 3776–3796,
#   DOI 10.1016/j.cma.2005.04.014; T. E. Tezduyar, *Stabilized finite element formulations for incompressible
#   flow computations*, Adv. Appl. Mech. 28 (1992) 1–44, DOI 10.1016/S0065-2156(08)70153-4.
