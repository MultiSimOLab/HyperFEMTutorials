# # Tutorial 10: Stokes flow around a cylinder
#
# Creeping (Stokes) flow is the simplest incompressible flow problem and already contains its main
# difficulty: velocity and pressure are coupled by a saddle-point system, and not every pair of finite
# element spaces gives a stable pressure. We first check the pipeline on Poiseuille flow in a straight
# channel, where the stable Taylor–Hood element reproduces the exact solution, and then compute the flow
# around the cylinder of the DFG benchmark (Schäfer & Turek 1996) with Taylor–Hood (P2–P1) and with
# equal-order P1–P1 elements stabilised by PSPG. Tutorial 11 adds inertia and time to the same geometry.
#
# Measured on a Mac (M5 Max, 4 Julia threads, 4 BLAS threads, Julia 1.13.0, HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8): FAST and full runs are the same case (the problem is linear and small), 45 s including
# compilation and package loading; 335 dofs (Poiseuille, Q2–Q1), 32 878 dofs (P2–P1) and 10 972 dofs (P1–P1)
# on the cylinder mesh. After compilation, assembling and solving P2–P1 takes 0.085 s; one assembly of its
# matrix takes 4 ms with the compiled backend and 49 ms with Gridap's (P1–P1 + PSPG: 2.5 ms and 20 ms).
#
# **What you will learn**
# - mixed velocity–pressure spaces: `MultiFieldFESpace` and `MultiFieldBC([Du, NothingBC()])`;
# - a Dirichlet inflow profile given as a function, and the natural (do-nothing) outflow condition;
# - `StaticLinearModel` for a multi-field linear problem;
# - inf–sup stability: Taylor–Hood elements, and equal-order elements with PSPG stabilisation;
# - post-processing of a flow: drag and lift on a body, pressure drop and mass balance;
# - why a function evaluated at points must not read non-constant global variables;
# - assembling with the compiled backend of HyperFEMKernels (`backend=:compiled`).
#
# ## Problem statement
# Find the velocity ``\boldsymbol u`` and the pressure ``p`` in the channel ``\Omega`` such that
# ```math
# -\mu \Delta \boldsymbol u + \nabla p = \boldsymbol 0 , \qquad \nabla\cdot\boldsymbol u = 0 \quad \text{in } \Omega ,
# ```
# with a parabolic velocity ``\boldsymbol u = (4U_m y (H-y)/H^2, 0)`` on the inlet ``x = 0``, no slip
# (``\boldsymbol u = \boldsymbol 0``) on the walls and on the cylinder, and the *do-nothing* condition
# ``\mu \partial_{\boldsymbol n}\boldsymbol u - p\,\boldsymbol n = \boldsymbol 0`` on the outlet. The weak form:
# find ``(\boldsymbol u, p)`` such that, for all ``(\boldsymbol v, q)`` with ``\boldsymbol v = \boldsymbol 0``
# on the Dirichlet boundary,
# ```math
# \int_\Omega \mu\, \nabla\boldsymbol u : \nabla\boldsymbol v \,\mathrm d\Omega
# - \int_\Omega p\, \nabla\cdot\boldsymbol v \,\mathrm d\Omega
# - \int_\Omega q\, \nabla\cdot\boldsymbol u \,\mathrm d\Omega = 0 .
# ```
# The do-nothing condition is the natural condition of this form; it is not imposed anywhere. (With the
# symmetric form ``2\mu\,\boldsymbol\varepsilon(\boldsymbol u):\boldsymbol\varepsilon(\boldsymbol v)``, used in
# tutorial 11, the natural condition is ``(2\mu\boldsymbol\varepsilon(\boldsymbol u) - p\boldsymbol I)\boldsymbol n = \boldsymbol 0``
# instead, which a Poiseuille profile does not satisfy at the outlet.)
#
# The parameters are those of the DFG benchmark: ``H = 0.41`` m, ``U_m = 0.3`` m/s, ``\mu = 10^{-3}`` Pa s,
# ``\rho = 1`` kg/m³ (``\rho`` only enters the definition of the force coefficients).
#
# ## Setup
using HyperFEM
using Gridap
using GridapGmsh
using HyperFEMKernels                                          # enables `backend=:compiled`

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite (same case here)
outdir = joinpath(@__DIR__, "..", "data", "sims", "E10_Stokes")
setupfolder(outdir)                                            # VTK output (not versioned)

# The inflow profile `u_in` is a Julia function of the position, evaluated at every node of the inlet and,
# in the error of part 0, at every quadrature point. **A function evaluated at points may only read its
# arguments and `const` globals.** A non-constant global, such as `Um` written without `const`, can change its
# type at any time, so Julia cannot specialise the function on it: every one of the thousands of evaluations
# looks the variable up and dispatches dynamically. The result is the same, but slower; in the stabilised
# blood flow of tutorial 12 a non-constant `μ` inside the stabilisation parameter made the compiled Jacobian
# 8 times slower. Hence the `const` below. (A closure built by a function, as `inflow(Um, H)` in tutorial 11, is
# the other way.) A number that only multiplies a term of an integral, such as `μ` in `∫(μ * …)dΩ`, is read once
# when the form is built and does not need to be constant.
const μ = 1e-3                                                 # Pa s
const ρ = 1.0                                                  # kg/m³
const Um = 0.3                                                 # m/s, peak inlet velocity
const H = 0.41                                                 # m, channel height
u_in(x) = VectorValue(4Um * x[2] * (H - x[2]) / H^2, 0.0)       # parabolic inflow profile

# ## Part 0: Poiseuille flow in a straight channel
# In a channel without obstacle the solution is known: the inflow profile is carried unchanged to the
# outlet and the pressure falls linearly, ``p = 8\mu U_m (L - x)/H^2``. The velocity is quadratic in ``y``
# and the pressure linear in ``x``, so the Taylor–Hood pair Q2–Q1 (biquadratic velocity, bilinear
# pressure) contains the exact solution: the error must be at the level of round-off.
#
# ### Geometry and mesh
# The default tags of a 2D Cartesian model are 1–4 for the corners and 5–8 for the sides
# (``y = 0``, ``y = H``, ``x = 0``, ``x = L``). The walls take the corners.
const L0 = 1.0                                                 # m, channel length (read by p_exact below)
model0 = CartesianDiscreteModel((0, L0, 0, H), (10, 4))
labels0 = get_face_labeling(model0)
add_tag_from_tags!(labels0, "inlet", [7])
add_tag_from_tags!(labels0, "wall", [1, 2, 3, 4, 5, 6])

# ### Boundary conditions and spaces
# The velocity has Dirichlet conditions; the pressure has none (`NothingBC()`): it is fixed by the
# outlet condition. `MultiFieldBC` keeps one set of conditions per field, in the order of the fields.
Du0 = BoundaryConditions(
  BoundaryCondition("inlet", u_in),
  BoundaryCondition("wall", [0.0, 0.0]))
D0 = MultiFieldBC([Du0, NothingBC()])

reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)    # velocity: order 2
reffep = ReferenceFE(lagrangian, Float64, 1)                   # pressure: order 1
Vu0 = TestFESpace(model0, reffeu, D0[1], conformity=:H1)
Vp0 = TestFESpace(model0, reffep, D0[2], conformity=:H1)
V0 = MultiFieldFESpace([Vu0, Vp0])
U0 = MultiFieldFESpace([TrialFESpace(Vu0, D0[1], 1.0), TrialFESpace(Vp0, D0[2], 1.0)])

# ### Weak form and solution
# `StaticLinearModel(l, a, U, V, D)` assembles the bilinear form `a` and the linear form `l` (with the
# Dirichlet lift) and factorises the matrix when it is built; `solve!` then solves the system.
# `backend=:compiled` assembles with HyperFEMKernels: the forms are written as for Gridap and compiled into one
# kernel per element (`backend=:lazy`, the default, assembles with Gridap and gives the same solution). On the
# cylinder mesh below the compiled assembly of the P2–P1 matrix takes 4 ms and Gridap's 49 ms; the sparse LU
# factorisation, the right solver at this size, dominates the solve either way.
Ω0 = Triangulation(model0)
dΩ0 = Measure(Ω0, 4)
f = VectorValue(0.0, 0.0)                                       # body force
a0((u, p), (v, q)) = ∫(μ * (∇(u) ⊙ ∇(v)))dΩ0 - ∫(p * (∇ ⋅ v))dΩ0 - ∫(q * (∇ ⋅ u))dΩ0
l0((v, q)) = ∫(f ⋅ v)dΩ0
poiseuille = StaticLinearModel(l0, a0, U0, V0, D0; backend=:compiled)
solve!(poiseuille; Assembly=false)
uh0, ph0 = get_state(poiseuille)

p_exact(x) = 8μ * Um * (L0 - x[1]) / H^2
normL2(w) = sqrt(sum(∫(w ⊙ w)dΩ0))
eu0 = normL2(uh0 - u_in) / normL2(CellField(u_in, Ω0))
ep0 = normL2(ph0 - p_exact) / normL2(CellField(p_exact, Ω0))
println("Poiseuille (Q2–Q1, $(num_free_dofs(U0)) dofs): relative L2 errors u $(eu0), p $(ep0)")
writevtk(Ω0, joinpath(outdir, "poiseuille"), cellfields=["u" => uh0, "p" => ph0])

# ## Part 1: flow around a cylinder
# ### Geometry and mesh
# The channel of the DFG benchmark: ``2.2 \times 0.41`` m with a cylinder of diameter ``D = 0.1`` m centred
# at ``(0.2, 0.2)``, slightly below the axis of the channel. 7 402 linear triangles, refined around the
# cylinder; the mesh is made by `data/models/turek.jl` (Gmsh) with tags `inlet`, `outlet`, `wall`,
# `cylinder`.
model = GmshDiscreteModel(joinpath(@__DIR__, "..", "data", "models", "turek2d.msh"))
Dc = 0.1                                                       # m, cylinder diameter
Ū = 2Um / 3                                                    # mean inflow velocity
Ω = Triangulation(model)

D_bc = MultiFieldBC([
  BoundaryConditions(
    BoundaryCondition("inlet", u_in),
    BoundaryCondition("wall", [0.0, 0.0]),
    BoundaryCondition("cylinder", [0.0, 0.0])),
  NothingBC()])

# ### Post-processing quantities
# The force of the fluid on the cylinder is ``\boldsymbol F = -\int_{\Gamma_c} \boldsymbol\sigma\,\boldsymbol n \,\mathrm d\Gamma``
# with ``\boldsymbol\sigma = -p\boldsymbol I + \mu(\nabla\boldsymbol u + \nabla\boldsymbol u^T)`` and ``\boldsymbol n``
# the outward normal of the fluid domain (pointing into the cylinder); the drag and lift coefficients are
# ``C_{D,L} = 2F_{x,y}/(\rho \bar U^2 D)``. The pressure difference ``\Delta p = p(0.15, 0.2) - p(0.25, 0.2)``
# between the front and the back of the cylinder is the other output of the benchmark. The volume flow
# rates through the inlet and the outlet must be equal: the pressure test function ``q = 1`` turns the
# discrete continuity equation into ``\int_{\partial\Omega}\boldsymbol u\cdot\boldsymbol n\,\mathrm d\Gamma = 0``,
# for both discretisations below.
Γc = BoundaryTriangulation(model, tags="cylinder")
dΓc = Measure(Γc, 4)
nc = get_normal_vector(Γc)
Γin, Γout = BoundaryTriangulation(model, tags="inlet"), BoundaryTriangulation(model, tags="outlet")
dΓin, dΓout = Measure(Γin, 4), Measure(Γout, 4)
nin, nout = get_normal_vector(Γin), get_normal_vector(Γout)

function flow_outputs(uh, ph)
  σ = μ * (∇(uh) + ∇(uh)') - ph * I2D
  F = -sum(∫(σ ⋅ nc)dΓc)
  cD, cL = 2F[1] / (ρ * Ū^2 * Dc), 2F[2] / (ρ * Ū^2 * Dc)
  Δp = ph(Point(0.15, 0.2)) - ph(Point(0.25, 0.2))
  Qin, Qout = -sum(∫(uh ⋅ nin)dΓin), sum(∫(uh ⋅ nout)dΓout)
  (; cD, cL, Δp, Qin, Qout)
end

# Each solve writes its velocity and pressure through a `PostProcessor` (called once by `solve!` of a
# linear model).
function vtk_writer(name, model_)
  record(post) = (get_pvd(post)[0.0] = createvtk(Ω, joinpath(outdir, name),
                    cellfields=["u" => get_state(model_)[1], "p" => get_state(model_)[2]]))
  PostProcessor(model_, record; is_vtk=true, filepath=outdir)
end

# ### Taylor–Hood P2–P1
# The same element as in part 0, now on triangles. It satisfies the discrete inf–sup condition, so the
# plain weak form above is stable.
dΩ2 = Measure(Ω, 4)
Vu2 = TestFESpace(model, reffeu, D_bc[1], conformity=:H1)
Vp2 = TestFESpace(model, reffep, D_bc[2], conformity=:H1)
V2 = MultiFieldFESpace([Vu2, Vp2])
U2 = MultiFieldFESpace([TrialFESpace(Vu2, D_bc[1], 1.0), TrialFESpace(Vp2, D_bc[2], 1.0)])
a2((u, p), (v, q)) = ∫(μ * (∇(u) ⊙ ∇(v)))dΩ2 - ∫(p * (∇ ⋅ v))dΩ2 - ∫(q * (∇ ⋅ u))dΩ2
l2((v, q)) = ∫(f ⋅ v)dΩ2
stokes_th = StaticLinearModel(l2, a2, U2, V2, D_bc; backend=:compiled)
solve!(stokes_th; Assembly=false, post=vtk_writer("cylinder_P2P1", stokes_th))
th = flow_outputs(get_state(stokes_th)...)
println("P2–P1 ($(num_free_dofs(U2)) dofs): C_D = $(th.cD), C_L = $(th.cL), Δp = $(th.Δp), Q_in = $(th.Qin), Q_out = $(th.Qout)")

# ### Equal order P1–P1 with PSPG
# Linear velocity and pressure on the same mesh are cheaper and simpler (one set of nodes), but violate
# the inf–sup condition: the plain weak form admits spurious pressure oscillations. The pressure-stabilised
# Petrov–Galerkin method (PSPG; Hughes, Franca & Balestra 1986, DOI 10.1016/0045-7825(86)90025-3) adds the
# residual of the momentum equation tested with ``-\tau\nabla q`` to the continuity equation:
# ```math
# \ldots - \int_\Omega q\,\nabla\cdot\boldsymbol u\,\mathrm d\Omega
# - \sum_K \int_K \tau\, \nabla q \cdot (-\mu\Delta\boldsymbol u + \nabla p)\,\mathrm d\Omega = 0 ,
# \qquad \tau = \frac{h_K^2}{12\mu} .
# ```
# The method is consistent (the exact solution makes the added term vanish) and, with linear velocities,
# ``\Delta\boldsymbol u = \boldsymbol 0`` inside each element, so only ``\tau\nabla q\cdot\nabla p`` remains.
# ``h_K = \sqrt{2|K|}`` is the size of each triangle, a piecewise constant `CellField`; ``\tau`` is the value of
# the stabilisation parameter of tutorial 11 at zero velocity.
reffeu1 = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
dΩ1 = Measure(Ω, 2)
Vu1 = TestFESpace(model, reffeu1, D_bc[1], conformity=:H1)
Vp1 = TestFESpace(model, reffep, D_bc[2], conformity=:H1)
V1 = MultiFieldFESpace([Vu1, Vp1])
U1 = MultiFieldFESpace([TrialFESpace(Vu1, D_bc[1], 1.0), TrialFESpace(Vp1, D_bc[2], 1.0)])
hK = CellField(map(a -> sqrt(2a), collect(get_cell_measure(Ω))), Ω)
τ = hK * hK / (12μ)
a1((u, p), (v, q)) = ∫(μ * (∇(u) ⊙ ∇(v)))dΩ1 - ∫(p * (∇ ⋅ v))dΩ1 - ∫(q * (∇ ⋅ u))dΩ1 -
                     ∫(τ * (∇(q) ⋅ ∇(p)))dΩ1
l1((v, q)) = ∫(f ⋅ v)dΩ1
stokes_pspg = StaticLinearModel(l1, a1, U1, V1, D_bc; backend=:compiled)
solve!(stokes_pspg; Assembly=false, post=vtk_writer("cylinder_P1P1_PSPG", stokes_pspg))
ps = flow_outputs(get_state(stokes_pspg)...)
println("P1–P1 + PSPG ($(num_free_dofs(U1)) dofs): C_D = $(ps.cD), C_L = $(ps.cL), Δp = $(ps.Δp), Q_in = $(ps.Qin), Q_out = $(ps.Qout)")

# ## Checks
# Part 0 reproduces the exact solution; in part 1 the flow rate is conserved by both elements (up to
# round-off), the two elements agree on the drag and the pressure drop, and the Taylor–Hood values are
# kept as regression values (HyperFEM 0.3.0, Gridap 0.20.8, Julia 1.13.0, macOS arm64). The two elements
# give ``C_D \approx 3.13`` and ``3.10``, ``\Delta p \approx 0.0456`` and ``0.0453`` Pa: without inertia the drag is
# well below the ``C_D = 5.58`` of the same inflow with the Navier–Stokes equations (tutorial 11).
@assert eu0 < 1e-9 && ep0 < 1e-9 "Poiseuille flow not reproduced (errors $eu0, $ep0)"
@assert abs(th.Qin - 2Um * H / 3) < 1e-10 "P2–P1: inflow rate different from the exact 2 Um H / 3"
for r in (th, ps)
  @assert abs(r.Qout - r.Qin) < 1e-9 * r.Qin "outflow rate different from the inflow rate"
end
@assert isapprox(ps.cD, th.cD, rtol=0.05) "P1–P1 + PSPG drag differs from P2–P1 by more than 5 %"
@assert isapprox(ps.Δp, th.Δp, rtol=0.05) "P1–P1 + PSPG pressure drop differs from P2–P1 by more than 5 %"
@assert isapprox(th.cD, 3.1347333364256826, rtol=1e-5) "regression: C_D = $(th.cD) (P2–P1)"
@assert isapprox(th.Δp, 0.04555520370905265, rtol=1e-5) "regression: Δp = $(th.Δp) (P2–P1)"

# ## Going further
# - Remove the PSPG term (`τ = 0`): the P1–P1 pressure is polluted by oscillations (look at `p` in
#   ParaView) and the pressure drop is wrong, although the velocity may look reasonable.
# - Replace ``\mu\nabla\boldsymbol u:\nabla\boldsymbol v`` by the symmetric form
#   ``2\mu\boldsymbol\varepsilon(\boldsymbol u):\boldsymbol\varepsilon(\boldsymbol v)`` in part 0: the
#   solution is no longer the exact Poiseuille flow near the outlet (see the problem statement).
# - The Stokes flow is symmetric in time: reverse the inflow and the flow pattern reverses exactly.
#   With inertia it does not; tutorial 11 solves the Navier–Stokes equations on this mesh and compares
#   with the published values of the benchmark: M. Schäfer and S. Turek, *Benchmark computations of
#   laminar flow around a cylinder*, Notes on Numerical Fluid Mechanics 52 (1996) 547–566,
#   DOI 10.1007/978-3-322-89849-4_39.
# - A finer mesh: `julia --project=. data/models/turek.jl 0.01` writes `data/models/turek2d_h0.01.msh`.
