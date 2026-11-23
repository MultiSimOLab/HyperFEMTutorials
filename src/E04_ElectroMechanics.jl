# # Tutorial 04: Electro-mechanics of a dielectric elastomer
#
# A dielectric elastomer plate is clamped at one end and carries two compliant electrodes, one at mid-thickness
# and one on the top face. A voltage between them squeezes the upper layer across its thickness; being nearly
# incompressible, the layer elongates, while the lower layer (without field) does not: the plate bends like a
# bimorph. Mechanics and electrostatics are fully coupled and solved together (monolithic scheme), with the weak
# form written term by term.
#
# **What you will learn**
# - electro-mechanical constitutive models (`ElectroMechModel`, `IdealDielectric`) and their derivatives;
# - multi-field spaces (displacement and electric potential) and their conditions (`MultiFieldBC`);
# - the coupled weak form and its consistent linearisation written by hand with `⊙`, `⋅` and `⋅²`;
# - the same forms from HyperFEM's helpers `residual` and `jacobian`, and how to check that they agree.
#
# Measured on a Mac M5 Max (Julia 1.13.0, 4 threads, 4 BLAS threads; HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8; compiled backend), whole run including compilation:
# FAST 53 s, full 54 s (20 increments, 113 Newton iterations, 14 s in `solve!` including its compilation);
# 1 528 free dofs. One assembly of the tangent: 0.7 ms compiled, 11 ms with Gridap's lazy evaluation.
#
# ## Problem statement
# The body ``\Omega`` is described in its reference configuration by the displacement ``\boldsymbol u`` and the
# electric potential ``\varphi``. With the deformation gradient ``\boldsymbol F = \boldsymbol I + \nabla_0 \boldsymbol u``
# and the nominal electric field ``\boldsymbol E_0 = -\nabla_0 \varphi``, the material is described by a free
# energy ``\Psi(\boldsymbol F, \boldsymbol E_0)``: the first Piola stress is ``\boldsymbol P = \partial_{\boldsymbol F}\Psi``
# and the nominal electric displacement ``\boldsymbol D_0 = -\partial_{\boldsymbol E_0}\Psi``.
# Equilibrium and Gauss's law, ``\mathrm{Div}\,\boldsymbol P = \boldsymbol 0`` and ``\mathrm{Div}\,\boldsymbol D_0 = 0``,
# in weak form: find ``(\boldsymbol u, \varphi)`` such that for all ``(\boldsymbol v, \delta\varphi)``
# ```math
# R = \int_\Omega \nabla_0 \boldsymbol v : \partial_{\boldsymbol F}\Psi \,\mathrm dV
#   - \int_\Omega \nabla_0 \delta\varphi \cdot \partial_{\boldsymbol E_0}\Psi \,\mathrm dV = 0 .
# ```
# Its directional derivative along ``(\Delta\boldsymbol u, \Delta\varphi)`` is the tangent used by Newton's method,
# ```math
# \begin{aligned}
# K = {} & \int_\Omega \nabla_0 \boldsymbol v : \partial^2_{\boldsymbol F\boldsymbol F}\Psi : \nabla_0 \Delta\boldsymbol u \,\mathrm dV
#   + \int_\Omega \nabla_0 \delta\varphi \cdot \partial^2_{\boldsymbol E_0\boldsymbol E_0}\Psi \, \nabla_0 \Delta\varphi \,\mathrm dV \\
#   & - \int_\Omega \nabla_0 \Delta\varphi \cdot \partial^2_{\boldsymbol E_0\boldsymbol F}\Psi : \nabla_0 \boldsymbol v \,\mathrm dV
#   - \int_\Omega \nabla_0 \delta\varphi \cdot \partial^2_{\boldsymbol E_0\boldsymbol F}\Psi : \nabla_0 \Delta\boldsymbol u \,\mathrm dV .
# \end{aligned}
# ```
# The free energy is the sum of a compressible neo-Hookean energy (tutorial 02) and the energy of an ideal
# dielectric, written with the cofactor ``\boldsymbol H = J\boldsymbol F^{-T}`` and ``J = \det\boldsymbol F``:
# ```math
# \Psi = \tfrac{\mu}{2}(\boldsymbol F:\boldsymbol F - 3) - \mu \ln J + \tfrac{\lambda}{2}(J-1)^2
#      - \frac{\varepsilon}{2J}\,|\boldsymbol H \boldsymbol E_0|^2 ,
# ```
# with ``\mu = 1``, ``\lambda = 10`` and ``\varepsilon = 1`` in non-dimensional units (stresses in units of ``\mu``,
# fields in units of ``\sqrt{\mu/\varepsilon}``). The plate is ``8 \times 0.8 \times 40``; the face ``z = 0`` is
# clamped, ``\varphi = 0`` on the mid-plane ``y = 0.4`` and ``\varphi = \Lambda V`` on the top face ``y = 0.8``, with the
# load factor ``\Lambda`` going from 0 to 1. The rest of the boundary is free of tractions and of charge.
#
# ## Setup
using HyperFEM
using Gridap
using HyperFEMKernels                                          # compiled backend (`backend=:compiled`)
using LinearAlgebra: norm

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E04_ElectroMechanics")
setupfolder(outdir)                                            # VTK output (not versioned)

# ## Geometry and mesh
# A Gmsh mesh of 266 trilinear hexahedra (7 across the width, 19 along the length, one per layer across the
# thickness) with the physical groups `fixedup` (clamped face), `midsuf` (mid-plane electrode) and `topsuf`
# (top electrode). We add a tag for the free end ``z = 40`` from the
# coordinates of its vertices, to measure the deflection.
model = GmshDiscreteModel(joinpath(@__DIR__, "..", "data", "models", "test_static_EM.msh"))
labels = get_face_labeling(model)
add_tag_from_vertex_filter!(labels, "tip", model, x -> abs(x[3] - 40.0) < 1e-8)

# ## Constitutive model
# Calling an electro-mechanical model returns the energy and its derivatives as functions of ``(\boldsymbol F,
# \boldsymbol E_0)``: ``(\Psi, \partial_{\boldsymbol F}\Psi, \partial_{\boldsymbol E_0}\Psi,
# \partial^2_{\boldsymbol F\boldsymbol F}\Psi, \partial^2_{\boldsymbol E_0\boldsymbol F}\Psi,
# \partial^2_{\boldsymbol E_0\boldsymbol E_0}\Psi)``. The kinematic objects give ``\boldsymbol F(\nabla_0\boldsymbol u)``
# and ``\boldsymbol E_0(\nabla_0\varphi)``.
physmodel = ElectroMechModel(mechano=NeoHookean3D(λ=10.0, μ=1.0), electro=IdealDielectric(ε=1.0))
Ψ, ∂ΨF, ∂ΨE, ∂ΨFF, ∂ΨEF, ∂ΨEE = physmodel()
km = Kinematics(Mechano, Solid)
ke = Kinematics(Electro, Solid)
F, _, _ = get_Kinematics(km)
E = get_Kinematics(ke)

# ## Boundary conditions
# One set of conditions per field, gathered by `MultiFieldBC`. The clamp is constant; the two electrodes
# follow the load factor (``\varphi = 0`` on `midsuf` for every ``\Lambda``, ``\varphi = \Lambda V`` on `topsuf`).
V0 = FAST ? 0.1 : 0.3                                          # voltage between the electrodes
nsteps = FAST ? 5 : 20
Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]))
Dφ = BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", V0, Λ -> Λ))
D_bc = MultiFieldBC([Du, Dφ])

# ## Finite element spaces
# Trilinear elements for both fields. The multi-field spaces stack the two fields; a function of `U` is the
# pair ``(\boldsymbol u, \varphi)``.
order = 1
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)
Vu = TestFESpace(model, reffeu, D_bc[1], conformity=:H1)
Vφ = TestFESpace(model, reffeφ, D_bc[2], conformity=:H1)
Uu = TrialFESpace(Vu, D_bc[1], 0.0)
Uφ = TrialFESpace(Vφ, D_bc[2], 0.0)
V = MultiFieldFESpace([Vu, Vφ])
U = MultiFieldFESpace([Uu, Uφ])
Ω = Triangulation(model)
dΩ = Measure(Ω, 2 * order)
println("free dofs: ", num_free_dofs(U), " (u: ", num_free_dofs(Uu), ", φ: ", num_free_dofs(Uφ), ")")

# ## Weak form
# The residual and the tangent above, term by term. The constitutive functions are composed (`∘`) with the
# kinematics evaluated at the trial functions. `⊙` contracts two second-order tensors (``\boldsymbol A:\boldsymbol B``),
# `⋅` a vector with a vector or a second-order tensor, and `⋅²` the last two indices of a higher-order tensor with
# a second-order one: ``(\mathbb C \cdot^2 \boldsymbol A)_{ij} = C_{ijkl}A_{kl}`` and
# ``(\mathcal E \cdot^2 \boldsymbol A)_{i} = \mathcal E_{ikl}A_{kl}``.
res(Λ) = ((u, φ), (v, vφ)) ->
  ∫(∇(v)' ⊙ (∂ΨF ∘ (F ∘ ∇(u)', E ∘ ∇(φ))))dΩ -
  ∫(∇(vφ) ⋅ (∂ΨE ∘ (F ∘ ∇(u)', E ∘ ∇(φ))))dΩ

jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) ->
  ∫(∇(v)' ⊙ ((∂ΨFF ∘ (F ∘ ∇(u)', E ∘ ∇(φ))) ⋅² ∇(du)'))dΩ +
  ∫(∇(vφ) ⋅ ((∂ΨEE ∘ (F ∘ ∇(u)', E ∘ ∇(φ))) ⋅ ∇(dφ)))dΩ -
  ∫(∇(dφ) ⋅ ((∂ΨEF ∘ (F ∘ ∇(u)', E ∘ ∇(φ))) ⋅² ∇(v)'))dΩ -
  ∫(∇(vφ) ⋅ ((∂ΨEF ∘ (F ∘ ∇(u)', E ∘ ∇(φ))) ⋅² ∇(du)'))dΩ

# HyperFEM builds exactly these forms from the model and the pair of kinematics: in later tutorials we use
# the helpers. We check below that both give the same vectors and matrices.
k = (km, ke)
res_helper(Λ) = (x, y) -> residual(physmodel, k, x, y, dΩ)
jac_helper(Λ) = (x, dx, y) -> jacobian(physmodel, k, x, dx, y, dΩ)

# ## Solver and computational model
# `backend=:compiled` (tutorial 01): the hand-written forms are compiled once into threaded kernels.
nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8, verbose=true)
comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls, backend=:compiled)
xh = get_state(comp_model)                                     # (u, φ) of the current state
uh, φh = xh

# ## Post-processing
# After every converged increment: the voltage, the mean deflection ``u_y`` of the free end, and the VTK files.
Γtip = BoundaryTriangulation(model, tags="tip")
dΓtip = Measure(Γtip, 2 * order)
Atip = sum(∫(1.0)dΓtip)
ey = VectorValue(0.0, 1.0, 0.0)
voltage, tip_uy = Float64[], Float64[]
function record(post)
  Λ = post.Λ[post.iter]
  push!(voltage, Λ * V0)
  push!(tip_uy, sum(∫(uh ⋅ ey)dΓtip) / Atip)
  pvd = get_pvd(post)
  pvd[Λ] = createvtk(Ω, joinpath(outdir, "step_$(post.iter).vtu"), cellfields=["u" => uh, "φ" => φh])
end
post = PostProcessor(comp_model, record; is_vtk=true, filepath=outdir)

# ## Solve
# The potential grows with ``\Lambda``; `ProjectDirichlet=true` (tutorial 02) starts each increment from a linear
# prediction of the effect of the new electrode voltage on both fields.
x, flag = solve!(comp_model; stepping=LoadStepping(nsteps=nsteps, maxbisec=5), ProjectDirichlet=true, post=post)

for (vo, w) in zip(voltage, tip_uy)
  println("V = $(round(vo, digits=4))   mean tip deflection u_y = $(round(w, sigdigits=8))")
end

# ## Checks
# **Hand-written forms against the helpers.** At a state away from equilibrium (the solution scaled by 1.1), the
# residual vectors and the tangent matrices of the two formulations are the same up to round-off.
xt = FEFunction(U, 1.1 .* get_free_dof_values(xh))
r1 = assemble_vector(y -> res(1.0)(xt, y), V)
r2 = assemble_vector(y -> res_helper(1.0)(xt, y), V)
K1 = assemble_matrix((dx, y) -> jac(1.0)(xt, dx, y), U, V)
K2 = assemble_matrix((dx, y) -> jac_helper(1.0)(xt, dx, y), U, V)
err_r = norm(r1 - r2) / norm(r1)
err_K = norm(K1 - K2) / norm(K1)
println("hand-written vs helper: residual $(err_r), tangent $(err_K)")
@assert err_r < 1e-12 "hand-written residual differs from residual(physmodel, ...)"
@assert err_K < 1e-12 "hand-written tangent differs from jacobian(physmodel, ...)"

# **Physics.** The load stepping reached the full voltage. The plate bends towards ``-y``, away from the
# active layer, by an amount that grows with the voltage; at small voltages the electrostatic stresses, and with
# them the deflection, grow as ``V^2``: doubling the voltage of the first increment multiplies the deflection by 4.
# Geometry, material and loads are symmetric with respect to the plane ``x = 4``, and so is the solution: the
# mean of ``u_x`` vanishes.
Vol = sum(∫(1.0)dΩ)
ux_mean = sum(∫(uh ⋅ VectorValue(1.0, 0.0, 0.0))dΩ) / Vol
ratio = tip_uy[2] / tip_uy[1]
println("tip u_y = $(tip_uy[end])   u_y(2ΔV)/u_y(ΔV) = $(ratio)   mean u_x = $(ux_mean)")
@assert flag == 1 "the load stepping did not reach Λ = 1"
@assert all(diff(tip_uy) .< 0) && tip_uy[end] < 0 "the plate does not bend towards -y"
@assert abs(ratio - 4) < 0.05 "the deflection does not grow as V² at small voltages (ratio $ratio)"
@assert abs(ux_mean) < 1e-8 * abs(tip_uy[end]) "the solution is not symmetric about x = 4"

# **Regression** (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0, macOS arm64): norm of the
# final displacement (free values) and mean deflection of the free end.
nu = norm(get_free_dof_values(uh))
println("‖u‖ = $(nu)")
nu_ref, tip_ref = FAST ? (45.90961930695172, -4.4581346161107) : (394.05924551948493, -31.60568504003123)
@assert isapprox(nu, nu_ref; rtol=1e-5) "‖u‖ = $nu differs from the reference $nu_ref"
@assert isapprox(tip_uy[end], tip_ref; rtol=1e-5) "tip deflection $(tip_uy[end]) differs from the reference $tip_ref"

# ## Going further
# - Raise the voltage: the stiffness of the plate drops as the field grows and, past a critical voltage, the
#   Newton iterations fail (electro-mechanical instability). In HyperFEM 0.3.0 such a failure can end the run
#   with an uncaught `SingularException` of the LU factorisation instead of a bisection of the increment.
#   `HessianRegularization(physmodel)` and the hybrid Newton of tutorial 06 are the tools to go beyond it.
# - Solve the two fields one after the other instead of together: `StaggeredModel` (tutorial 09).
# - Tutorial 05 adds a fibre family to the same plate; tutorial 13 replaces `LUSolver` by block-preconditioned
#   Krylov solvers (`AMGSolver(U)`, `PETScSolver(U)`) for this kind of multi-field problem.
