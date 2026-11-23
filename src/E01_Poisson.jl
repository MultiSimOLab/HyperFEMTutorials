# # Tutorial 01: Poisson equation
#
# The Poisson equation is the simplest elliptic problem and the place to meet the pieces that every
# HyperFEM simulation shares: a mesh, boundary conditions, finite element spaces, a weak form and a
# computational model that assembles and solves it. The source term is manufactured from a known
# solution, so the error can be measured and the order of convergence of the method checked.
#
# **What you will learn**
# - Dirichlet data given by a function of the coordinates (`BoundaryConditions`, `BoundaryCondition`);
# - finite element spaces built from the conditions (`TestFESpace`, `TrialFESpace`);
# - a linear problem as a `StaticLinearModel`, its solution with `solve!` and the solution field (`get_state`);
# - a second right-hand side solved with the same factorised matrix (`solve!(model, b; Assembly=false)`);
# - measuring the ``L^2`` error and the order of convergence.
#
# Measured on a Mac M5 Max (Julia 1.13.0, 4 threads, 4 BLAS threads; HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8; compiled backend), whole run including compilation:
# full case (meshes 50×20 and 100×40, 3 900 free dofs on the finest) 26 s, FAST case (20×8 and 40×16) 27 s;
# almost all of it is compilation (each linear solve takes milliseconds).
#
# ## Problem statement
# Find ``u`` in the rectangle ``\Omega = (-\pi,\pi)\times(-\pi/2,\pi/2)`` such that
# ```math
# -\Delta u = f \ \text{in } \Omega, \qquad u = u_0 \ \text{on } \Gamma_D, \qquad
# \nabla u\cdot\boldsymbol n = g \ \text{on } \Gamma_N ,
# ```
# where ``\Gamma_N`` is the right side ``x = \pi`` and ``\Gamma_D`` the other three. The data come from the
# manufactured solution ``u_0(x,y) = \cos x \,\sin(y+\pi)``: ``f = -\Delta u_0 = 2u_0`` and
# ``g = \nabla u_0\cdot\boldsymbol n``. The weak form reads: find ``u`` with ``u = u_0`` on ``\Gamma_D`` such that
# ```math
# a(u,v) = \int_\Omega \nabla u\cdot\nabla v \,\mathrm d\Omega
# = \int_\Omega f\,v \,\mathrm d\Omega + \int_{\Gamma_N} g\,v \,\mathrm d\Gamma = l(v)
# \qquad \forall v,\ v = 0 \text{ on } \Gamma_D .
# ```
# With bilinear elements (Q1) and a smooth solution, the ``L^2`` error decreases as ``h^2``.
#
# ## Setup
using HyperFEM
using Gridap
using HyperFEMKernels                                          # compiled backend (`backend=:compiled`)

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E01_Poisson")
setupfolder(outdir)                                            # VTK output (not versioned)

# ## Exact solution and data
# The gradient and the Laplacian are written by hand. `VectorValue` is Gridap's small vector type.
u₀(x) = cos(x[1]) * sin(x[2] + π)
∇u₀(x) = VectorValue(-sin(x[1]) * sin(x[2] + π), cos(x[1]) * cos(x[2] + π))
f(x) = 2 * u₀(x)                                               # f = -Δu₀

# ## Geometry and mesh
# A Cartesian mesh of bilinear quadrilaterals. In 2D, Gridap names the four vertices `tag_1`…`tag_4`
# and the sides `tag_5` (``y = -\pi/2``), `tag_6` (``y = \pi/2``), `tag_7` (``x = -\pi``) and
# `tag_8` (``x = \pi``). The Dirichlet tag takes the vertices too, the Neumann tag only the open side.
domain = (-π, π, -π / 2, π / 2)
partitions = FAST ? [(20, 8), (40, 16)] : [(50, 20), (100, 40)]
model = CartesianDiscreteModel(domain, partitions[1])
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "dirichlet", [1, 2, 3, 4, 5, 6, 7])
add_tag_from_tags!(labels, "neumann", [8])

# ## Boundary conditions
# A `BoundaryCondition` is a tag and a value: a number, a vector or, as here, a function of the
# coordinates. `BoundaryConditions` groups the conditions of a problem.
D_bc = BoundaryConditions(BoundaryCondition("dirichlet", u₀))

# ## Finite element spaces
# The test space vanishes on the Dirichlet tags; the trial space takes the values of the conditions.
order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
Ω = Triangulation(model)
dΩ = Measure(Ω, 2 * order)
Γ = BoundaryTriangulation(model, tags="neumann")
dΓ = Measure(Γ, 2 * order)
nΓ = get_normal_vector(Γ)                                       # outward unit normal on Γ_N
V = TestFESpace(Ω, reffe, D_bc, conformity=:H1)
U = TrialFESpace(V, D_bc)
println("free dofs: ", num_free_dofs(V))

# ## Weak form
# Plain Gridap syntax: ``a`` and ``l`` are Julia functions that return integrals. A function of the
# coordinates (here `f` and `∇u₀`) can be combined with fields such as the normal `nΓ`.
# For this ``u_0`` the flux ``g`` on ``x = \pi`` happens to vanish; the second load case below has a non-zero one.
a(u, v) = ∫(∇(u) ⋅ ∇(v))dΩ
l(v) = ∫(f * v)dΩ + ∫((∇u₀ ⋅ nΓ) * v)dΓ

# ## Computational model and solution
# `StaticLinearModel(l, a, U, V, D_bc)` assembles the matrix and the right-hand side (with the
# Dirichlet values moved to it) and factorises the matrix with the default `LUSolver()`.
# `solve!` returns the vector of free values; `get_state` wraps it as a finite element function.
# `Assembly=false` uses the system assembled by the constructor instead of assembling it again.
# `backend=:compiled` assembles with the compiled, threaded kernels of HyperFEMKernels (loaded above) instead of
# Gridap's lazy cell-by-cell evaluation: the weak form is turned into Julia code once, when the model is built.
# The result is the same. On a mesh this small the gain is invisible, but every tutorial uses it, because it
# grows with the mesh and matters most in nonlinear problems, which assemble at every Newton iteration.
comp_model = StaticLinearModel(l, a, U, V, D_bc; backend=:compiled)
solve!(comp_model; Assembly=false)
uh = get_state(comp_model)

# ## Error
# The error is integrated with a higher quadrature degree than the one of the weak form.
dΩe = Measure(Ω, 4 * order)
l2(e) = sqrt(sum(∫(e * e)dΩe))
error_0 = l2(uh - u₀)
println("L2 error, u₀: ", error_0, "   (relative to ‖u₀‖: ", error_0 / l2(CellField(u₀, Ω)), ")")
writevtk(Ω, joinpath(outdir, "poisson_u0"), cellfields=["uh" => uh, "error" => uh - u₀])

# ## A second right-hand side with the same factorisation
# A new load case on the same mesh, with the same Dirichlet data: ``u_1 = u_0 + w`` with
# ``w = (x+\pi)\cos y``, which vanishes on ``\Gamma_D``. Its data are ``f_1 = f - \Delta w = f + (x+\pi)\cos y``
# and ``g_1 = \nabla u_1\cdot\boldsymbol n`` (now ``\cos y`` on ``x = \pi``).
u₁(x) = u₀(x) + (x[1] + π) * cos(x[2])
∇u₁(x) = ∇u₀(x) + VectorValue(cos(x[2]), -(x[1] + π) * sin(x[2]))
f₁(x) = f(x) + (x[1] + π) * cos(x[2])
l₁(v) = ∫(f₁ * v)dΩ + ∫((∇u₁ ⋅ nΓ) * v)dΓ
# The matrix does not change, so only the right-hand side is assembled. It must include the Dirichlet
# lift, ``b_1 = l_1(v) - a(u_D, v)``, where ``u_D`` = `zero(U)` is the function of the trial space with
# zero free values (only its Dirichlet values are non-zero). `solve!(model, b; Assembly=false)` then
# solves with the matrix already factorised: no assembly and no new factorisation.
uD = zero(U)
b₁ = assemble_vector(v -> l₁(v) - a(uD, v), V)
x₁ = solve!(comp_model, b₁; Assembly=false)
uh₁ = get_state(comp_model)
# `get_state` wraps the model's own vector of free values, which `solve!` overwrites: `uh` above now holds the
# second solution too (its error and VTK file were taken before). To keep a solution, copy its values:
# `FEFunction(U, copy(get_free_dof_values(uh)))`.
error_1 = l2(uh₁ - u₁)
println("L2 error, u₁: ", error_1, "   (relative to ‖u₁‖: ", error_1 / l2(CellField(u₁, Ω)), ")")
# The same problem assembled and solved from scratch with Gridap's `AffineFEOperator`, for comparison:
x₁_ref = get_free_dof_values(solve(LUSolver(), AffineFEOperator(a, l₁, U, V)))
writevtk(Ω, joinpath(outdir, "poisson_u1"), cellfields=["uh" => uh₁, "error" => uh₁ - u₁])

# ## Convergence
# The same steps on a mesh twice as fine in each direction. With ``h`` halved, the observed order is
# ``\log_2(e_h / e_{h/2})``.
function l2_error_on(partition)
  model = CartesianDiscreteModel(domain, partition)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "dirichlet", [1, 2, 3, 4, 5, 6, 7])
  add_tag_from_tags!(labels, "neumann", [8])
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2 * order)
  Γ = BoundaryTriangulation(model, tags="neumann")
  dΓ = Measure(Γ, 2 * order)
  nΓ = get_normal_vector(Γ)
  V = TestFESpace(Ω, reffe, D_bc, conformity=:H1)
  U = TrialFESpace(V, D_bc)
  a(u, v) = ∫(∇(u) ⋅ ∇(v))dΩ
  l(v) = ∫(f * v)dΩ + ∫((∇u₀ ⋅ nΓ) * v)dΓ
  m = StaticLinearModel(l, a, U, V, D_bc; backend=:compiled)
  solve!(m; Assembly=false)
  e = get_state(m) - u₀
  sqrt(sum(∫(e * e)Measure(Ω, 4 * order))), num_free_dofs(V)
end
error_fine, ndofs_fine = l2_error_on(partitions[2])
rate = log2(error_0 / error_fine)
println("L2 error on $(partitions[2]) ($ndofs_fine free dofs): ", error_fine, "   observed order: ", rate)

# ## Checks
# The order of convergence is the one of Q1 elements and the errors are of the size expected on these
# meshes. The second load case, solved with the factorisation of the first, is the solution of a fresh
# assembly to round-off.
@assert abs(rate - 2) < 0.2 "observed order $rate, expected 2"
@assert error_0 < (FAST ? 5e-2 : 1e-2) "L2 error too large: $error_0"
@assert error_1 / l2(CellField(u₁, Ω)) < (FAST ? 2e-2 : 4e-3) "L2 error of the second load case too large: $error_1"
@assert maximum(abs.(x₁ .- x₁_ref)) < 1e-10 * maximum(abs.(x₁_ref)) "re-used factorisation ≠ fresh solve"

# ## Going further
# - Use `order = 2` (biquadratic elements): the observed order becomes 3.
# - Solve several load cases at once: `StaticLinearModel([l, l₁], a, U, V, D_bc)` keeps one matrix
#   for a vector of linear forms (the load vectors are assembled without the Dirichlet lift).
# - Tutorial 02 moves to nonlinear elasticity: the same spaces and conditions, with a Newton solver.
