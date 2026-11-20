# # Tutorial 02: Uniaxial stretch of a hyperelastic bar
#
# A neo-Hookean bar is stretched to 1.75 times its length while its lateral faces are free to contract.
# The deformation is homogeneous, so the force–stretch curve has a closed form: the first nonlinear
# tutorial is also a validation of the whole pipeline (model, weak form, Newton, load stepping).
#
# **What you will learn**
# - choosing a constitutive model (`NeoHookean3D`) and getting its weak form (`residual`, `jacobian`);
# - Dirichlet conditions by components (`BoundaryCondition(...; mask)`) that grow with the load factor;
# - `StaticNonlinearModel`, `NewtonRaphson` and the load increments of `LoadStepping`;
# - recording results at every increment with a `PostProcessor`.
#
# ## Problem statement
# Find the displacement ``\boldsymbol u`` of the bar ``\Omega = (0,L)\times(0,W)\times(0,T)`` such that
# ```math
# \int_\Omega \boldsymbol P(\boldsymbol F) : \nabla_0 \boldsymbol v \,\mathrm dV = 0 \quad \forall \boldsymbol v ,
# \qquad \boldsymbol F = \boldsymbol I + \nabla_0 \boldsymbol u , \qquad \boldsymbol P = \partial_{\boldsymbol F}\Psi ,
# ```
# with the compressible neo-Hookean energy
# ``\Psi = \tfrac{\mu}{2}(\boldsymbol F:\boldsymbol F - 3) - \mu \ln J + \tfrac{\lambda}{2}(J-1)^2``, ``J = \det \boldsymbol F``.
# The face ``x = 0`` cannot move along ``x``, the faces ``y = 0`` and ``z = 0`` are symmetry planes, and the face
# ``x = L`` is pulled: ``u_x = \Lambda\,\delta``, with the load factor ``\Lambda`` going from 0 to 1.
#
# With these conditions ``\boldsymbol F = \mathrm{diag}(\lambda_1, \lambda_t, \lambda_t)`` is uniform. The
# lateral faces are free of traction (``P_{22} = 0``), which gives ``J = \lambda_1\lambda_t^2`` as the positive
# root of ``\lambda J^2 - (\lambda - \mu/\lambda_1) J - \mu = 0``, and the nominal stress
# ``P_{11} = \mu\,(\lambda_1 - J/\lambda_1^2)``.
#
# ## Setup
using HyperFEM
using Gridap

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E02_Hyperelastic_stretch")
setupfolder(outdir)                                            # VTK output (not versioned)

# ## Geometry and mesh
# A Cartesian mesh of trilinear hexahedra. `CartesianTags` names the faces of the box; the `⁺` variants
# include the edges and vertices of the face.
L, W, T = 1.0, 0.2, 0.5                                        # m
partition = FAST ? (8, 2, 4) : (20, 4, 10)
model = CartesianDiscreteModel((0, L, 0, W, 0, T), partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "x0", CartesianTags.face0YZ⁺)
add_tag_from_tags!(labels, "x1", CartesianTags.face1YZ⁺)
add_tag_from_tags!(labels, "y0", CartesianTags.faceX0Z)
add_tag_from_tags!(labels, "z0", CartesianTags.faceXY0)

# ## Constitutive model
# Every model of HyperFEM is a small struct; calling it returns the energy and its analytical
# derivatives ``(\Psi, \partial_{\boldsymbol F}\Psi, \partial^2_{\boldsymbol F\boldsymbol F}\Psi)``.
λ, μ = 100.0, 1.0                                              # Pa (nearly incompressible, ν ≈ 0.495)
physmodel = NeoHookean3D(λ=λ, μ=μ)
kinematics = Kinematics(Mechano, Solid)

# ## Boundary conditions
# A `BoundaryCondition` is a tag, a value and (optionally) a function of the load factor; `mask` selects
# the constrained components. Conditions without a function are constant.
δ = 0.75 * L                                                   # final elongation: λ₁ = 1.75
D_bc = BoundaryConditions(
  BoundaryCondition("x0", [0.0, 0.0, 0.0]; mask=(true, false, false)),
  BoundaryCondition("x1", [δ, 0.0, 0.0], EvolutionFunctions.ramp(); mask=(true, false, false)),
  BoundaryCondition("y0", [0.0, 0.0, 0.0]; mask=(false, true, false)),
  BoundaryCondition("z0", [0.0, 0.0, 0.0]; mask=(false, false, true)))

# ## Finite element spaces
order = 1
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
Ω = Triangulation(model)
dΩ = Measure(Ω, 2 * order)
V = TestFESpace(Ω, reffe, D_bc, conformity=:H1)
U = TrialFESpace(V, D_bc, 0.0)                                 # Dirichlet values at Λ = 0

# ## Weak form
# `residual` and `jacobian` build the forms above from the model. They are functions of the load factor
# because other problems have loads or material parameters that depend on it; here they do not.
res(Λ) = (u, v) -> residual(physmodel, kinematics, u, v, dΩ)
jac(Λ) = (u, du, v) -> jacobian(physmodel, kinematics, u, du, v, dΩ)

# ## Solver and computational model
nls = NewtonRaphson(LUSolver(); maxiter=20, rtol=1e-10, verbose=true)
comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls)
uh = get_state(comp_model)                                     # FE function of the current state

# ## Post-processing
# The function below runs after every converged increment: it stores the stretch and the force on the
# pulled face, ``\int_{\Gamma} (\boldsymbol P \boldsymbol N)_x \,\mathrm dA``, and writes the displacement.
_, ∂Ψ∂F, _ = physmodel()
F, _, _ = get_Kinematics(kinematics)
Γ = BoundaryTriangulation(model, tags="x1")
dΓ = Measure(Γ, 2 * order)
N = get_normal_vector(Γ)
stretch, force = Float64[], Float64[]
function record(post)
  Λ = post.Λ[post.iter]
  push!(stretch, 1 + Λ * δ / L)
  push!(force, sum(∫((∂Ψ∂F ∘ (F ∘ ∇(uh)')) ⋅ N)dΓ)[1])
  pvd = get_pvd(post)
  pvd[Λ] = createvtk(Ω, joinpath(outdir, "step_$(post.iter).vtu"), cellfields=["u" => uh])
end
post = PostProcessor(comp_model, record; is_vtk=true, filepath=outdir)

# ## Solve
# Five equal increments; an increment whose Newton iteration fails is halved (three times in a row at most).
# `ProjectDirichlet=true` starts every increment from a linear prediction of the effect of the new boundary
# values; without it Newton starts with only the boundary nodes moved and diverges for increments this large.
x, flag = solve!(comp_model; stepping=LoadStepping(nsteps=5, maxbisec=3), ProjectDirichlet=true, post=post)
vtk_save(post)

# ## Checks
# The numerical force–stretch curve against the closed form (the mesh represents the homogeneous
# deformation exactly, so the difference is the Newton tolerance).
J_exact(λ₁) = ((λ - μ / λ₁) + sqrt((λ - μ / λ₁)^2 + 4λ * μ)) / (2λ)
P11_exact(λ₁) = μ * (λ₁ - J_exact(λ₁) / λ₁^2)
force_exact = P11_exact.(stretch) .* (W * T)
for (s, f, fe) in zip(stretch, force, force_exact)
  println("λ₁ = $(round(s, digits=2))   force = $(round(f, sigdigits=8)) N   exact = $(round(fe, sigdigits=8)) N")
end
@assert flag == 1 "the load stepping did not reach Λ = 1"
@assert length(stretch) == 5 "an increment was bisected"
@assert maximum(abs.(force .- force_exact) ./ force_exact) < 1e-6 "force–stretch curve away from the closed form"
dΩe = Measure(Ω, 4)
λt = sqrt(J_exact(1 + δ / L) / (1 + δ / L))                     # lateral stretch at the end
u_exact(X) = VectorValue(δ * X[1] / L, (λt - 1) * X[2], (λt - 1) * X[3])
err = sqrt(sum(∫((uh - u_exact) ⋅ (uh - u_exact))dΩe))
@assert err < 1e-6 * δ "displacement away from the homogeneous solution (error $err)"

# ## Going further
# - Clamp both ends (remove the masks of `x0` and `x1`): the deformation is no longer homogeneous and the
#   bar necks; compare the force with the curve above.
# - Try `MooneyRivlin3D` or `Yeoh3D` instead of `NeoHookean3D`: nothing else changes.
# - `LoadStepping(nsteps=50, grow=1.5, ΔΛmax=0.2)` starts with small increments and lets them grow.
# - Tutorial 03 adds fibres and a pressure that follows the deformation; tutorial 13 replaces `LUSolver`.
