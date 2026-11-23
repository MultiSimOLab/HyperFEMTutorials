# # Tutorial 03: Fibre-reinforced cylinder under internal pressure
#
# A thick-walled tube, clamped at both ends, is inflated by an internal pressure. Its wall is a
# soft isotropic matrix reinforced by four families of collagen-like fibres (axial, circumferential and
# two helical ones), the arrangement used to model arteries. Two features of soft-tissue problems
# appear here: anisotropic models with fibre directions that vary in space, and a pressure that
# follows the deformed surface, whose weak form and tangent are written by hand.
#
# **What you will learn**
# - a quarter of a tube from a Cartesian mesh and a polar `map`, with symmetry planes imposed by `mask`;
# - summing an isotropic and an anisotropic model (`MooneyRivlin3D() + HGO_4Fibers()`);
# - fibre directions as finite element fields, passed to the model as extra arguments;
# - a follower pressure: its weak form with the cofactor ``\boldsymbol H`` and its consistent tangent;
# - checking the setup against the Lamé solution before trusting the nonlinear results.
#
# Measured on a Mac M5 Max (Julia 1.13.0, 4 threads, 4 BLAS threads; HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8; compiled backend), whole run including compilation:
# full case (triquadratic Q2 elements, mesh 2×12×30, 21 535 free dofs, 20 increments of 4 Newton iterations)
# 51 s, of which 16 s in `solve!` (0.8 s per increment, mostly the LU factorisations, 0.16 s each); FAST case
# (1×8×20, 5 733 free dofs, 4 increments to 20 % of the pressure) 40 s. One assembly of the tangent on the full
# mesh: 0.016 s compiled, 0.49 s with Gridap's lazy evaluation.
#
# ## Problem statement
# The tube has inner radius ``R_i``, outer radius ``R_o`` and length ``L``, both ends are clamped and the
# inner surface ``\Gamma_i`` carries the pressure ``p`` in the deformed configuration. Find the
# displacement ``\boldsymbol u`` such that
# ```math
# \int_\Omega \boldsymbol P(\boldsymbol F) : \nabla_0 \boldsymbol v \,\mathrm dV
# + \Lambda\, p \int_{\Gamma_i} (\boldsymbol H \boldsymbol N)\cdot \boldsymbol v \,\mathrm dA = 0
# \quad \forall \boldsymbol v , \qquad \boldsymbol H = \operatorname{cof}\boldsymbol F = J\boldsymbol F^{-T},
# ```
# with the load factor ``\Lambda`` going from 0 to 1. The second term is minus the virtual work of the pressure,
# ``+\int_{\gamma_i} p\,\boldsymbol n\cdot\boldsymbol v\,\mathrm da`` (the pressure acts along ``-\boldsymbol n``), pulled back to the reference surface with
# Nanson's formula ``\boldsymbol n\,\mathrm da = \boldsymbol H\boldsymbol N\,\mathrm dA``; ``\boldsymbol N`` is the
# outward normal of the solid (it points to the axis on ``\Gamma_i``). Because ``\boldsymbol H`` depends on
# ``\boldsymbol u``, the load contributes to the tangent: with ``D\boldsymbol H[\delta\boldsymbol u] = \boldsymbol F \times
# \nabla_0\delta\boldsymbol u`` (the tensor cross product of Bonet, Gil and Ortigosa, `×` in HyperFEM),
# ```math
# \Lambda\, p \int_{\Gamma_i} \big((\boldsymbol F\times\nabla_0\delta\boldsymbol u)\,\boldsymbol N\big)\cdot\boldsymbol v \,\mathrm dA .
# ```
#
# The strain energy is a Mooney–Rivlin matrix plus four fibre families of Holzapfel–Gasser–Ogden type
# (Holzapfel, Gasser and Ogden 2000, [doi:10.1023/A:1010835316564](https://doi.org/10.1023/A:1010835316564)),
# ```math
# \Psi = \frac{\mu_1}{2}\boldsymbol F:\boldsymbol F + \frac{\mu_2}{2}\boldsymbol H:\boldsymbol H
# - (\mu_1+2\mu_2)\ln J + \frac{\lambda}{2}(J-1)^2 - \frac{3}{2}(\mu_1+\mu_2)
# + \sum_{a=1}^{4} \frac{c_{1a}}{4c_{2a}}\left(e^{c_{2a}(I_{4a}-1)^2} - 1\right),
# \qquad I_{4a} = |\boldsymbol F\boldsymbol M_a|^2 ,
# ```
# with unit fibre directions ``\boldsymbol M_a``: axial ``\boldsymbol e_z``, circumferential ``\boldsymbol e_\theta``
# and the two helices ``(\boldsymbol e_\theta \pm \boldsymbol e_z)/\sqrt 2``. Units: cm and Pa.
#
# ## Setup
using HyperFEM
using Gridap
using HyperFEMKernels                                          # compiled backend (`backend=:compiled`)

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E03_Fibre_reinforced_cylinder")
setupfolder(outdir)                                            # VTK output (not versioned)

# ## Geometry and mesh
# The material and the loads are symmetric with respect to the planes ``x = 0`` and ``y = 0`` (the two
# helical families are mirror images of each other), so a quarter of the tube is enough. It is the
# box ``(R_i,R_o)\times(0,\pi/2)\times(0,L)`` in coordinates ``(r,\theta,z)``, mapped to Cartesian
# coordinates by `map`. The map moves the vertices: the cells stay trilinear, so the cross-section is a
# polygon, and the quadratic elements below do not curve it.
Ri, Ro, L = 1.0, 1.1, 4.4                                       # cm
polar(X) = Point(X[1] * cos(X[2]), X[1] * sin(X[2]), X[3])
partition = FAST ? (1, 8, 20) : (2, 12, 30)                    # cells along r, θ, z
model = CartesianDiscreteModel((Ri, Ro, 0, π / 2, 0, L), partition; map=polar)

# `CartesianTags` names the faces, edges and corners of the box (``X = r``, ``Y = \theta``, ``Z = z``).
# Overlapping tags would leave unclear which mask applies at the shared nodes, so the tags are chosen
# disjoint: the clamped ends take their edges and corners (`⁺`), the symmetry planes only their
# interior and their edges along ``z``.
T = CartesianTags
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "inner", T.face0YZ)                  # r = Ri
add_tag_from_tags!(labels, "clamped", [T.faceXY0⁺; T.faceXY1⁺]) # z = 0 and z = L
add_tag_from_tags!(labels, "sym_y", [T.faceX0Z; T.edge00Z; T.edge10Z])   # θ = 0, plane y = 0
add_tag_from_tags!(labels, "sym_x", [T.faceX1Z; T.edge01Z; T.edge11Z])   # θ = π/2, plane x = 0

# ## Constitutive model
# Parameters of the original HyperFEM example (artery-like: ``\mu_1+\mu_2`` = 74.4 kPa, nearly
# incompressible). The first two fibre families (axial, circumferential) are very weak; the helices carry
# the load. Summing an isotropic and an anisotropic model gives a model whose functions take ``\boldsymbol F``
# followed by the four fibre directions.
μ1, μ2 = 37.2e3, 37.2e3                                        # Pa
λ = 100 * (μ1 + μ2)
c1 = [4.68e-3, 2.9e-2, 28.16e3, 28.16e3]                       # Pa (axial, circumferential, helix +, helix −)
c2 = [2.97e-7, 1.68e-6, 3.48, 3.48]                            # dimensionless
matrix = MooneyRivlin3D(λ=λ, μ1=μ1, μ2=μ2)
physmodel = matrix + HGO_4Fibers(c1=c1, c2=c2)
kinematics = Kinematics(Mechano, Solid)
F, H, _ = get_Kinematics(kinematics)                           # F(∇u'), H(F) = cof F

# ## Finite element spaces
# Q2 elements: the material is nearly incompressible, and trilinear elements lock (they are about
# 1.5 % too stiff in the Lamé check below on a mesh like this one).
order = 2
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
Ω = Triangulation(model)
dΩ = Measure(Ω, 2 * order)
Γi = BoundaryTriangulation(model, tags="inner")
dΓi = Measure(Γi, 2 * order)
N = get_normal_vector(Γi)                                       # reference outward normal on Γi

# ## Fibre fields
# The directions are interpolated in a vector space without boundary conditions and passed to the
# model as finite element functions (the model normalises them at the quadrature points).
eθ(x) = VectorValue(-x[2], x[1], 0.0) / sqrt(x[1]^2 + x[2]^2)
ez(x) = VectorValue(0.0, 0.0, 1.0)
Vf = FESpace(model, reffe)
fibres = (interpolate_everywhere(ez, Vf),                         # M₁: axial
          interpolate_everywhere(eθ, Vf),                         # M₂: circumferential
          interpolate_everywhere(x -> (eθ(x) + ez(x)) / √2, Vf),  # M₃: helix +45°
          interpolate_everywhere(x -> (-eθ(x) + ez(x)) / √2, Vf)) # M₄: helix −45°

# ## Weak form
# The weak form of the model comes from `residual`/`jacobian`, with the fibres after the load factor;
# the pressure terms are written as in the equations above. `⋅` between a tensor and a vector is the
# matrix–vector product.
p = 5e3                                                         # Pa (≈ 37.5 mmHg)
p_final = FAST ? 0.2 * p : p
pressure_res(u, v, pΛ) = pΛ * ∫(((H ∘ (F ∘ ∇(u)')) ⋅ N) ⋅ v)dΓi
pressure_jac(u, du, v, pΛ) = pΛ * ∫((((F ∘ ∇(u)') × ∇(du)') ⋅ N) ⋅ v)dΓi
res(Λ) = (u, v) -> residual(physmodel, kinematics, u, v, dΩ, Λ, fibres...) + pressure_res(u, v, Λ * p_final)
jac(Λ) = (u, du, v) -> jacobian(physmodel, kinematics, u, du, v, dΩ, Λ, fibres...) + pressure_jac(u, du, v, Λ * p_final)

# ## Points where the solution is read
# Three points of the inner surface at mid-length, at ``\theta = 0, \pi/4, \pi/2``. A finite element
# function can be evaluated at a vector of points (one search for all of them).
θs = [0.0, π / 4, π / 2]
probe = [polar(Point(Ri, θ, L / 2)) for θ in θs]
er(θ) = VectorValue(cos(θ), sin(θ), 0.0)
radial(uh) = [u ⋅ er(θ) for (u, θ) in zip(uh(probe), θs)]

# ## A check first: the isotropic tube at a small pressure
# With the matrix alone, a small pressure and ends that can slide radially but not axially, the
# solution is Lamé's plane-strain solution ``u_r = A r + B/r``, where
# ```math
# B = \frac{p}{2\mu\,(R_i^{-2} - R_o^{-2})}, \qquad A = \frac{\mu B}{(\lambda_L+\mu) R_o^2},
# \qquad \mu = \mu_1+\mu_2, \qquad \lambda_L = \lambda + 2\mu_2 ,
# ```
# with the Lamé constants of the Mooney–Rivlin energy linearised at ``\boldsymbol F = \boldsymbol I``. It checks the
# mesh, the symmetry conditions and the pressure term in a few seconds. The sliding ends need ``u_z = 0``
# and, on the edges they share with the symmetry planes, the symmetry component as well: three more tags.
add_tag_from_tags!(labels, "ends", [T.faceXY0; T.faceXY1; T.edge0Y0; T.edge1Y0; T.edge0Y1; T.edge1Y1])
add_tag_from_tags!(labels, "ends_y", [T.edgeX00⁺; T.edgeX01⁺])
add_tag_from_tags!(labels, "ends_x", [T.edgeX10⁺; T.edgeX11⁺])
zero3 = [0.0, 0.0, 0.0]
D_lame = BoundaryConditions(
  BoundaryCondition("ends", zero3; mask=(false, false, true)),
  BoundaryCondition("ends_y", zero3; mask=(false, true, true)),
  BoundaryCondition("ends_x", zero3; mask=(true, false, true)),
  BoundaryCondition("sym_y", zero3; mask=(false, true, false)),
  BoundaryCondition("sym_x", zero3; mask=(true, false, false)))
V_lame = TestFESpace(Ω, reffe, D_lame, conformity=:H1)
U_lame = TrialFESpace(V_lame, D_lame)
p_small = 50.0                                                  # Pa: strains of 0.2 %
res_lame(Λ) = (u, v) -> residual(matrix, kinematics, u, v, dΩ) + pressure_res(u, v, Λ * p_small)
jac_lame(Λ) = (u, du, v) -> jacobian(matrix, kinematics, u, du, v, dΩ) + pressure_jac(u, du, v, Λ * p_small)
lame_model = StaticNonlinearModel(res_lame, jac_lame, U_lame, V_lame, D_lame;
  nls=NewtonRaphson(LUSolver(); maxiter=10, rtol=1e-10, verbose=false), backend=:compiled)
_, flag_lame = solve!(lame_model; stepping=LoadStepping(nsteps=1))
μL, λL = μ1 + μ2, λ + 2μ2
B = p_small / (2μL * (1 / Ri^2 - 1 / Ro^2))
A = μL * B / ((λL + μL) * Ro^2)
ur_lame = radial(get_state(lame_model))
println("Lamé check: u_r(Ri) = ", ur_lame[2], " cm, exact ", A * Ri + B / Ri, " cm")

# ## Boundary conditions of the clamped tube
D_bc = BoundaryConditions(
  BoundaryCondition("clamped", zero3),
  BoundaryCondition("sym_y", zero3; mask=(false, true, false)),
  BoundaryCondition("sym_x", zero3; mask=(true, false, false)))
V = TestFESpace(Ω, reffe, D_bc, conformity=:H1)
U = TrialFESpace(V, D_bc)
println("free dofs: ", num_free_dofs(V))

# ## Solver and computational model
# `backend=:compiled` (tutorial 01) compiles the residual and the tangent, the pressure terms included, once; at
# every increment the model only rebinds them to the new load factor, without compiling again.
nls = NewtonRaphson(LUSolver(); maxiter=20, rtol=1e-10, verbose=true)
comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls, backend=:compiled)
uh = get_state(comp_model)

# ## Post-processing
# After every converged increment: the pressure and the radial displacement at the three probes, and
# the displacement and fibre fields in a VTK file.
pressures, ur_history = Float64[], Vector{Float64}[]
function record(post)
  Λ = post.Λ[post.iter]
  push!(pressures, Λ * p_final)
  push!(ur_history, radial(uh))
  pvd = get_pvd(post)
  pvd[Λ] = createvtk(Ω, joinpath(outdir, "step_$(post.iter).vtu"),
    cellfields=["u" => uh, "M_axial" => fibres[1], "M_circ" => fibres[2], "M_helix+" => fibres[3], "M_helix-" => fibres[4]])
end
post = PostProcessor(comp_model, record; is_vtk=true, filepath=outdir)

# ## Solve
# The pressure is applied in equal increments; an increment whose Newton iteration fails is halved
# (up to five times in a row). The load does not enter through Dirichlet values, so `ProjectDirichlet`
# is not needed here.
nsteps = FAST ? 4 : 20
x, flag = solve!(comp_model; stepping=LoadStepping(nsteps=nsteps, maxbisec=5), post=post)   # also closes the .pvd
for (pk, ur) in zip(pressures, ur_history)
  println("p = $(round(pk, digits=1)) Pa   u_r(Ri, L/2) = $(round(ur[2], sigdigits=10)) cm")
end

# ## Checks
# (a) The isotropic tube matches Lamé's solution within 0.5 % (measured: 0.13 % FAST, 0.19 % full; the rest is
# discretisation and the small geometric nonlinearity; trilinear elements, about 1.5 % too stiff, would fail). (b) The discrete problem is invariant under rotations about the axis, so the radial
# displacement is the same at the three probes and there is no twist. (c) A regression value of the
# final radial displacement (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0, macOS arm64).
@assert flag_lame == 1 && flag == 1 "the load stepping did not reach Λ = 1"
@assert abs(ur_lame[2] / (A * Ri + B / Ri) - 1) < 0.005 "isotropic tube away from Lamé's solution"
ur_end = ur_history[end]
@assert maximum(abs.(ur_end .- ur_end[2])) < 1e-8 * abs(ur_end[2]) "radial displacement not axisymmetric"
u45 = uh(probe[2])
@assert abs(u45 ⋅ VectorValue(-sin(π / 4), cos(π / 4), 0.0)) < 1e-8 * abs(ur_end[2]) "the tube twists"
ur_ref = FAST ? 0.0393139021903048 : 0.260654941449461                        # cm, regression values
@assert isapprox(ur_end[2], ur_ref; rtol=1e-5) "radial displacement $(ur_end[2]) ≠ regression value $ur_ref"

# ## Going further
# - Change the helix angle (here ±45°) or remove the helices: the tube becomes softer and, with
#   unbalanced families, twists (the quarter model then no longer applies: mesh the whole tube with
#   `(0, 2π)` in the map and remove the symmetry tags).
# - Increase the pressure: the exponential fibres stiffen and the pressure–radius curve turns up.
# - Tutorial 05 uses a single fibre family in an electro-active solid; tutorial 13 replaces `LUSolver`
#   for larger meshes.
