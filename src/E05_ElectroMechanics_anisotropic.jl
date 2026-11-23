# # Tutorial 05: Anisotropic electro-mechanics
#
# The dielectric elastomer plate of tutorial 04, now reinforced by a family of stiff fibres lying in the plane
# of the plate at 45° between its width and its length. The fibres resist the in-plane elongation of the active
# layer along their direction, so the plate no longer deforms symmetrically: besides bending, it shears and
# twists. The constitutive model is built by adding three models, and the fibre direction enters the weak
# form as an extra field.
#
# **What you will learn**
# - composing constitutive models with `+` (isotropic matrix, fibres, dielectric);
# - passing a field of material directions to an anisotropic model (the extra arguments `vars...`);
# - the field-by-field helpers `residual(model, Mechano, ...)`, `residual(model, Electro, ...)` and the coupling
#   tangent `jacobian(model, ElectroMechano, ...)`, which accept those extra arguments;
# - detecting the effect of anisotropy through a broken symmetry.
#
# Measured on a Mac M5 Max (Julia 1.13.0, 4 threads, 4 BLAS threads; HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8; compiled backend), whole run including compilation:
# FAST 45 s, full 46 s (20 increments, 109 Newton iterations, 16 s in `solve!` including its compilation);
# 1 528 free dofs. One assembly of the tangent: 0.7 ms compiled, 13 ms with Gridap's lazy evaluation.
#
# ## Problem statement
# Geometry, boundary conditions and weak form are those of tutorial 04. The free energy adds to the
# neo-Hookean and ideal-dielectric energies the Holzapfel–Gasser–Ogden energy of one family of fibres with unit
# direction ``\boldsymbol N`` in the reference configuration,
# ```math
# \Psi = \Psi_{\mathrm{NH}}(\boldsymbol F) + \frac{c_1}{4 c_2}\left(e^{c_2 (I_4 - 1)^2} - 1\right)
#      - \frac{\varepsilon}{2J}\,|\boldsymbol H \boldsymbol E_0|^2 ,
# \qquad I_4 = \boldsymbol F\boldsymbol N \cdot \boldsymbol F\boldsymbol N ,
# ```
# with ``c_1 = 10``, ``c_2 = 20`` (in units of ``\mu = 1``) and ``\boldsymbol N = (1, 0, 1)/\sqrt 2``: the fibres lie
# in the plane of the plate, at 45° between the width ``x`` and the length ``z``. ``I_4`` is the squared stretch
# of the fibres. (This form of the energy acts in compression too: fibres that can only be pulled would need a
# switch on ``I_4 > 1``.)
#
# Reflecting the plate about its mid-plane ``x = 4`` maps ``\boldsymbol N`` to ``(-1, 0, 1)/\sqrt 2``, a different
# material: the symmetry of tutorial 04 is lost.
#
# ## Setup
using HyperFEM
using Gridap
using HyperFEMKernels                                          # compiled backend (`backend=:compiled`)
using LinearAlgebra: norm

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E05_ElectroMechanics_anisotropic")
setupfolder(outdir)                                            # VTK output (not versioned)

# ## Geometry and mesh
model = GmshDiscreteModel(joinpath(@__DIR__, "..", "data", "models", "test_static_EM.msh"))
labels = get_face_labeling(model)
add_tag_from_vertex_filter!(labels, "tip", model, x -> abs(x[3] - 40.0) < 1e-8)

# ## Constitutive model
# `+` composes models: an isotropic and an anisotropic mechanical model give an anisotropic mechanical model,
# and a mechanical and an electrical model give an `ElectroMechModel`. The derivatives of the composition are
# the sums of the derivatives; they now take the fibre direction as a third argument, ``(\boldsymbol F,
# \boldsymbol E_0, \boldsymbol N)``.
physmodel = NeoHookean3D(λ=10.0, μ=1.0) + HGO_1Fiber(c1=10.0, c2=20.0) + IdealDielectric(ε=1.0)
k = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))

# ## Boundary conditions
# The full case stops at 0.25 instead of the 0.3 of tutorial 04: with the fibres, Newton's iterations diverge
# at about 0.27 (the iterations of the previous increments already converge erratically) and, in HyperFEM 0.3.0,
# the run ends with an uncaught `SingularException` of the LU factorisation instead of a bisection.
V0 = FAST ? 0.1 : 0.25                                         # voltage between the electrodes
nsteps = FAST ? 5 : 20
Du = BoundaryConditions(BoundaryCondition("fixedup", [0.0, 0.0, 0.0]))
Dφ = BoundaryConditions(BoundaryCondition("midsuf", 0.0, Λ -> Λ), BoundaryCondition("topsuf", V0, Λ -> Λ))
D_bc = MultiFieldBC([Du, Dφ])

# ## Finite element spaces
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

# ## Fibre field
# The direction is a finite element field, so it could vary in space (read from a file, or computed, as the
# fibres of tutorial 03). Here it is uniform. We give it unit length; `HGO_1Fiber` also normalises it at every
# quadrature point, so only its direction matters.
VN = FESpace(model, reffeu)                                    # no boundary conditions
Nh = interpolate_everywhere(VectorValue(1.0, 0.0, 1.0) / sqrt(2.0), VN)

# ## Weak form
# The forms of tutorial 04 with `Nh` appended to the arguments of every constitutive function. The helpers do it
# for us: every one of them takes the extra arguments of the model after the load factor ``\Lambda``. The
# residual is the sum of the mechanical and electrical equations, and the tangent the sum of the two diagonal
# blocks and the two coupling blocks (`ElectroMechano`).
res(Λ) = ((u, φ), (v, vφ)) ->
  residual(physmodel, Mechano, k, (u, φ), v, dΩ, Λ, Nh) +
  residual(physmodel, Electro, k, (u, φ), vφ, dΩ, Λ, Nh)

jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) ->
  jacobian(physmodel, Mechano, k, (u, φ), du, v, dΩ, Λ, Nh) +
  jacobian(physmodel, Electro, k, (u, φ), dφ, vφ, dΩ, Λ, Nh) +
  jacobian(physmodel, ElectroMechano, k, (u, φ), (du, dφ), (v, vφ), dΩ, Λ, Nh)

# ## Solver and computational model
# The compiled backend (tutorial 01) also accepts the fibre field: `Nh` is read at the quadrature points like the state.
nls = NewtonRaphson(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-8, verbose=true)
comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls, backend=:compiled)
xh = get_state(comp_model)
uh, φh = xh

# ## Post-processing
# The mean displacement of the free end at every increment, and the VTK files (with the fibre field).
Γtip = BoundaryTriangulation(model, tags="tip")
dΓtip = Measure(Γtip, 2 * order)
Atip = sum(∫(1.0)dΓtip)
voltage, tip_u = Float64[], VectorValue{3,Float64}[]
function record(post)
  Λ = post.Λ[post.iter]
  push!(voltage, Λ * V0)
  push!(tip_u, sum(∫(uh)dΓtip) / Atip)
  pvd = get_pvd(post)
  pvd[Λ] = createvtk(Ω, joinpath(outdir, "step_$(post.iter).vtu"), cellfields=["u" => uh, "φ" => φh, "N" => Nh])
end
post = PostProcessor(comp_model, record; is_vtk=true, filepath=outdir)

# ## Solve
x, flag = solve!(comp_model; stepping=LoadStepping(nsteps=nsteps, maxbisec=5), ProjectDirichlet=true, post=post)

for (vo, w) in zip(voltage, tip_u)
  println("V = $(round(vo, digits=4))   mean tip displacement (u_x, u_y, u_z) = $(round.(Tuple(w), sigdigits=6))")
end

# ## Checks
# **Broken symmetry.** In tutorial 04 the problem is symmetric about the plane ``x = 4``: the displacement at a
# point and at its mirror image ``(8 - x, y, z)`` are mirror images of each other, ``u_x`` with the opposite sign
# and ``u_y``, ``u_z`` equal. We measure the largest departure from that, at a set of points of the plate,
# relative to the deflection of the free end. It is zero to round-off in tutorial 04; the fibres make it a visible
# fraction of the deflection. (The mean of ``u_x`` would also do, but it changes sign along the load path.)
mirror(p) = Point(8.0 - p[1], p[2], p[3])
pts = [Point(x, y, z) for x in (0.5, 1.5, 2.5, 3.5) for y in (0.2, 0.6) for z in (10.0, 20.0, 30.0, 39.5)]
asym = maximum(norm(VectorValue(a[1] + b[1], a[2] - b[2], a[3] - b[3])) for (a, b) in zip(uh(pts), uh(mirror.(pts))))
tip_uy = tip_u[end][2]
println("asymmetry max|u(p) - mirrored u(p')| = $(asym)   tip u = $(tip_u[end])")
@assert flag == 1 "the load stepping did not reach Λ = 1"
@assert tip_uy < 0 "the plate does not bend towards -y"
@assert asym > 1e-2 * abs(tip_uy) "the fibres do not break the symmetry about x = 4"

# **Regression** (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0, macOS arm64).
nu = norm(get_free_dof_values(uh))
println("‖u‖ = $(nu)")
nu_ref, tip_ref = FAST ? (41.0162661692127, [0.03549296225574539, -3.9131266697605196, -0.08863170134311517]) :
                         (273.70414034301757, [11.101859070803826, -20.15049709580238, -9.803265915334809])
@assert isapprox(nu, nu_ref; rtol=1e-5) "‖u‖ = $nu differs from the reference $nu_ref"
@assert isapprox(collect(Tuple(tip_u[end])), tip_ref; rtol=1e-5) "tip displacement $(tip_u[end]) differs from the reference $tip_ref"

# ## Going further
# - Turn the fibres: along the length, `VectorValue(0.0, 0.0, 1.0)`, or across the width, `VectorValue(1.0, 0.0, 0.0)`,
#   they keep the symmetry about ``x = 4`` (the mean of ``u_x`` vanishes again); compare the deflections. A fibre
#   field that varies in space only needs a function in `interpolate_everywhere`.
# - Add a second family with another `HGO_1Fiber` and another field (the arguments then follow the order of the
#   models in the sum), or use `HGO_4Fibers` as in tutorial 03.
# - Compare with tutorial 04: the fibres reduce the deflection at the same voltage.
