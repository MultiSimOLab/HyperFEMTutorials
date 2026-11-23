# # Tutorial 06: Third-medium contact
#
# A slender column, clamped at both ends, is compressed by a rigid piston inside a box. It buckles, bends
# sideways until it touches the walls of the box, and keeps folding against them. The contact is not
# detected by any search algorithm: the empty space around the column is meshed and filled with a
# *third medium*, a very soft hyperelastic material that transmits almost no force while it is open and
# stiffens strongly when it is squeezed to zero thickness. This is the piston benchmark of Faltus, Amato and
# Horák (2026, Sec. 3.3), a hard test for the nonlinear solver because of the buckling and the snaps.
#
# **What you will learn**
# - meshes with several subdomains (`Triangulation(model, tags=...)`) and a different material in each;
# - the third-medium contact method: `ARAP2D + LogBarrier` and the skeleton regularisation of the gradient jump
#   (`SkeletonTriangulation`, `jump`, `mean`);
# - a robust Newton for problems that lose stability: the injectivity-preserving line search `InjectivityLS`
#   with Armijo on the energy (`energy=`, `injectivity=`), and the hybrid Newton (`HybridTangent`,
#   `HessianRegularization` sharing a `TangentSwitch`);
# - increments that grow when Newton converges easily (`LoadStepping(grow=...)`);
# - reaction forces computed from the residual, and a force–displacement history written to a CSV file.
#
# Measured on a Mac M5 Max (Julia 1.13.0, 4 threads, 4 BLAS threads; HyperFEM 0.3.0, HyperFEMKernels 0.2.6,
# Gridap 0.20.8; compiled backend), whole run including compilation:
# FAST 51 s (mesh `Piston_1.msh`, 1 340 free dofs, 15 increments, 22 s of them in `solve!`, mostly compilation);
# full 107 s (mesh `Piston_0.5.msh`, 5 494 free dofs, 400 increments, 900 Newton iterations, 78 s in `solve!`:
# 0.2 s per increment: a third in Newton, the rest in the Dirichlet projection, the updates of the compiled
# forms and the post-processing).
# One assembly of the tangent on the full mesh: 5.8 ms compiled, 31 ms with Gridap's lazy evaluation.
#
# ## Problem statement
# The box is ``(0.1, 0.9)\times(0.1, 1.9)`` (plane strain, dimensionless lengths). The column is the strip
# ``0.45 \le x \le 0.55`` of thickness ``d = 0.1`` and length ``L = 1.8``; the two regions on its sides are the
# third medium. The bottom of the column is clamped. The top of the box is the piston: it moves down by
# ``u_y = \Lambda\,\bar u``, with the top of the column clamped to it and the third medium free to slide along it.
# The third medium slides on the bottom and on the side walls of the box.
#
# Find ``\boldsymbol u`` such that, for all ``\boldsymbol v``,
# ```math
# \int_{\Omega_s} \partial_{\boldsymbol F}\Psi_s : \nabla_0 \boldsymbol v \,\mathrm dA
# + \int_{\Omega_{tm}} \partial_{\boldsymbol F}\Psi_{tm} : \nabla_0 \boldsymbol v \,\mathrm dA
# + \int_{\Gamma_{tm}} \frac{\sigma\beta}{h_F} [\![\nabla_0 \boldsymbol u]\!] : [\![\nabla_0 \boldsymbol v]\!] \,\mathrm ds = 0 ,
# ```
# which is the stationarity condition of the total energy
# ```math
# \Pi(\boldsymbol u) = \int_{\Omega_s} \Psi_s \,\mathrm dA + \int_{\Omega_{tm}} \Psi_{tm} \,\mathrm dA
# + \frac12 \int_{\Gamma_{tm}} \frac{\sigma\beta}{h_F} [\![\nabla_0 \boldsymbol u]\!] : [\![\nabla_0 \boldsymbol u]\!] \,\mathrm ds .
# ```
#
# **Solid.** A compressible Mooney–Rivlin material written for plane strain (``\mu_1 = \mu_2 = \mu``):
# ```math
# \Psi_s = \frac{\mu_1}{2}\left(\boldsymbol F:\boldsymbol F + 1 - 3\right) + \frac{\mu_2}{2}\left(\boldsymbol F:\boldsymbol F + J^2 - 3\right)
# - (\mu_1 + 2\mu_2)\ln J + \frac{\lambda}{2}(J - 1)^2 ,
# ```
# with ``\mu = 10^5`` and ``\lambda = 6\mu``. Linearised, its shear modulus is ``\mu_1 + \mu_2 = 2\mu`` and its first
# Lamé constant ``\lambda + 2\mu`` (Poisson's ratio 0.4 in plane strain, as in the paper).
#
# **Third medium** (Wriggers, Schröder and Schwarz 2013). The MIPS energy (called `ARAP2D` in HyperFEM), which
# penalises the distortion of the elements but not their size, plus a barrier that grows without bound when the
# area of an element goes to zero:
# ```math
# \Psi_{tm} = \frac{\mu_{tm}}{2}\left(\frac{\boldsymbol F:\boldsymbol F}{J} - 2\right) + \frac{\kappa}{2p}(\ln J)^{2p} ,
# \qquad \mu_{tm} = f_c\,\mu ,\quad \kappa = \mu_{tm} ,\quad p = 2 ,\quad f_c = 10^{-6} .
# ```
# While the gap is open the third medium is a million times softer than the column; when the column presses it
# against a wall, ``J \to 0`` and the barrier transmits the contact pressure.
#
# **Regularisation** (Bluhm, Sigmund and Poulios 2021). A soft medium that is squeezed develops spurious
# zig-zag modes. Bluhm et al. add the energy ``\tfrac{\beta}{2}\int_{\Omega_{tm}} \nabla\nabla\boldsymbol u \,\vdots\, \nabla\nabla\boldsymbol u``,
# which penalises the curvature of the deformation of the third medium. With linear triangles ``\nabla\nabla\boldsymbol u = 0``
# inside every element and the curvature sits on the facets, as the jump of the gradient: the skeleton term above
# (interior facets ``\Gamma_{tm}`` of the third medium, facet length ``h_F``, ``\beta = 10^{-5}\mu``, ``\sigma = 1``) is its
# discrete counterpart, as in interior-penalty methods.
#
# **Reactions.** The piston force is reported as ``f/f_{cr}``, with the Euler load of the double-clamped column in plane
# strain, ``f_{cr} = \pi^2 E' I / (L/2)^2``, ``I = d^3/12``, ``E' = E/(1 - \nu^2)`` of the linearised solid
# (the normalisation of Horák's curves).
#
# ## Setup
using HyperFEM
using Gridap
using HyperFEMKernels                                          # compiled backend (`backend=:compiled`)
using Gridap.CellData: get_array                               # cell-wise values of an integral
using GridapGmsh
using LinearAlgebra
using Printf

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # reduced case for the test suite
outdir = joinpath(@__DIR__, "..", "data", "sims", "E06_ThirdMedium_contact")
setupfolder(outdir)                                            # VTK and CSV output (not versioned)

# ## Geometry and mesh
# Gmsh meshes of linear triangles with two (`Piston_1.msh`, FAST) or four (`Piston_0.5.msh`, full) elements across the
# column. Their physical groups name the two subdomains, `domain` (the column) and `third` (the third medium), and the
# boundary pieces used below.
meshfile = FAST ? "Piston_1.msh" : "Piston_0.5.msh"
geomodel = GmshDiscreteModel(joinpath(@__DIR__, "..", "data", "models", meshfile))
d, Lb = 0.1, 1.8                                               # thickness and length of the column

# ## Constitutive models
# The hybrid Newton (see *Solver* below) needs two tangents of the same energy: the consistent one, and a
# *projected* one, positive definite at every quadrature point (the eigenvalues of ``\partial^2_{\boldsymbol F\boldsymbol F}\Psi``
# clamped from below at ``\delta``). `HessianRegularization` wraps a model and returns the projected tangent while the
# shared `TangentSwitch` is on; the energy and the stress are not changed.
μ = 1e5                                                        # Pa
fc = 1e-6                                                      # stiffness of the third medium relative to μ
hrs = 1e-6                                                     # projection threshold relative to the stiffness
switch = TangentSwitch()
solid = HessianRegularization(NonlinearMooneyRivlin2D(λ=6μ, μ1=μ, μ2=μ, α1=1.0, α2=1.0); δ=hrs * μ, switch=switch)
medium = HessianRegularization(ARAP2D(μ=fc * μ) + LogBarrier(κ=fc * μ, p=2); δ=hrs * fc * μ, switch=switch)
Ψs, ∂Ψs∂F, ∂Ψs∂FF = solid()
Ψm, ∂Ψm∂F, ∂Ψm∂FF = medium()
F, _, _ = get_Kinematics(Kinematics(Mechano, Solid))
Fu(u) = F ∘ (∇(u)')                                            # deformation gradient of a displacement field

# ## Boundary conditions
# The tags come from the mesh. `Imposed_Y_fixed_X` is the top of the column, glued to the piston;
# `Imposed_Y_free_X` is the top of the third medium, which follows the piston but slides along it.
ū = FAST ? 0.15 : 0.40                                         # final piston displacement: 1.5 d (FAST) or 4 d
D_bc = BoundaryConditions(
  BoundaryCondition("Fixed_XY", [0.0, 0.0]),
  BoundaryCondition("Fixed_Y", [0.0, 0.0]; mask=(false, true)),
  BoundaryCondition("Fixed_X", [0.0, 0.0]; mask=(true, false)),
  BoundaryCondition("Imposed_Y_free_X", [0.0, -ū], EvolutionFunctions.ramp(); mask=(false, true)),
  BoundaryCondition("Imposed_Y_fixed_X", [0.0, -ū], EvolutionFunctions.ramp()))

# ## Finite element spaces
# One displacement field over the whole box (column and third medium together), and one measure per subdomain.
order = 1
reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
Ω = Triangulation(geomodel, tags=["domain", "third"])
Ωs = Triangulation(geomodel, tags=["domain"])
Ωm = Triangulation(geomodel, tags=["third"])
dΩ, dΩs, dΩm = Measure(Ω, 2order), Measure(Ωs, 2order), Measure(Ωm, 2order)
V = TestFESpace(Ω, reffe, D_bc, conformity=:H1)
U = TrialFESpace(V, D_bc, 0.0)
uh = FEFunction(U, zeros(num_free_dofs(U)))
println("free dofs: ", num_free_dofs(V))

# ## Weak form
# The skeleton of the third medium: its interior facets, where `jump(∇(u))` is the jump of the gradient between the
# two triangles of a facet. ``h_F`` is computed as the integral of 1 over each facet.
Γ = SkeletonTriangulation(geomodel, tags=["third"])
dΓ = Measure(Γ, 2order)
hF = CellField(get_array(∫(1)dΓ), Γ)
β, σ = 1e-5 * μ, 1.0
cJ = (σ * β) / hF
qJ(a, b) = jump(∇(a)) ⊙ jump(∇(b))

res(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψs∂F ∘ Fu(u)))dΩs +
                   ∫(∇(v)' ⊙ (∂Ψm∂F ∘ Fu(u)))dΩm +
                   ∫(cJ * qJ(u, v))dΓ
jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ Fu(u)) ⋅² ∇(du)'))dΩs +
                       ∫(∇(v)' ⊙ ((∂Ψm∂FF ∘ Fu(u)) ⋅² ∇(du)'))dΩm +
                       ∫(cJ * qJ(du, v))dΓ
# The energy ``\Pi``, used by the line search below. Its first variation is the residual above.
en(Λ) = u -> ∫(Ψs ∘ Fu(u))dΩs + ∫(Ψm ∘ Fu(u))dΩm + ∫(0.5 * cJ * qJ(u, u))dΓ

# ## Solver and computational model
# Three ingredients make Newton robust here.
#
# 1. **Injectivity.** The barrier is infinite at ``J = 0`` and the energies are not defined for ``J \le 0``: a full Newton
#    step that inverts one triangle of the third medium is useless. `InjectivityLS` computes the largest step ``\alpha``
#    that keeps ``\det(\boldsymbol F + \alpha\nabla_0\delta\boldsymbol u) \ge J_{min}`` at every quadrature point of `injectivity`
#    (a quadratic equation per point in 2D, solved exactly) and starts the line search from ``0.999999\,\alpha``.
# 2. **Armijo on the energy.** From there the step is halved until the energy decreases enough,
#    ``\Pi(\boldsymbol u + \alpha\delta\boldsymbol u) \le \Pi(\boldsymbol u) + c_1\alpha\,\boldsymbol R\cdot\delta\boldsymbol u``
#    (`armijo=true`, the energy given with `energy=`).
# 3. **Hybrid Newton.** After the buckling the consistent tangent is indefinite and its Newton direction may not be a
#    descent direction, so the line search fails. `HybridTangent` then switches the `TangentSwitch` on and continues with
#    the projected tangent, which always gives a descent direction; it goes back to the consistent tangent (quadratic
#    convergence) once ``\|\boldsymbol R\|`` has dropped by `hybback`. In projected mode the convergence test is absolute,
#    ``\|\boldsymbol R\| < 10^{-3} f_{cr}``.
#
# With the increments used below the consistent tangent turns out to be enough: the solver log reports 0 % of the
# iterations in projected mode, in FAST and in the full run. The hybrid tangent is the safety net for larger
# increments; with the growing increments of *Going further* it takes over in about a fifth of the iterations.
#
# With `backend=:compiled` (tutorial 01) the residual, the tangent, the energy of the line search and the injectivity
# step are all evaluated by compiled kernels, which matters most for the line search: it evaluates the energy
# several times per Newton iteration.
μl, λl = 2μ, 6μ + 2μ                                           # linearised solid
Eps = 4μl * (λl + μl) / (λl + 2μl)                             # plane-strain modulus E/(1 - ν²)
fcr = π^2 * Eps * (d^3 / 12) / (0.5Lb)^2                       # Euler load of the double-clamped column

linesearch = InjectivityLS(jmin=1e-6, armijo=true, β=1 - 1e-6, ρ=0.5, maxiter=20, αmin=1e-16)
tangent = HybridTangent(hybback=1e-2, atol_projected=1e-3 * fcr)
# Newton stops at ``\|\boldsymbol R\| < 10^{-6}`` (about ``10^{-9} f_{cr}``): every increment ends at an equilibrium,
# so the results (and the regression values below) do not depend on where the iterations happen to stop.
nls = NewtonRaphson(LUSolver(); maxiter=500, atol=1e-6, rtol=1e-10, verbose=1, linesearch=linesearch, tangent=tangent)
comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls, xh=uh, injectivity=dΩ, energy=en, tangent_switch=switch,
                                backend=:compiled)

# ## Post-processing
# **Reaction forces from the residual.** At a converged state the residual vanishes on the free dofs, and on the
# Dirichlet dofs it is the reaction. Testing the residual with a function ``\boldsymbol\varphi`` that is ``\boldsymbol e_y`` on the
# nodes of a boundary and zero elsewhere gives the vertical force on that boundary, ``f = \boldsymbol R(\boldsymbol u)\cdot\boldsymbol\varphi``.
# Such a ``\boldsymbol\varphi`` is a finite element function whose free values are zero and whose Dirichlet values on the
# chosen tags are ``(0, 1)``.
function unit_y_on(tags)
  Vr = TestFESpace(Ω, reffe, conformity=:H1, dirichlet_tags=tags, dirichlet_masks=[(false, true) for _ in tags])
  FEFunction(TrialFESpace(Vr, [VectorValue(0.0, 1.0) for _ in tags]), zeros(num_free_dofs(Vr)))
end
φpiston = unit_y_on(["Imposed_Y_fixed_X", "Imposed_Y_free_X"])
φclamp = unit_y_on(["Fixed_XY"])

# After every converged increment we store the piston displacement and both reactions (the piston pushes down,
# so its force is ``-\boldsymbol R\cdot\boldsymbol\varphi``) and write the displacement and ``J`` every few increments.
uy_d, fpiston, fclamp = Float64[], Float64[], Float64[]
vtkevery = FAST ? 5 : 25
function record(post)
  Λ = post.Λ[post.iter]
  push!(uy_d, Λ * ū / d)
  push!(fpiston, -sum(res(Λ)(uh, φpiston)) / fcr)
  push!(fclamp, sum(res(Λ)(uh, φclamp)) / fcr)
  if post.iter % vtkevery == 0
    pvd = get_pvd(post)
    pvd[Λ] = createvtk(Ω, joinpath(outdir, "step_$(post.iter).vtu"), cellfields=["u" => uh, "J" => det ∘ Fu(uh)])
  end
end
post = PostProcessor(comp_model, record; is_vtk=true, filepath=outdir)

# ## Solve
# FAST: 15 equal increments up to ``1.5\,d``, just past the buckling of the column. Full: 400 increments up to
# ``4\,d``, through the first contact and the first snap; a bisected increment is allowed to grow back by 1.5 after
# an increment that converged in at most 10 Newton iterations. `ProjectDirichlet=true` predicts every increment from
# the linearised effect of the new piston position (tutorial 02), scaled down if needed so that no element is inverted.
stepping = FAST ? LoadStepping(nsteps=15, maxbisec=5) : LoadStepping(nsteps=400, maxbisec=5, grow=1.5)
nbisect = Ref(0)
t_solve = @elapsed x, flag = solve!(comp_model; stepping=stepping, ProjectDirichlet=true, post=post,
                                    on_bisection=(Λ, ΔΛ) -> nbisect[] += 1)   # also closes the .pvd
@printf("solve: %.1f s, %d increments, %d bisections\n", t_solve, length(uy_d), nbisect[])

open(joinpath(outdir, "force_displacement.csv"), "w") do io
  println(io, "uy_over_d,f_piston_over_fcr,f_clamp_over_fcr")
  for (a, b, c) in zip(uy_d, fpiston, fclamp)
    println(io, a, ",", b, ",", c)
  end
end

# ## Checks
# **Equilibrium.** The third medium is so soft that the piston force must reach the bottom clamp through the column:
# both reactions agree within 1 % at every increment.
@assert flag == 1 "the load stepping did not reach Λ = 1"
@assert all(abs.(fpiston .- fclamp) .< 1e-2 .* abs.(fpiston)) "piston and clamp reactions are not in equilibrium"

# **Buckling.** Without buckling the force would grow linearly with the shortening of the column (by a factor of 3 from
# ``0.5\,d`` to ``1.5\,d``); a buckled column carries an almost constant force, the plateau of the curve.
i05, i15 = argmin(abs.(uy_d .- 0.5)), argmin(abs.(uy_d .- 1.5))
@assert fpiston[i15] < 1.1 * fpiston[i05] "the column did not buckle before u_y = 0.5 d"

# **Comparison with the paper** (printed, not asserted). Horák's curve is computed with quadratic elements. Linear
# triangles are too stiff in bending when there are few of them across the column: with two (FAST mesh) the buckling
# plateau comes out about 60 % above the paper's; with four (full mesh) about 7 % above, and the snap
# (``3.70\,f_{cr}`` at ``2.96\,d``, against ``3.43\,f_{cr}`` at ``3.0\,d`` in the paper) is in the right place.
function horak_curve(file, key)
  s = read(file, String)
  block = match(Regex("\"$key\": \\[(.*?)\\]\\]"), s).captures[1]
  pts = [parse.(Float64, (m[1], m[2])) for m in eachmatch(r"\[(-?[0-9.eE+-]+), (-?[0-9.eE+-]+)\]", block * "]")]
  first.(pts), last.(pts)
end
hx, hy = horak_curve(joinpath(@__DIR__, "..", "data", "reference", "horak2026_piston_f_vs_uy.json"), "piston")
interp(xs, ys, x) = (i = clamp(searchsortedlast(xs, x), 1, length(xs) - 1); ys[i] + (ys[i+1] - ys[i]) * (x - xs[i]) / (xs[i+1] - xs[i]))
for x in (FAST ? (0.5, 1.0, 1.5) : (0.5, 1.0, 2.0, 3.0, 3.5, 4.0))
  i = argmin(abs.(uy_d .- x))
  @printf("u_y/d = %.3f   f/f_cr = %.4f   (Horák %.4f)\n", uy_d[i], fpiston[i], interp(hx, hy, uy_d[i]))
end
imax = argmax(fpiston)
@printf("maximum force f/f_cr = %.4f at u_y/d = %.3f\n", fpiston[imax], uy_d[imax])
@printf("‖u‖ = %.10f   f/f_cr at the end = %.10f\n", norm(get_free_dof_values(uh)), fpiston[end])

# **Regression** (FAST; HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0, macOS arm64), and,
# in the full run, the **first snap**: the column touches the walls, the force rises steeply and then drops when the
# column snaps into a new shape: the maximum of the force is reached before ``3.4\,d``, and the force at ``4\,d`` is
# below it.
if FAST
  @assert length(uy_d) == 15 && nbisect[] == 0 "an increment was bisected"
  @assert isapprox(norm(get_free_dof_values(uh)), 3.8344902564, rtol=1e-5) "regression: ‖u‖"
  @assert isapprox(fpiston[end], 1.7420291165, rtol=1e-5) "regression: piston force at 1.5 d"
else
  @assert uy_d[imax] < 3.4 && fpiston[end] < fpiston[imax] "no snap before u_y = 3.4 d"
end

# ## Going further
# - Run the whole benchmark, up to ``15\,d``: `ū = 1.5` and `LoadStepping(nsteps=1000, maxbisec=5, grow=1.5)` (several
#   minutes); the column folds twice more against the walls. Compare with the full curve in
#   `data/reference/horak2026_piston_f_vs_uy.json` (keys `piston` and `clamp`).
# - Let the increments grow beyond the initial one: `LoadStepping(nsteps=400, maxbisec=5, grow=1.5, ΔΛmax=0.01,
#   neasy=2)` reaches ``4\,d`` in about 100 increments. On `Piston_1.msh` it gives the same curve as 400 fixed
#   increments (maximum ``5.78\,f_{cr}`` at ``3.32\,d``), and the hybrid Newton is now at work: 21 % of the Newton
#   iterations use the projected tangent (5 switches). `mode_history(nls)` lists the mode of every iteration of the last Newton solve.
# - Remove the barrier (`ARAP2D(μ=fc * μ)` alone) or the skeleton term, and compare the force curve and the field
#   ``J`` of the third medium under the column.
# - Bluhm et al. also weight the regularisation with a function ``w(\boldsymbol F)`` that grows with compression and distortion;
#   the residual is then no longer the derivative of an energy, and the line search must use
#   `InjectivityLS(armijo=false)` (backtracking on ``|\boldsymbol R\cdot\delta\boldsymbol u|``) instead of Armijo.
# - `ConsistentTangent()` instead of `HybridTangent`: a failed line search now fails the increment, and the driver
#   has to bisect it; compare the number of bisections after the buckling.
#
# **References.**
# O. Faltus, M. Amato, M. Horák (2026), Deformation gradient averaging regularization for third medium contact,
# arXiv:2603.13063 (piston benchmark, Sec. 3.3, Fig. 3b).
# P. Wriggers, J. Schröder, A. Schwarz (2013), A finite element method for contact using a third medium,
# Comput. Mech. 52, 837–847, doi:10.1007/s00466-013-0848-5.
# G. L. Bluhm, O. Sigmund, K. Poulios (2021), Internal contact modeling for finite strain topology optimization,
# Comput. Mech. 67, 1099–1114, doi:10.1007/s00466-021-01974-x.
