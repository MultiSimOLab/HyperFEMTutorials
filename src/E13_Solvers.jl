# # Tutorial 13: Linear solvers for large problems
#
# Every Newton iteration of a nonlinear simulation solves a linear system with the tangent matrix. The sparse LU
# factorisation used in the other tutorials is robust and exact, but its cost and memory grow much faster than the
# number of unknowns: beyond some tens of thousands of degrees of freedom in 3D it is most of the run. This tutorial
# solves the same hyperelastic problem with the linear solvers that HyperFEM offers for large problems and checks
# that they all reach the same solution, then does the same for a coupled electro-mechanical problem.
#
# **What you will learn**
# - plugging a linear solver into `NewtonRaphson`: `LUSolver`, the pure-Julia `AMGSolver` (Krylov method
#   preconditioned by algebraic multigrid) and, with PETSc, `PETScSolver` (CG + GAMG, CG + hypre's BoomerAMG, MUMPS);
# - why the rigid-body modes of the displacement space (`near_nullspace`) matter to algebraic multigrid;
# - the inexact Newton method (`forcing = EisenstatWalker()`) and lagged preconditioners (`reuse`);
# - block (field-split) preconditioners for multi-field problems: `AMGSolver(U; split)`, `PETScSolver(U; reuse)`;
# - reading the solver counters: `num_iterations`, `linear_history`, `preconditioner_builds`.
#
# Measured on a Mac M5 Max (Julia 1.13.0, 4 Julia threads, 4 BLAS threads, PETSc serial; HyperFEM 0.3.0,
# HyperFEMKernels 0.2.6, Gridap 0.20.8): FAST (bar with 1 425 dofs, electro-mechanical box with 549 dofs) 56 s
# with compilation; full run (bar with 68 493 dofs, box with 23 895 dofs) 178 s. Setup + linear solve per Newton
# iteration on the 68 493-dof bar: LU 2.2 s, CG + AMG with rigid modes 1.05 s (0.37 s with `reuse = 3` and the
# inexact Newton), CG + GAMG 0.51 s, CG + BoomerAMG (nodal coarsening 4, rotations) 0.78 s, MUMPS 1.05 s; one
# tangent assembly 0.015 s compiled, 0.44 s with Gridap.
#
# ## Problem statement
# The neo-Hookean bar ``\Omega = (0,1)\times(0,0.2)\times(0,0.2)`` (``\lambda = 10``, ``\mu = 1``; the energy of
# tutorial 02) is clamped at ``x = 0``; the face ``x = 1`` is moved by ``u_z = 0.05\,\Lambda`` (bending), with
# the load factor ``\Lambda`` going from 0 to 1 in two increments. Trilinear hexahedra (Q1), ``5n \times n \times n``
# cells. The Newton iteration solves, at every iteration ``k``,
# ```math
# \boldsymbol K(\boldsymbol x_k)\, \Delta\boldsymbol x = -\boldsymbol R(\boldsymbol x_k),
# \qquad \boldsymbol x_{k+1} = \boldsymbol x_k + \Delta\boldsymbol x ,
# ```
# where ``\boldsymbol K`` is the (symmetric positive definite, along this path) tangent stiffness. A Krylov method
# (conjugate gradients, CG, for symmetric positive definite systems; GMRES otherwise) only needs products
# ``\boldsymbol K \boldsymbol v`` and a preconditioner, an approximate inverse of ``\boldsymbol K`` that is cheap to
# apply. Algebraic multigrid (AMG) builds that approximation from the matrix alone: a hierarchy of coarser
# problems that removes the smooth error components a simple smoother cannot.
#
# ## Setup
using HyperFEM
using Gridap
using LinearAlgebra, Printf
using HyperFEMKernels                          # compiled, threaded assembly (`backend = :compiled`)
import AlgebraicMultigrid                      # activates `AMGSolver` (an extension of HyperFEM)

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"    # reduced case for the test suite
const PETSC = get(ENV, "HYPERFEM_TUTORIALS_PETSC", "1") == "1"  # parts that need PETSc (GridapPETSc)
if PETSC
  using GridapPETSc                            # activates `PETScSolver` (PETSc is initialised on first use)
end
outdir = joinpath(@__DIR__, "..", "data", "sims", "E13_Solvers")
setupfolder(outdir)                                             # VTK output (not versioned)

# ## The mechanical problem
# The problem is built by a function of the mesh size `n`, so that the same code gives a small problem to compile
# every solver once (the times reported below are then those of the solve, not of Julia's compilation) and the
# problem of the tutorial. Everything it needs is created inside it.
function bar_problem(n)
  model = CartesianDiscreteModel((0, 1, 0, 0.2, 0, 0.2), (5n, n, n))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "moved", CartesianTags.face1YZ⁺)
  physmodel = NeoHookean3D(λ=10.0, μ=1.0)
  kinematics = Kinematics(Mechano, Solid)
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  D = BoundaryConditions(BoundaryCondition("fixed", [0.0, 0.0, 0.0]),
                         BoundaryCondition("moved", [0.0, 0.0, 0.05], EvolutionFunctions.ramp()))
  V = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), D, conformity=:H1)
  U = TrialFESpace(V, D, 0.0)
  res(Λ) = (u, v) -> residual(physmodel, kinematics, u, v, dΩ)
  jac(Λ) = (u, du, v) -> jacobian(physmodel, kinematics, u, du, v, dΩ)
  (; res, jac, U, V, D, Ω)
end

# ## Running one solver
# `solve_with` runs the whole simulation (two load increments, no bisection, the Dirichlet projection of tutorial 02)
# with the linear solver `ls` and returns the solution and the counters of the run:
# - `nls.log.niters`: Newton iterations over the run (`num_iterations(nls)` and `linear_history(nls)` give the
#   iterations of the last increment, the latter per Newton iteration);
# - `nls.log.nlinear`: Krylov iterations over the run (0 for a direct solver);
# - `preconditioner_builds(ns)`: how many times the preconditioner (the AMG hierarchy, the factorisation) was built.
#   The numerical setup `ns` of the linear solver lives in the Newton cache of the model. Gridap's `LUSolver` does
#   not count its factorisations (0 in the tables); MUMPS does (one per tangent);
# - `nls.log.t_assembly`, `t_setup`, `t_linear`: the time of the run spent assembling tangents, setting up the
#   linear solver (factorisation, AMG hierarchy) and solving the linear systems.
# The Newton tolerance is tight (``10^{-10}`` relative) so that all the solvers stop at the same accuracy.
# `backend = :compiled` assembles residual and tangent with the compiled, threaded kernels of HyperFEMKernels
# instead of Gridap's generic (lazy) evaluation: with an iterative solver the assembly would otherwise be a large
# part of every Newton iteration.
function solve_with(ls, p; forcing=nothing)
  nls = NewtonRaphson(ls; maxiter=20, atol=1e-14, rtol=1e-10, forcing)
  m = StaticNonlinearModel(p.res, p.jac, p.U, p.V, p.D; nls, backend=:compiled)
  t = @elapsed ((x, flag) = solve!(m; stepping=LoadStepping(nsteps=2, maxbisec=0), ProjectDirichlet=true))
  ns = m.caches.nls_cache.ns
  log = nls.log
  (; x=copy(x), ok=(flag == 1), newton=log.niters, krylov=log.nlinear, last=collect(linear_history(nls)),
     builds=preconditioner_builds(ns), t, t_asm=log.t_assembly, t_setup=log.t_setup, t_lin=log.t_linear, model=m)
end

# The solvers compared, each as a function of the problem (some need its trial space). Krylov tolerances are
# relative to the right-hand side.
# - `LUSolver()`: Gridap's sparse LU (UMFPACK), the reference.
# - `AMGSolver(:cg)`: CG preconditioned by one V-cycle of smoothed-aggregation AMG (AlgebraicMultigrid.jl).
# - `AMGSolver(:cg; nullspace = near_nullspace(U))`: the same, told the six rigid-body modes (three translations,
#   three rotations) of the displacement field. They are the near-kernel of an elasticity operator: the error
#   components the coarse levels must represent. Without them AlgebraicMultigrid uses its default near-null space,
#   a single vector of ones over all the dofs (one translation along ``(1,1,1)``), and the iterations grow.
# - `forcing = EisenstatWalker()`: an inexact Newton. Far from the solution the linear system only needs to be solved
#   loosely (its tolerance ``\eta_k`` follows the reduction of the Newton residual); close to it, tightly. Fewer Krylov
#   iterations, sometimes one more Newton iteration.
# - `reuse = 3`: the AMG hierarchy is rebuilt every three Newton iterations and reused in between (the operator of
#   CG is always the current tangent, only the preconditioner lags).
mech_solvers = [
  ("LU", p -> LUSolver(), nothing),
  ("CG + AMG", p -> AMGSolver(:cg; rtol=1e-10, maxiter=1000), nothing),
  ("CG + AMG, rigid modes", p -> AMGSolver(:cg; rtol=1e-10, maxiter=1000, nullspace=near_nullspace(p.U)), nothing),
  ("  + inexact Newton", p -> AMGSolver(:cg; rtol=1e-10, maxiter=1000, nullspace=near_nullspace(p.U)), EisenstatWalker()),
  ("  + inexact, reuse = 3", p -> AMGSolver(:cg; rtol=1e-10, maxiter=1000, reuse=3, nullspace=near_nullspace(p.U)), EisenstatWalker()),
]
# With PETSc (`using GridapPETSc`), the production solvers of HyperFEM for mechanics: CG with PETSc's algebraic
# multigrid GAMG and the rigid-body modes, and the parallel direct solver MUMPS (the reference when algebraic
# multigrid struggles, e.g. thin structures with few elements across the thickness).
#
# PETSc also gives access to hypre's BoomerAMG, a *classical* AMG: it chooses coarse points among the unknowns by the
# strength of the matrix couplings, instead of aggregating them as GAMG and AlgebraicMultigrid do. Its V-cycle uses
# the same symmetric smoother on the way down and up (PETSc's default, symmetric SOR/Jacobi), so it is symmetric
# and CG applies. Extra options go through `options`; HyperFEM adds the option prefix of the solver, so they are
# written as on PETSc's command line, without `-ksp_type`/`-pc_type` (the constructor sets them). BoomerAMG ignores
# the near-null space given with `nullspace` unless both `nodal_coarsen` and `vec_interp_variant` are set (PETSc's
# `PCHYPRE` manual page). The configurations compared:
# - the defaults of PETSc's interface to hypre (Falgout coarsening, classical interpolation; not hypre's own
#   recommended settings, HMIS coarsening with extended interpolation): one scalar AMG over all the dofs, as if the
#   three displacement components were unrelated unknowns;
# - a configuration for elasticity that uses the rigid-body modes, with `blocksize = 3` (the matrix is made of
#   3 × 3 nodal blocks; the bar has no component-wise Dirichlet masks, so its free dofs come in complete nodes):
#   - `nodal_coarsen 6` coarsens the graph of the nodes instead of the scalar dofs, the strength of the coupling
#     between two nodes being the sum of all the entries of their 3 × 3 block (`4` uses the row-sum norm of the
#     block, `1` its Frobenius norm: `HYPRE_BoomerAMGSetNodal` in hypre's reference manual);
#   - `vec_interp_variant 2` adds the given vectors to the interpolation (hypre's "GM approach 2"), so that the
#     coarse levels can represent the rigid-body modes;
#   - `interp_type ext+i`: the extended interpolation, which reaches the neighbours of neighbours and stays accurate
#     on the sparse coarse grids of `coarsen_type HMIS` (a parallel coarsening that keeps fewer coarse points than
#     the Falgout default);
#   - `strong_threshold 0.5`: a coupling is strong if it is above half of the largest one in its row. The default,
#     0.25, suits 2D diffusion; hypre's manual suggests 0.5–0.6 for 3D Laplace operators and larger values, such as
#     0.9, for elasticity, but 0.9 was worse on this bar;
#   - `agg_nl 1`: aggressive coarsening on the first level (far fewer coarse points, a cheaper hierarchy). With the
#     vector interpolation it is essential: without it the hierarchy is so dense that a Newton iteration takes about
#     20 s on the full bar.
# - the same with `nodal_coarsen 4`; and with `nodal_coarsen 4` given only the three rotations,
#   `near_nullspace(U)[:, 4:6]` (the translations, the first three columns, are already reproduced by the
#   interpolation of every displacement component; this is also what MFEM passes to BoomerAMG for elasticity).
boomeramg_options(nodal) = "-pc_hypre_boomeramg_nodal_coarsen $nodal -pc_hypre_boomeramg_vec_interp_variant 2 " *
  "-pc_hypre_boomeramg_interp_type ext+i -pc_hypre_boomeramg_coarsen_type HMIS -pc_hypre_boomeramg_strong_threshold 0.5 " *
  "-pc_hypre_boomeramg_agg_nl 1"
boomeramg(p, nodal, modes) = PETScSolver(:cg, :boomeramg; rtol=1e-10, maxiter=1000, nullspace=modes,
                                         blocksize=3, options=boomeramg_options(nodal))
if PETSC
  push!(mech_solvers,
    ("CG + GAMG (PETSc), rigid modes", p -> PETScSolver(:cg, :gamg; rtol=1e-10, maxiter=1000, nullspace=near_nullspace(p.U)), nothing))
  ## the PETSc defaults cost about 2 minutes per Newton iteration on the bar of the full run: FAST only
  FAST && push!(mech_solvers,
    ("CG + BoomerAMG, PETSc defaults", p -> PETScSolver(:cg, :boomeramg; rtol=1e-10, maxiter=1000), nothing))
  push!(mech_solvers,
    ("CG + BoomerAMG, nodal 6, rigid modes", p -> boomeramg(p, 6, near_nullspace(p.U)), nothing),
    ("CG + BoomerAMG, nodal 4, rigid modes", p -> boomeramg(p, 4, near_nullspace(p.U)), nothing),
    ("CG + BoomerAMG, nodal 4, rotations", p -> boomeramg(p, 4, near_nullspace(p.U)[:, 4:6]), nothing),
    ("MUMPS (PETSc)", p -> PETScSolver(:preonly, :mumps), nothing))
end

# ## Solve
# First every solver on a small bar (compilation), then on the bar of the tutorial.
n = FAST ? 4 : 16
for (name, make, forcing) in mech_solvers
  pw = bar_problem(2)
  solve_with(make(pw), pw; forcing)
end
p = bar_problem(n)
ndofs = num_free_dofs(p.U)
println("Bar: $(5n)×$(n)×$(n) Q1 hexahedra, $ndofs free dofs")
mech = Dict{String,Any}()
for (name, make, forcing) in mech_solvers
  mech[name] = solve_with(make(p), p; forcing)
end

# ## Results
# Newton iterations, Krylov iterations (total over the run and per Newton iteration), preconditioner builds, time of
# the run and, per Newton iteration, of the setup of the linear solver plus the linear solve ("solve/it"), and the
# difference from the LU solution.
function print_table(results, names, ref)
  @printf("  %-42s %6s %7s %7s %6s %8s %9s %9s\n", "solver", "Newton", "Krylov", "per it", "builds", "run (s)", "solve/it", "vs LU")
  for name in names
    r = results[name]
    err = norm(r.x - ref.x) / norm(ref.x)
    @printf("  %-42s %6d %7d %7.1f %6d %8.2f %9.3f %9.1e\n", name, r.newton, r.krylov, r.krylov / r.newton, r.builds, r.t,
            (r.t_setup + r.t_lin) / r.newton, err)
  end
end
print_table(mech, first.(mech_solvers), mech["LU"])
# A measured example (full run, Mac M5 Max, 4 Julia threads, 4 BLAS threads, PETSc serial; times change with the machine, the
# iterations do not):
# ```
# Bar: 80×16×16 Q1 hexahedra, 68493 free dofs
#   solver                                     Newton  Krylov  per it builds  run (s)  solve/it     vs LU
#   LU                                              6       0     0.0      0    18.14     2.183   0.0e+00
#   CG + AMG                                        6    1163   193.8      8    21.20     2.583   9.8e-17
#   CG + AMG, rigid modes                           6     203    33.8      8     9.02     1.059   9.6e-17
#     + inexact Newton                              6      73    12.2      8     6.73     0.695   3.1e-13
#     + inexact, reuse = 3                          6      78    13.0      3     4.30     0.377   4.0e-13
#   CG + GAMG (PETSc), rigid modes                  6     212    35.3      8     4.99     0.505   9.3e-17
#   CG + BoomerAMG, nodal 6, rigid modes            6     404    67.3      8    19.40     2.340   9.5e-17
#   CG + BoomerAMG, nodal 4, rigid modes            6     193    32.2      8     9.32     1.082   8.7e-17
#   CG + BoomerAMG, nodal 4, rotations              6     190    31.7      8     6.80     0.784   9.1e-17
#   MUMPS (PETSc)                                   6       6     1.0      8     8.92     1.044   8.1e-17
# ```
# Reading the table (full run):
# - without the rigid-body modes CG + AMG needs about six times more iterations (194 instead of 34 per Newton
#   iteration) and is slower than LU; with them the Newton iterations are those of LU, to round-off;
# - the inexact Newton cuts the Krylov iterations by a factor of about three with the same Newton iterations;
# - `reuse = 3` builds the hierarchy 3 times instead of 8 (6 Newton iterations plus the 2 Dirichlet projections, which
#   go through the same linear solver);
# - at 68 000 dofs the AMG configurations with rigid modes already beat LU per Newton iteration, and the gap widens
#   with the size (the cost of LU grows much faster than the number of dofs; tutorial 14 shows multigrid iterations
#   that do not grow). With the 1 425 dofs of the FAST run LU is the fastest.
# - MUMPS, also a direct solver, factorises this matrix faster than UMFPACK here; it is the reference when algebraic
#   multigrid fails.
# - BoomerAMG with PETSc's defaults (FAST only) needs 41.5 CG iterations per Newton iteration on the small bar, but
#   its hierarchy is expensive: on the full bar, measured separately, 121 iterations and about 127 s per Newton
#   iteration, against 2.2 s for LU.
# - With `nodal_coarsen 6` the elasticity configuration converges, but its iterations grow with the mesh (41.5 in
#   FAST, 67 in the full run). With `nodal_coarsen 4` (row-sum norm of the blocks) they stay at about 32 and the
#   time per Newton iteration falls by half; giving only the three rotations keeps the iterations and makes the
#   hierarchy cheaper. That last configuration is the BoomerAMG to use here: about 1.6 times the time of GAMG with
#   the rigid modes, which remains the PETSc choice for mechanics on this bar.
#
# The compiled assembly runs on Julia's threads; AlgebraicMultigrid's `AMGSolver` and PETSc run in serial here. The
# times also depend on the load of the machine: the iterations are what to compare. With PETSc and several Julia
# threads, GridapPETSc may print `Error in running finalizer: AssertionError("Threads.threadid() == 1")` and
# "objects still not finalized" at exit: they come from GridapPETSc's clean-up and are harmless.

# ## Assembly: compiled kernels against Gridap
# One tangent assembly at the final state, after a warm-up, with the compiled operator the model built and with
# Gridap's `assemble_matrix` on the same weak form. `HyperFEMKernels.work_summary` tells how every integration group
# was compiled: `:factorised` and `:linear_maps` are the fast paths, `:duals` the generic one (automatic
# differentiation at every point), and `fallback_*` counts parts left to Gridap.
let r = mech["LU"], op = r.model.caches.op
  ws = HyperFEMKernels.work_summary(op)
  println("Compiled groups: ", [(g.kind, g.domain, g.path) for g in ws.groups],
          "; Gridap fallbacks: $(ws.fallback_residual + ws.fallback_jacobian)")
  A = Gridap.Algebra.allocate_jacobian(op, r.x)
  Gridap.Algebra.jacobian!(A, op, r.x)
  t_compiled = minimum(@elapsed(Gridap.Algebra.jacobian!(A, op, r.x)) for _ in 1:3)
  uh = get_state(r.model)
  a(du, v) = p.jac(1.0)(uh, du, v)
  Ag = assemble_matrix(a, p.U, p.V)
  t_gridap = minimum(@elapsed(assemble_matrix(a, p.U, p.V)) for _ in 1:2)
  @printf("One tangent assembly (%d dofs): compiled %.3f s, Gridap %.3f s, same matrix to %.1e\n",
          ndofs, t_compiled, t_gridap, norm(A - Ag) / norm(Ag))
  global assembly_diff = norm(A - Ag) / norm(Ag)
  global compiled_paths = [g.path for g in ws.groups]
end

# The deformed bar computed with LU, for ParaView.
writevtk(p.Ω, joinpath(outdir, "bar"), cellfields=["u" => get_state(mech["LU"].model)])

# ## Multi-field problems: block preconditioners
# A dielectric elastomer box ``(0,1)\times(0,0.2)\times(0,0.2)`` (`ElectroMechModel` of tutorial 04: neo-Hookean
# with ``\lambda = 10``, ``\mu = 1``, ideal dielectric ``\varepsilon = 1``), clamped at ``x = 0``, with a potential
# difference of 0.02 between its bottom and top faces. The unknowns are ``(\boldsymbol u, \varphi)`` and the
# tangent is a symmetric saddle point,
# ```math
# \begin{pmatrix} \boldsymbol K_{uu} & \boldsymbol K_{u\varphi} \\ \boldsymbol K_{\varphi u} & \boldsymbol K_{\varphi\varphi} \end{pmatrix},
# \qquad \boldsymbol K_{uu} \text{ positive definite}, \quad \boldsymbol K_{\varphi\varphi} \text{ negative definite},
# ```
# on which a single AMG hierarchy fails. A block (field-split) preconditioner uses one AMG per field, with the
# rigid-body modes on the displacement block: additively (block Jacobi), multiplicatively (block Gauss–Seidel) or
# through the Schur complement of ``\boldsymbol K_{uu}``, inside GMRES. `AMGSolver(U; split)` and `PETScSolver(U; …)`
# take the field structure from the multi-field trial space `U`.
function em_problem(nx, ny, nz)
  model = CartesianDiscreteModel((0.0, 1.0, 0.0, 0.2, 0.0, 0.2), (nx, ny, nz))
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
  add_tag_from_tags!(labels, "top", CartesianTags.faceXY1⁺)
  add_tag_from_tags!(labels, "bottom", CartesianTags.faceXY0⁺)
  physmodel = ElectroMechModel(IdealDielectric(ε=1.0), NeoHookean3D(λ=10.0, μ=1.0))
  kinematics = (Kinematics(Mechano, Solid), Kinematics(Electro, Solid))
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  Du = BoundaryConditions(BoundaryCondition("fixed", [0.0, 0.0, 0.0]))
  Dφ = BoundaryConditions(BoundaryCondition("bottom", 0.0), BoundaryCondition("top", 0.02, EvolutionFunctions.ramp()))
  D = MultiFieldBC([Du, Dφ])
  Vu = TestFESpace(Ω, ReferenceFE(lagrangian, VectorValue{3,Float64}, 1), Du, conformity=:H1)
  Vφ = TestFESpace(Ω, ReferenceFE(lagrangian, Float64, 1), Dφ, conformity=:H1)
  V = MultiFieldFESpace([Vu, Vφ])
  U = MultiFieldFESpace([TrialFESpace(Vu, Du, 0.0), TrialFESpace(Vφ, Dφ, 0.0)])
  res(Λ) = ((u, φ), (v, vφ)) -> residual(physmodel, kinematics, (u, φ), (v, vφ), dΩ)
  jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> jacobian(physmodel, kinematics, (u, φ), (du, dφ), (v, vφ), dΩ)
  (; res, jac, U, V, D, Ω)
end

em_solvers = [
  ("LU", p -> LUSolver(), nothing),
  ("GMRES + AMG blocks, additive", p -> AMGSolver(p.U; split=:additive, rtol=1e-10), nothing),
  ("GMRES + AMG blocks, Schur", p -> AMGSolver(p.U; split=:schur, rtol=1e-10), nothing),
  ("  + inexact, reuse = 3", p -> AMGSolver(p.U; split=:schur, rtol=1e-10, reuse=3), EisenstatWalker()),
]
# The production configuration of HyperFEM for electro- and magneto-mechanics: PETSc's field split with GAMG on
# every block, rebuilt every three Newton iterations, and the inexact Newton.
if PETSC
  push!(em_solvers, ("PETSc field split, reuse = 3, inexact", p -> PETScSolver(p.U; rtol=1e-10, reuse=3), EisenstatWalker()))
end

em_cells = FAST ? (10, 2, 4) : (40, 8, 16)
for (name, make, forcing) in em_solvers
  pw = em_problem(5, 1, 2)
  solve_with(make(pw), pw; forcing)
end
pem = em_problem(em_cells...)
ndofs_em = num_free_dofs(pem.U)
println("Electro-mechanical box: $(join(em_cells, "×")) Q1 hexahedra, $ndofs_em free dofs")
em = Dict{String,Any}()
for (name, make, forcing) in em_solvers
  em[name] = solve_with(make(pem), pem; forcing)
end
print_table(em, first.(em_solvers), em["LU"])
# A measured example (full run, same machine):
# ```
# Electro-mechanical box: 40×8×16 Q1 hexahedra, 23895 free dofs
#   solver                                     Newton  Krylov  per it builds  run (s)  solve/it     vs LU
#   LU                                              6       0     0.0      0     3.96     0.433   0.0e+00
#   GMRES + AMG blocks, additive                    6     336    56.0      8     3.16     0.389   2.9e-16
#   GMRES + AMG blocks, Schur                       6     330    55.0      8     3.19     0.387   8.3e-16
#     + inexact, reuse = 3                          7     135    19.3      3     1.73     0.143   1.5e-11
#   PETSc field split, reuse = 3, inexact           6     147    24.5      3     1.15     0.078   5.0e-11
# ```
# The block preconditioners follow the Newton trajectory of LU; with the inexact Newton and `reuse = 3` the
# AMG blocks are built 3 times instead of 8 (one more Newton iteration here) and the Krylov iterations drop by a
# factor of about 2.5. The PETSc field split is the fastest configuration on this problem.

# ## Checks
# Every solver reaches the LU solution (the Newton tolerance, ``10^{-10}``, through the conditioning of the
# tangent, bounds the difference). The solvers that solve every linear system to ``10^{-10}`` follow the Newton
# trajectory of LU: the same number of Newton iterations. The counters behave as described above.
for (results, names) in ((mech, first.(mech_solvers)), (em, first.(em_solvers)))
  ref = results["LU"]
  for name in names
    r = results[name]
    @assert r.ok "$name: the load stepping did not reach Λ = 1"
    @assert norm(r.x - ref.x) ≤ 1e-6 * norm(ref.x) "$name: solution away from LU"
  end
  for name in names
    occursin("inexact", name) && continue
    @assert results[name].newton == ref.newton "$name: Newton iterations differ from LU"
  end
end
@assert mech["CG + AMG, rigid modes"].krylov < mech["CG + AMG"].krylov "the rigid-body modes did not reduce the iterations"
@assert mech["  + inexact Newton"].krylov < mech["CG + AMG, rigid modes"].krylov "the inexact Newton did not reduce the Krylov iterations"
@assert mech["  + inexact, reuse = 3"].builds < mech["  + inexact Newton"].builds "reuse did not reduce the preconditioner builds"
@assert em["  + inexact, reuse = 3"].builds < em["GMRES + AMG blocks, Schur"].builds "reuse did not reduce the preconditioner builds (EM)"
@assert assembly_diff < 1e-12 "the compiled tangent differs from Gridap's"
@assert all(!=(:duals), compiled_paths) "a group of the compiled operator took the generic (dual number) path"
if PETSC
  @assert norm(mech["MUMPS (PETSc)"].x - mech["LU"].x) ≤ 1e-10 * norm(mech["LU"].x) "MUMPS away from LU"
  ## BoomerAMG, the recommended configuration: CG iterations per Newton iteration in a band around the measured value
  bamg_its = mech["CG + BoomerAMG, nodal 4, rotations"].krylov / mech["CG + BoomerAMG, nodal 4, rotations"].newton
  bamg_band = FAST ? (24.0, 36.0) : (25.0, 40.0)              # measured: 29.7 (FAST), 31.7 (full)
  bamg6_its = mech["CG + BoomerAMG, nodal 6, rigid modes"].krylov / mech["CG + BoomerAMG, nodal 6, rigid modes"].newton
  @assert bamg_its < bamg6_its "BoomerAMG: nodal coarsening 4 with the rotations did not improve on nodal coarsening 6"
  @assert bamg_band[1] ≤ bamg_its ≤ bamg_band[2] "BoomerAMG: $(round(bamg_its, digits=1)) iterations per Newton iteration, out of $bamg_band"
end
# Krylov iterations per Newton iteration of CG + AMG with the rigid modes, in a band around the measured value
# (a regression of the preconditioner, not of the solution).
amg_per_newton = mech["CG + AMG, rigid modes"].krylov / mech["CG + AMG, rigid modes"].newton
println("CG + AMG (rigid modes): $(round(amg_per_newton, digits=1)) iterations per Newton iteration")
amg_band = FAST ? (17.0, 28.0) : (25.0, 42.0)          # measured: 22.7 (FAST), 33.8 (full)
@assert amg_band[1] ≤ amg_per_newton ≤ amg_band[2] "CG + AMG iterations per Newton iteration out of the band $amg_band"
# Regression values of the LU solution (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0,
# macOS arm64).
xref_mech, xref_em = FAST ? (0.6610687966879161, 0.10822488398193163) : (4.59603534564497, 0.8492741596170085)
println("‖x‖ bar = $(norm(mech["LU"].x)), ‖x‖ box = $(norm(em["LU"].x))")
@assert isapprox(norm(mech["LU"].x), xref_mech; rtol=1e-5) "bar: regression value"
@assert isapprox(norm(em["LU"].x), xref_em; rtol=1e-5) "box: regression value"

# ## Going further
# - Tutorial 14 replaces algebraic by geometric multigrid on a hierarchy of refined meshes (`HMultigridSolver`) and
#   by p-multigrid with matrix-free high-order operators.
# - `options = "-ksp_monitor"` (or `monitor = true`) prints PETSc's residual history; any PETSc option of the
#   solver goes through `options`. BoomerAMG has many more (smoothers, cycle type): PETSc's `PCHYPRE` page lists them.
# - `backend = CompiledBackend(jacobian = :matrixfree)` never assembles the tangent of the Newton: tutorial 14 uses it
#   with p-multigrid, and `HyperFEMKernels.LaggedAssembled(PETScSolver(:preonly, :mumps); every = 3)` preconditions
#   it with a factorisation renewed every three Newton iterations.
# - `NewtonRaphson(…; verbose = 2)` prints, per Newton iteration, the linear solver, its iterations and the time.
# - References: Eisenstat and Walker, SIAM J. Sci. Comput. 17 (1996) 16–32, doi:10.1137/0917003 (forcing terms);
#   Vaněk, Mandel and Brezina, Computing 56 (1996) 179–196, doi:10.1007/BF02238511 (smoothed aggregation).
