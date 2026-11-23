# # Tutorial 19: Constitutive models: energies, stresses and tangents
#
# Every simulation of HyperFEM starts from a constitutive model: a small object that returns the stored
# energy ``\Psi(\boldsymbol F)``, the first Piola–Kirchhoff stress ``\partial_{\boldsymbol F}\Psi`` and the
# tangent ``\partial^2_{\boldsymbol F\boldsymbol F}\Psi`` as plain Julia functions. This tutorial uses them
# without finite elements: it evaluates several hyperelastic models in the three classical homogeneous tests
# (uniaxial tension, equibiaxial tension and simple shear), checks them against closed-form solutions and
# against their own derivatives, and writes a new model with the same interface.
#
# **What you will learn**
# - the interface of the `PhysicalModels`: `Ψ, ∂Ψ∂F, ∂Ψ∂FF = model()`, and the extra arguments of the
#   anisotropic models (the fibre directions);
# - the catalogue of 3D hyperelastic models: `NeoHookean3D`, `MooneyRivlin3D`, `Yeoh3D`, `EightChain`,
#   `IncompressibleNeoHookean3D`, `TransverseIsotropy3D`, `HGO_1Fiber`, `HGO_4Fibers`, and how to add
#   models with `+`;
# - the kinematics object (`Kinematics`, `get_Kinematics`, `getIsoInvariants`);
# - homogeneous tests: the incompressible closed forms and the compressible ones (lateral faces free of
#   traction, found with a scalar Newton iteration that uses the tangent);
# - how to verify a model (derivatives, objectivity, stress-free reference state) and how to write your own;
# - why the models are fast inside the compiled kernels: they are allocation-free `isbits` closures.
#
# Measured on a Mac M5 Max (Julia 1.13.0, 4 Julia threads, 4 BLAS threads; HyperFEM 0.3.0,
# HyperFEMKernels 0.2.6, Gridap 0.20.8): about 7 s including package loading and compilation; there is no FAST variant (the
# reduced and the full runs are the same, the computation takes milliseconds). No degrees of freedom: only
# point evaluations.
#
# ## Problem statement
# A hyperelastic material is defined by a stored energy per unit reference volume ``\Psi(\boldsymbol F)``
# of the deformation gradient ``\boldsymbol F = \boldsymbol I + \nabla_0\boldsymbol u``. The first
# Piola–Kirchhoff (nominal) stress and the tangent used by Newton's method are
# ```math
# \boldsymbol P = \frac{\partial \Psi}{\partial \boldsymbol F}, \qquad
# \mathbb A = \frac{\partial^2 \Psi}{\partial \boldsymbol F\,\partial \boldsymbol F}, \qquad
# \mathrm d\boldsymbol P = \mathbb A : \mathrm d\boldsymbol F .
# ```
# Most models are written with the cofactor ``\boldsymbol H = J\boldsymbol F^{-T}`` and the Jacobian
# ``J = \det\boldsymbol F``, and with the invariants ``I_1 = \boldsymbol F:\boldsymbol F`` and
# ``I_2 = \boldsymbol H:\boldsymbol H``.
#
# Homogeneous deformations, with ``\boldsymbol e_i`` the Cartesian basis:
# - uniaxial tension, ``\boldsymbol F = \mathrm{diag}(\lambda, \lambda_t, \lambda_t)`` with ``P_{22} = P_{33} = 0``;
# - equibiaxial tension, ``\boldsymbol F = \mathrm{diag}(\lambda, \lambda, \lambda_3)`` with ``P_{33} = 0``;
# - simple shear, ``\boldsymbol F = \boldsymbol I + \gamma\, \boldsymbol e_1 \otimes \boldsymbol e_2`` (``J = 1``).
#
# For an incompressible material (``J = 1``, ``\lambda_t = \lambda^{-1/2}``, ``\lambda_3 = \lambda^{-2}``) with
# ``\Psi = W(I_1, I_2)``, ``W_1 = \partial W/\partial I_1`` and ``W_2 = \partial W/\partial I_2``, the nominal
# stresses are (Ogden, *Non-linear elastic deformations*, 1984, §7.2)
# ```math
# P_{\mathrm{uni}} = 2\left(\lambda - \lambda^{-2}\right)\left(W_1 + \lambda^{-1} W_2\right), \qquad
# P_{\mathrm{equi}} = 2\left(\lambda - \lambda^{-5}\right)\left(W_1 + \lambda^{2} W_2\right), \qquad
# P_{12} = 2\gamma\,(W_1 + W_2).
# ```
# The models of HyperFEM are compressible: they add volumetric terms of ``J`` to ``W``. At ``J = 1`` those
# terms only contribute stresses proportional to ``\boldsymbol F^{-T}``, which an incompressible material
# absorbs in its pressure: ``\boldsymbol P = \hat{\boldsymbol P}(\boldsymbol F) - p\,\boldsymbol F^{-T}``,
# with ``p`` given by the traction-free lateral faces. That is how the closed forms are checked below.
#
# ## Setup
using HyperFEM
using Gridap
using LinearAlgebra
using Printf

const FAST = get(ENV, "HYPERFEM_TUTORIALS_FAST", "0") == "1"   # same case in both modes (point evaluations)
outdir = joinpath(@__DIR__, "..", "data", "sims", "E19_Constitutive_models")
setupfolder(outdir)                                            # CSV tables (not versioned)

const e1, e2, e3 = VectorValue(1.0, 0.0, 0.0), VectorValue(0.0, 1.0, 0.0), VectorValue(0.0, 0.0, 1.0)
diagF(a, b, c) = TensorValue(a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c)   # column-major storage

# ## Kinematics
# `Kinematics(Mechano, Solid)` holds the maps used by the weak forms: ``\boldsymbol F(\nabla\boldsymbol u)``,
# ``\boldsymbol H(\boldsymbol F)`` and ``J(\boldsymbol F)``. In a weak form they are composed with the FE
# gradient, `F ∘ ∇(u)'` (Gridap's `∇(u)` is the transpose of ``\nabla_0\boldsymbol u``); here we call them
# with a tensor. `getIsoInvariants` gives ``I_1``, ``I_2`` and ``I_3 = J``.
kinematics = Kinematics(Mechano, Solid)
Fmap, Hmap, Jmap = get_Kinematics(kinematics)
I1map, I2map, I3map = getIsoInvariants(kinematics)
∇u = TensorValue(0.10, 0.02, 0.00, 0.05, -0.03, 0.01, 0.00, 0.02, 0.04)
F = Fmap(∇u)
println("F = ", F)
println("J = ", Jmap(F), ",  I1 = ", I1map(F), ",  I2 = ", I2map(F))
@assert Fmap(∇u) == one(∇u) + ∇u
@assert norm(Hmap(F) - det(F) * inv(F)') < 1e-14 * norm(Hmap(F)) "H is the cofactor J F⁻ᵀ"
@assert abs(I2map(F) - Hmap(F) ⊙ Hmap(F)) < 1e-14 * I2map(F)

# ## A first model
# A model is a struct with its parameters; calling it with no arguments returns three functions of ``\boldsymbol F``
# (closures that capture the parameters). The stress is a `TensorValue{3,3}` and the tangent a
# `FourthOrderTensorValue{3}`, applied to an increment with the double contraction `⋅²`, as in the
# jacobian of tutorial 02.
nh = NeoHookean3D(λ=100.0, μ=1.0)                              # kPa
Ψ, ∂Ψ∂F, ∂Ψ∂FF = nh()
println("Ψ(F) = ", Ψ(F), " kPa")
println("P(F) = ", ∂Ψ∂F(F))
println(typeof(∂Ψ∂FF(F)))
dF = 1e-6 * TensorValue(1.0, -2.0, 0.5, 0.3, 1.0, -0.7, 0.2, 0.4, -1.0)
println("‖P(F + dF) - P(F) - A:dF‖ / ‖A:dF‖ = ", norm(∂Ψ∂F(F + dF) - ∂Ψ∂F(F) - ∂Ψ∂FF(F) ⋅² dF) / norm(∂Ψ∂FF(F) ⋅² dF))

# ## The model catalogue
# The 3D hyperelastic models of HyperFEM 0.3.0 (their code is in HyperFEM's
# `src/PhysicalModels/MechanicalModels.jl`; some, like `?Yeoh3D` or `?EightChain`, have docstrings), all with a
# shear modulus close to 1 kPa and a bulk penalty ``\lambda = 100`` kPa (nearly incompressible):
# - `NeoHookean3D(λ, μ)`: ``\tfrac{\mu}{2}(I_1-3) - \mu\ln J + \tfrac{\lambda}{2}(J-1)^2``;
# - `MooneyRivlin3D(λ, μ1, μ2)`: ``\tfrac{\mu_1}{2}(I_1-3) + \tfrac{\mu_2}{2}(I_2-3) - (\mu_1+2\mu_2)\ln J + \tfrac{\lambda}{2}(J-1)^2``;
# - `Yeoh3D(λ, C10, C20, C30)`: ``\sum_{k} C_{k0}(I_1-3)^k - 2C_{10}\ln J + \tfrac{\lambda}{2}(J-1)^2``;
# - `EightChain(μ, N)`: Arruda–Boyce in the isochoric invariant ``\bar I_1 = J^{-2/3} I_1`` only, so it needs
#   a volumetric term: here `+ VolumetricEnergy(λ)`, ``\tfrac{\lambda}{2}(J-1)^2``;
# - `IncompressibleNeoHookean3D(λ, μ, δ)`: neo-Hookean in ``\bar I_1`` with a regularised ``J`` (parameter
#   ``\delta``) and a penalty ``\lambda``. HyperFEM has no exactly incompressible (mixed ``u``–``p``) model:
#   incompressibility is always a penalty;
# - fibre models, functions of ``(\boldsymbol F, \boldsymbol N)`` with ``\boldsymbol N`` the fibre direction in
#   the reference configuration: `TransverseIsotropy3D(μ, α1, α2)`, polyconvex in ``I_4 = |\boldsymbol F\boldsymbol N|^2``
#   and ``I_5 = |\boldsymbol H\boldsymbol N|^2``, and the Holzapfel–Gasser–Ogden fibre energy `HGO_1Fiber(c1, c2)`,
#   ``\tfrac{c_1}{4c_2}\left(e^{c_2(I_4-1)^2}-1\right)`` (`HGO_4Fibers` has four families). They have no
#   volumetric term of their own and are added to an isotropic matrix.
#
# **Adding models.** `+` composes an isotropic model with another isotropic one (the result is isotropic) or
# with an anisotropic one (the result takes the fibre directions as extra arguments). `Gent` exists only in
# 2D (`Gent2D`, plane strain); a 3D Gent model is written below as the example of a user model.
const N_fibre = e1                                             # fibres along the loading direction
const N4 = (e1, VectorValue(0.0, 1.0, 1.0), VectorValue(1.0, 1.0, 0.0), VectorValue(1.0, 0.0, 1.0))

# ## Writing your own model
# A model is any subtype of `Mechano` callable with no arguments that returns `(Ψ, ∂Ψ∂F, ∂Ψ∂FF)`. That is
# all that `residual`, `jacobian` and `energy` use, so the new model works in every mechanical tutorial
# (02, 03, 13–15). Subtyping the (non-exported) `HyperFEM.PhysicalModels.IsoElastic` instead also enables
# `+` with the other models.
#
# The compressible Gent model in 3D, with ``J_m`` the limit of ``I_1 - 3`` (finite extensibility):
# ```math
# \Psi = -\frac{\mu J_m}{2}\ln\left(1 - \frac{I_1-3}{J_m}\right) - \mu\ln J + \frac{\lambda}{2}(J-1)^2 ,
# \qquad \boldsymbol P = \frac{\mu}{g}\boldsymbol F + \Psi_J \boldsymbol H ,
# \qquad g = 1 - \frac{I_1-3}{J_m}, \quad \Psi_J = -\frac{\mu}{J} + \lambda (J-1),
# ```
# ```math
# \mathbb A = \frac{\mu}{g}\,\mathbb I + \frac{2\mu}{J_m g^2}\,\boldsymbol F\otimes\boldsymbol F
#   + \left(\frac{\mu}{J^2} + \lambda\right)\boldsymbol H\otimes\boldsymbol H + \Psi_J\,\frac{\partial\boldsymbol H}{\partial\boldsymbol F} .
# ```
# `𝕀3D` is the fourth-order identity and `×ᵢ⁴(F)` is ``\partial\boldsymbol H/\partial\boldsymbol F`` (both
# exported by HyperFEM). The three functions read only their argument and the parameters captured from the
# struct, never a non-constant global variable (a non-constant global has an unknown type, so every call
# that reads it is resolved at run time and is many times slower); so they compile to allocation-free code.
struct Gent3D <: HyperFEM.PhysicalModels.IsoElastic
  μ::Float64
  Jm::Float64
  λ::Float64
end
Gent3D(; μ::Float64, Jm::Float64, λ::Float64) = Gent3D(μ, Jm, λ)

function (obj::Gent3D)()
  μ, Jm, λ = obj.μ, obj.Jm, obj.λ
  J(F) = det(F)
  H(F) = det(F) * inv(F)'
  g(F) = 1.0 - (tr(F' * F) - 3.0) / Jm
  ΨJ(F) = -μ / J(F) + λ * (J(F) - 1.0)
  Ψ(F) = -0.5 * μ * Jm * log(g(F)) - μ * log(J(F)) + 0.5 * λ * (J(F) - 1.0)^2
  ∂Ψ∂F(F) = (μ / g(F)) * F + ΨJ(F) * H(F)
  ∂Ψ∂FF(F) = (μ / g(F)) * 𝕀3D + (2.0 * μ / (Jm * g(F)^2)) * (F ⊗ F) +
             (μ / J(F)^2 + λ) * (H(F) ⊗ H(F)) + ΨJ(F) * ×ᵢ⁴(F)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂FF)
end

# A second user model: a fibre-reinforced material that adds an isotropic matrix and a fibre model, like
# `+`, but with the types of both parts as parameters of the struct and with the fibre arguments declared as
# `Vararg{Any,K}`. That declaration makes Julia compile one method per number of fibres; the reason is
# explained in the Efficiency section.
struct FibreReinforced{M<:Mechano,A<:Mechano} <: Mechano
  matrix::M
  fibres::A
end

function (obj::FibreReinforced)()
  Ψm, Pm, Am = obj.matrix()
  Ψf, Pf, Af = obj.fibres()
  Ψ(F, N::Vararg{Any,K}) where {K} = Ψm(F) + Ψf(F, N...)
  ∂Ψ∂F(F, N::Vararg{Any,K}) where {K} = Pm(F) + Pf(F, N...)
  ∂Ψ∂FF(F, N::Vararg{Any,K}) where {K} = Am(F) + Af(F, N...)
  return (Ψ, ∂Ψ∂F, ∂Ψ∂FF)
end

# The models of this tutorial: a name, the model and the extra arguments of its functions (the fibres).
models = [
  (name="NeoHookean3D",       model=nh,                                                      args=()),
  (name="MooneyRivlin3D",     model=MooneyRivlin3D(λ=100.0, μ1=0.7, μ2=0.3),                 args=()),
  (name="Yeoh3D",             model=Yeoh3D(λ=100.0, C10=0.5, C20=0.02, C30=0.001),            args=()),
  (name="EightChain+Vol",     model=EightChain(μ=1.0, N=8.0) + VolumetricEnergy(λ=100.0),    args=()),
  (name="IncompNeoHookean3D", model=IncompressibleNeoHookean3D(λ=100.0, μ=1.0),             args=()),
  (name="Gent3D (user)",      model=Gent3D(μ=1.0, Jm=20.0, λ=100.0),                         args=()),
  (name="NH+TransvIso",       model=nh + TransverseIsotropy3D(μ=0.5, α1=1.5, α2=1.3),        args=(N_fibre,)),
  (name="NH+HGO_1Fiber",      model=nh + HGO_1Fiber(c1=1.0, c2=0.2),                         args=(N_fibre,)),
  (name="NH+HGO_4Fibers",     model=nh + HGO_4Fibers(c1=[2.0, 1.0, 1.0, 1.0], c2=[0.5, 0.5, 0.5, 0.5]), args=N4),
  (name="FibreReinforced",    model=FibreReinforced(nh, HGO_4Fibers(c1=[2.0, 1.0, 1.0, 1.0], c2=[0.5, 0.5, 0.5, 0.5])), args=N4),
]

# ## Homogeneous tests
# For a compressible model the lateral stretch is unknown. The scalar equation ``P_{22}(\lambda_t) = 0`` is
# solved by Newton's method with the model's own tangent, ``\mathrm dP_{22}/\mathrm d\lambda_t =
# [\mathbb A : (\boldsymbol e_2\otimes\boldsymbol e_2 + \boldsymbol e_3\otimes\boldsymbol e_3)]_{22}``; for the
# equibiaxial test the unknown is ``\lambda_3`` and the equation ``P_{33} = 0``. The fibres of the anisotropic
# models are along ``\boldsymbol e_1``, so ``\lambda_2 = \lambda_3`` still holds in uniaxial tension (with
# four fibre families it does not: those models are only used in the checks).
function lateral_newton(∂Ψ∂F, ∂Ψ∂FF, Fof, dFda, i, a, args; tol=1e-13, maxiter=30)
  for _ in 1:maxiter
    F = Fof(a)
    r = ∂Ψ∂F(F, args...)[i, i]
    abs(r) < tol && return a, F
    a -= r / (∂Ψ∂FF(F, args...) ⋅² dFda)[i, i]
  end
  error("lateral equilibrium did not converge")
end

uniaxial(∂Ψ∂F, ∂Ψ∂FF, λ, args) = lateral_newton(∂Ψ∂F, ∂Ψ∂FF, a -> diagF(λ, a, a), e2 ⊗ e2 + e3 ⊗ e3, 2, 1 / sqrt(λ), args)
equibiaxial(∂Ψ∂F, ∂Ψ∂FF, λ, args) = lateral_newton(∂Ψ∂F, ∂Ψ∂FF, a -> diagF(λ, λ, a), e3 ⊗ e3, 3, 1 / λ^2, args)
shearF(γ) = one(TensorValue{3,3,Float64}) + γ * (e1 ⊗ e2)

stretches = collect(1.0:0.1:2.0)
shears = collect(0.0:0.1:1.0)
iso_models = filter(m -> length(m.args) <= 1, models)    # the four-fibre models are only checked
table_uni = zeros(length(stretches), length(iso_models))
table_lat = zeros(length(stretches), length(iso_models))
table_equi = zeros(length(stretches), length(iso_models))
table_shear = zeros(length(shears), length(iso_models))
for (j, m) in enumerate(iso_models)
  _, Pm, Am = m.model()
  for (k, λ) in enumerate(stretches)
    λt, Fu = uniaxial(Pm, Am, λ, m.args)
    table_uni[k, j] = Pm(Fu, m.args...)[1, 1]
    table_lat[k, j] = λt
    _, Fe = equibiaxial(Pm, Am, λ, m.args)
    table_equi[k, j] = Pm(Fe, m.args...)[1, 1]
  end
  for (k, γ) in enumerate(shears)
    table_shear[k, j] = Pm(shearF(γ), m.args...)[1, 2]
  end
end

function print_table(title, xname, x, T, names)
  println("\n", title)
  @printf("%6s", xname)
  foreach(n -> @printf(" %19s", n), names)
  println()
  for k in eachindex(x)
    @printf("%6.2f", x[k])
    foreach(v -> @printf(" %19.6f", v), T[k, :])
    println()
  end
end
function write_csv(file, xname, x, T, names)
  open(joinpath(outdir, file), "w") do io
    println(io, join([xname; names], ","))
    for k in eachindex(x)
      println(io, join(string.([x[k]; T[k, :]]), ","))
    end
  end
end
colnames = [m.name for m in iso_models]
print_table("Uniaxial tension, lateral faces free: nominal stress P11 (kPa)", "λ", stretches, table_uni, colnames)
print_table("Equibiaxial tension, P33 = 0: nominal stress P11 (kPa)", "λ", stretches, table_equi, colnames)
print_table("Simple shear: nominal shear stress P12 (kPa)", "γ", shears, table_shear, colnames)
# Reading the tables: all the isotropic models share the initial slope (``3\mu`` in uniaxial tension,
# ``6\mu`` in equibiaxial tension and ``\mu`` in shear, with ``\mu = 1`` kPa) and differ at large stretch:
# Mooney–Rivlin softens in uniaxial tension and stiffens in equibiaxial tension, Yeoh, Gent and the
# eight-chain model stiffen. The fibre models are much stiffer along the fibre; in simple shear the fibre
# ``\boldsymbol N = \boldsymbol e_1`` is not stretched (``\boldsymbol F\boldsymbol e_1 = \boldsymbol e_1``), so
# `HGO_1Fiber` adds nothing, whereas `TransverseIsotropy3D` does, through ``I_5``.
write_csv("uniaxial_P11.csv", "stretch", stretches, table_uni, colnames)
write_csv("uniaxial_lateral_stretch.csv", "stretch", stretches, table_lat, colnames)
write_csv("equibiaxial_P11.csv", "stretch", stretches, table_equi, colnames)
write_csv("simple_shear_P12.csv", "gamma", shears, table_shear, colnames)

# ## Checks
# **(a) Closed forms.** Incompressible tests: the model is evaluated at the isochoric ``\boldsymbol F`` and the
# pressure ``p`` is eliminated with the traction-free lateral faces. The derivatives ``W_1``, ``W_2`` of the
# isochoric part of each model are written by hand (neo-Hookean ``W_1 = \mu/2``; Mooney–Rivlin
# ``W_1 = \mu_1/2``, ``W_2 = \mu_2/2``; Yeoh ``W_1 = C_{10} + 2C_{20}(I_1-3) + 3C_{30}(I_1-3)^2``; Gent
# ``W_1 = \mu/(2g)``).
function incompressible_uniaxial(P, λ)
  s = 1 / sqrt(λ)
  P̂ = P(diagF(λ, s, s))
  p = P̂[2, 2] * s                                              # P22 = P̂22 - p/s = 0
  P̂[1, 1] - p / λ
end
function incompressible_equibiaxial(P, λ)
  s = 1 / λ^2
  P̂ = P(diagF(λ, λ, s))
  p = P̂[3, 3] * s
  P̂[1, 1] - p / λ
end
incompressible_shear(P, γ) = P(shearF(γ))[1, 2]                # F⁻ᵀ has no 12 component: p does not enter

closed_forms = [
  ("NeoHookean3D",   NeoHookean3D(λ=100.0, μ=1.0),                    (I1, I2) -> (0.5, 0.0)),
  ("MooneyRivlin3D", MooneyRivlin3D(λ=100.0, μ1=0.7, μ2=0.3),         (I1, I2) -> (0.35, 0.15)),
  ("Yeoh3D",         Yeoh3D(λ=100.0, C10=0.5, C20=0.02, C30=0.001),    (I1, I2) -> (0.5 + 0.04 * (I1 - 3) + 0.003 * (I1 - 3)^2, 0.0)),
  ("Gent3D (user)",  Gent3D(μ=1.0, Jm=20.0, λ=100.0),                  (I1, I2) -> (0.5 / (1 - (I1 - 3) / 20.0), 0.0)),
]
max_err_a = 0.0
for (name, model, W) in closed_forms
  _, Pm, _ = model()
  err = 0.0
  for λ in stretches[2:end]
    W1, W2 = W(λ^2 + 2 / λ, 2λ + 1 / λ^2)
    Pex = 2 * (λ - λ^-2) * (W1 + W2 / λ)
    err = max(err, abs(incompressible_uniaxial(Pm, λ) - Pex) / abs(Pex))
    W1, W2 = W(2λ^2 + λ^-4, λ^4 + 2 / λ^2)
    Pex = 2 * (λ - λ^-5) * (W1 + λ^2 * W2)
    err = max(err, abs(incompressible_equibiaxial(Pm, λ) - Pex) / abs(Pex))
  end
  for γ in shears[2:end]
    W1, W2 = W(3 + γ^2, 3 + γ^2)
    Pex = 2γ * (W1 + W2)
    err = max(err, abs(incompressible_shear(Pm, γ) - Pex) / abs(Pex))
  end
  @printf("closed forms, %-15s max relative error %.2e\n", name, err)
  global max_err_a = max(max_err_a, err)
end
@assert max_err_a < 1e-10 "incompressible closed forms not reproduced (max error $max_err_a)"

# The compressible neo-Hookean in uniaxial tension also has a closed form (tutorial 02):
# ``J`` is the positive root of ``\lambda J^2 - (\lambda - \mu/\lambda_1)J - \mu = 0`` and
# ``P_{11} = \mu(\lambda_1 - J/\lambda_1^2)``. The nearly incompressible models are close to the
# incompressible curve: the difference is of order ``\mu/\lambda``.
J_nh(λ₁; λ=100.0, μ=1.0) = ((λ - μ / λ₁) + sqrt((λ - μ / λ₁)^2 + 4λ * μ)) / (2λ)
P11_nh(λ₁; μ=1.0) = μ * (λ₁ - J_nh(λ₁) / λ₁^2)
err_nh = maximum(abs.(table_uni[2:end, 1] .- P11_nh.(stretches[2:end])) ./ P11_nh.(stretches[2:end]))
println("compressible neo-Hookean, uniaxial: max relative error $(@sprintf("%.2e", err_nh))")
@assert err_nh < 1e-10 "compressible neo-Hookean uniaxial curve away from the closed form"
gap = maximum(abs.(table_uni[2:end, 1] .- 1.0 .* (stretches[2:end] .- stretches[2:end] .^ -2)) ./ table_uni[2:end, 1])
println("compressible (λ = 100 μ) vs incompressible neo-Hookean, uniaxial: max relative difference $(@sprintf("%.2e", gap))")
@assert gap < 0.05

# **(b) Invariants of every model.** At three states (small strain, large strain, compression), the stress
# must be the derivative of the energy and the tangent the derivative of the stress. Both are checked
# against central finite differences with Richardson extrapolation along the nine directions
# ``\boldsymbol e_i\otimes\boldsymbol e_j`` (relative error of the whole Jacobian, tolerance 1e-6). Also:
# the tangent has the major symmetry of a second derivative; the energy is objective,
# ``\Psi(\boldsymbol Q\boldsymbol F) = \Psi(\boldsymbol F)`` and ``\boldsymbol P(\boldsymbol Q\boldsymbol F) =
# \boldsymbol Q\boldsymbol P(\boldsymbol F)`` for a rotation ``\boldsymbol Q`` (the fibres are referential);
# and the reference state is free of energy and stress.
const Fs = (small=one(TensorValue{3,3,Float64}) + 1e-3 * TensorValue(1.0:9.0...),
            large=TensorValue(1.3, 0.1, 0.0, 0.2, 0.9, -0.1, 0.0, 0.15, 1.1),
            compression=TensorValue(0.5, 0.02, 0.0, 0.01, 0.6, 0.0, 0.0, 0.03, 0.67))
const Ebasis = ntuple(m -> TensorValue(ntuple(n -> n == m ? 1.0 : 0.0, 9)), 9)
const Q = let a = VectorValue(1.0, 2.0, 3.0) / sqrt(14.0), θ = 0.7     # rotation (Rodrigues)
  K = TensorValue(0.0, a[3], -a[2], -a[3], 0.0, a[1], a[2], -a[1], 0.0)
  one(K) + sin(θ) * K + (1 - cos(θ)) * (K ⋅ K)
end

function fd(f, F, dF; h=1e-4)                                  # Richardson-extrapolated central difference
  d(δ) = (f(F + δ * dF) - f(F - δ * dF)) / (2δ)
  (4 * d(h / 2) - d(h)) / 3
end
function derivative_errors(Ψ, P, A, F, args)
  eP = eA = nP = nA = 0.0
  for E in Ebasis
    dΨ = fd(G -> Ψ(G, args...), F, E)
    eP += (P(F, args...) ⊙ E - dΨ)^2
    nP += dΨ^2
    dP = fd(G -> P(G, args...), F, E)
    eA += norm(A(F, args...) ⋅² E - dP)^2
    nA += norm(dP)^2
  end
  sqrt(eP / nP), sqrt(eA / nA)
end
function symmetry_error(A, F, args)
  B, C = TensorValue(1.0:9.0...), TensorValue(0.3, -1.0, 2.0, 0.5, 0.1, -0.4, 1.5, 0.2, -0.6)
  AF = A(F, args...)
  abs(C ⊙ (AF ⋅² B) - B ⊙ (AF ⋅² C)) / abs(C ⊙ (AF ⋅² B))
end

println("\nmodel                 P vs dΨ/dF   A vs dP/dF   A symmetry   objectivity  Ψ(I), |P(I)|")
for m in models
  Ψm, Pm, Am = m.model()
  eP = eA = es = eo = 0.0
  for (state, Fm) in pairs(Fs)
    a, b = derivative_errors(Ψm, Pm, Am, Fm, m.args)
    eP, eA = max(eP, a), max(eA, b)
    es = max(es, symmetry_error(Am, Fm, m.args))
    state == :small && continue                                # Ψ ≈ 1e-6: relative round-off too large
    eo = max(eo, abs(Ψm(Q ⋅ Fm, m.args...) - Ψm(Fm, m.args...)) / abs(Ψm(Fm, m.args...)),
                 norm(Pm(Q ⋅ Fm, m.args...) - Q ⋅ Pm(Fm, m.args...)) / norm(Pm(Fm, m.args...)))
  end
  Id = one(TensorValue{3,3,Float64})
  rest = max(abs(Ψm(Id, m.args...)), norm(Pm(Id, m.args...)))
  @printf("%-20s  %.2e     %.2e     %.2e     %.2e     %.2e\n", m.name, eP, eA, es, eo, rest)
  @assert eP < 1e-6 "$(m.name): stress is not the derivative of the energy ($eP)"
  @assert eA < 1e-6 "$(m.name): tangent is not the derivative of the stress ($eA)"
  @assert es < 1e-12 "$(m.name): tangent without major symmetry ($es)"
  @assert eo < 1e-12 "$(m.name): not objective ($eo)"
  @assert rest < 1e-12 "$(m.name): energy or stress at F = I ($rest)"
end

# The user composition `FibreReinforced` and the library's `+` are the same material:
let (Ψa, Pa, Aa) = models[end-1].model(), (Ψb, Pb, Ab) = models[end].model(), F = Fs.large
  @assert Ψa(F, N4...) ≈ Ψb(F, N4...) && Pa(F, N4...) ≈ Pb(F, N4...) && Aa(F, N4...) ≈ Ab(F, N4...)
end

# **(c) Regression values** (HyperFEM 0.3.0, HyperFEMKernels 0.2.6, Gridap 0.20.8, Julia 1.13.0, macOS arm64):
# the nominal stress at ``\lambda = 2`` in uniaxial tension with free lateral faces. Point arithmetic without
# BLAS: the tolerance is tighter than in the FE tutorials.
P11_ref = Dict("NeoHookean3D" => 1.748762315869, "MooneyRivlin3D" => 1.487635363752,
               "Yeoh3D" => 2.071539451067, "EightChain+Vol" => 1.844431004853,
               "IncompNeoHookean3D" => 1.731210061009, "Gent3D (user)" => 1.943697469481,
               "NH+TransvIso" => 3.705533075008, "NH+HGO_1Fiber" => 38.046647102346)
for (j, m) in enumerate(iso_models)
  @printf("%-20s  P11(λ = 2) = %.12f kPa   λt = %.12f\n", m.name, table_uni[end, j], table_lat[end, j])
  @assert isapprox(table_uni[end, j], P11_ref[m.name]; rtol=1e-8) "$(m.name): regression value"
end

# ## Efficiency
# The FE forms call these functions at every quadrature point (`∂Ψ∂F ∘ (F ∘ ∇(u)', args...)`), and the
# compiled backend of HyperFEMKernels captures the forms, closures included, in type-specialised loops.
# That is fast because a model is an immutable struct of numbers and its functions are closures over those
# numbers: `isbits` values that the compiler inlines, with no heap allocation per evaluation. The check below
# measures the allocations of one stress and one tangent evaluation after compilation (in a function, so that
# the measurement does not include the dynamic call from global scope) and the time per evaluation. Both are
# printed, not asserted: whether a closure is inlined depends on the compiler's heuristics, which change
# between Julia versions and platforms (the table below was measured with Julia 1.13 on macOS arm64), so a
# zero-allocation assertion could fail on another machine for reasons that have nothing to do with the
# model. Composite models (`+`) and `HGO_4Fibers`, whose parameters are `Vector`s, are not `isbits`
# structs, but their stress functions still do not allocate, with one exception (measured).
#
# **The exception: `+` with four fibre families.** In HyperFEM 0.3.0, `nh + HGO_4Fibers(...)` allocates at
# every evaluation: 720 bytes per stress and 1920 bytes per tangent, and its tangent takes about twice as long
# as that of the same material written as `FibreReinforced` (663 ns against 327 ns in our measurement). This is
# a known issue of the library, and a fix is pending in HyperFEM; the model is correct (it passes every check
# above) and remains a valid choice.
# `+` builds `(F, N...) -> Ψ1(F) + Ψ2(F, N...)`, and Julia does not specialise a function on a `Vararg`
# argument that it only passes on to another function (Julia manual, Performance tips, «Be aware of when
# Julia avoids specializing»): the call with four fibres goes through a generic method. With one fibre the
# compiler still inlines it. The lesson for your own models is in `FibreReinforced` above: declaring
# `N::Vararg{Any,K}` forces the specialisation, with the same energy and stresses and no allocation. Writing
# the sum of the two models with explicit arguments has the same effect.
function allocations(f, F, args)
  f(F, args...)
  @allocated f(F, args...)
end
function ns_per_eval(f, F, args; n=200_000)
  s = 0.0
  f(F, args...)
  t = @elapsed for i in 1:n
    s += first(Tuple(f(F + (1e-12 * i) * one(F), args...)))
  end
  1e9 * t / n, s
end
println("\nmodel                 isbits(model)  isbits(P)  alloc P  alloc A   ns/P    ns/A")
for m in models
  _, Pm, Am = m.model()
  aP, aA = allocations(Pm, Fs.large, m.args), allocations(Am, Fs.large, m.args)
  tP, _ = ns_per_eval(Pm, Fs.large, m.args)
  tA, _ = ns_per_eval(Am, Fs.large, m.args)
  @printf("%-20s  %-13s  %-9s  %7d  %7d  %6.1f  %6.1f\n", m.name, isbits(m.model), isbits(Pm), aP, aA, tP, tA)
end

# ## Going further
# - `SecondPiola(model)` gives ``\Psi``, ``\boldsymbol S`` and ``\partial\boldsymbol S/\partial\boldsymbol C``
#   as functions of ``\boldsymbol C = \boldsymbol F^T\boldsymbol F`` for the models that implement it
#   (`IsochoricNeoHookean3D` in 0.3.0).
# - Coupled models follow the same pattern with more arguments: `ElectroMechModel` returns derivatives with
#   respect to ``(\boldsymbol F, \boldsymbol E)`` (tutorials 04 and 05), `MagnetoMechModel` with respect to
#   ``(\boldsymbol F, \boldsymbol{\mathcal H})`` (tutorials 08 and 09), and the viscous models take the previous
#   deformation and an internal variable (tutorial 16).
# - Put the user model in a simulation: replace `NeoHookean3D` by `Gent3D(μ=1.0, Jm=20.0, λ=100.0)` in
#   tutorial 02; for a homogeneous bar the force–stretch curve is the uniaxial column above.
# - The derivative checks of the library itself are in HyperFEM's
#   `test/ConstitutiveModelsTests/DerivativeConsistencyTests.jl`.
