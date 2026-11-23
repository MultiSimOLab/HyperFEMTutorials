# Mesh of the DFG benchmark "flow around a cylinder" 2D (Schäfer & Turek 1996), used by tutorials 10 and 11.
#
#   julia --project=. data/models/turek.jl          # h = 0.02: writes data/models/turek2d.msh
#   julia --project=. data/models/turek.jl 0.01     # finer: writes data/models/turek2d_h0.01.msh
#
# Rectangle 2.2 × 0.41 with a circle of radius 0.05 centred at (0.2, 0.2); element size h on the channel and
# h/5 on the cylinder (linear triangles). Physical tags: "fluid", "inlet" (x = 0), "outlet" (x = 2.2),
# "wall" (y = 0 and y = 0.41), "cylinder"; the points of each curve are included, and the corners of the
# inlet and the outlet belong to the wall too. With h = 0.02 it gives the 7 402 triangles of
# HyperFEM.jl/test/models/turek2d.msh (same script as `benchmark/meshes/generate.jl turek 0.02` of
# HyperFEMKernels.jl; the node numbering may change with the version of Gmsh).

using GridapGmsh: gmsh

function turek(h::Real; file=joinpath(@__DIR__, h == 0.02 ? "turek2d.msh" : "turek2d_h$(h).msh"))
  gmsh.initialize()
  gmsh.option.setNumber("General.Terminal", 0)
  gmsh.model.add("turek")
  geo = gmsh.model.geo
  L, H, cx, cy, r = 2.2, 0.41, 0.2, 0.2, 0.05
  hc = h / 5
  p1 = geo.addPoint(0, 0, 0, h); p2 = geo.addPoint(L, 0, 0, h); p3 = geo.addPoint(L, H, 0, h); p4 = geo.addPoint(0, H, 0, h)
  bottom = geo.addLine(p1, p2); outlet = geo.addLine(p2, p3); top = geo.addLine(p3, p4); inlet = geo.addLine(p4, p1)
  c0 = geo.addPoint(cx, cy, 0, hc)
  cp = [geo.addPoint(cx + r * cos(θ), cy + r * sin(θ), 0, hc) for θ in (0, π / 2, π, 3π / 2)]
  arcs = [geo.addCircleArc(cp[i], c0, cp[mod1(i + 1, 4)]) for i in 1:4]
  s = geo.addPlaneSurface([geo.addCurveLoop([bottom, outlet, top, inlet]), geo.addCurveLoop(arcs)])
  geo.synchronize()
  gmsh.model.addPhysicalGroup(2, [s], 1, "fluid")
  gmsh.model.addPhysicalGroup(1, [inlet], 2, "inlet"); gmsh.model.addPhysicalGroup(0, [p1, p4], 2, "inlet")
  gmsh.model.addPhysicalGroup(1, [outlet], 3, "outlet"); gmsh.model.addPhysicalGroup(0, [p2, p3], 3, "outlet")
  gmsh.model.addPhysicalGroup(1, [bottom, top], 4, "wall"); gmsh.model.addPhysicalGroup(0, [p1, p2, p3, p4], 4, "wall")
  gmsh.model.addPhysicalGroup(1, arcs, 5, "cylinder"); gmsh.model.addPhysicalGroup(0, cp, 5, "cylinder")
  gmsh.model.mesh.generate(2)
  gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
  _, tags, _ = gmsh.model.mesh.getElements(2)
  println("triangles: ", sum(length, tags), ", nodes: ", length(gmsh.model.mesh.getNodes()[1]))
  gmsh.write(file)
  gmsh.finalize()
  println("written $file")
  file
end

if abspath(PROGRAM_FILE) == @__FILE__
  turek(isempty(ARGS) ? 0.02 : parse(Float64, ARGS[1]))
end
