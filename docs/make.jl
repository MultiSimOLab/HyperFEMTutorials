# Builds the documentation: one page and one Jupyter notebook per tutorial, generated with Literate from
# src/E<NN>_<Name>.jl (the code is not executed). Usage, from the repository root:
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#   julia --project=docs docs/make.jl              # HTML in docs/build
#   julia --project=docs docs/make.jl notebooks    # only the notebooks, in notebooks/ at the repository root
using Literate

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "src", "HyperFEMTutorials.jl"))
using .HyperFEMTutorials: TUTORIALS

const REPO = "https://github.com/MultiSimOLab/HyperFEMTutorials.jl"
const SITE = "https://multisimolab.github.io/HyperFEMTutorials.jl/dev"
const NBVIEWER = "https://nbviewer.org/urls/multisimolab.github.io/HyperFEMTutorials.jl/dev/notebooks"

available = filter(t -> isfile(joinpath(ROOT, "src", t.file)), TUTORIALS)
stem(t) = splitext(t.file)[1]

# Links under the title of every page (markdown only: `#md` lines are dropped from the notebooks)
function add_links(t)
  content -> replace(content, r"^(# # [^\n]*\n)"m =>
    SubstitutionString("\\1#md #\n#md # [Source]($REPO/blob/main/src/$(t.file)) · " *
                       "[Notebook]($SITE/notebooks/$(stem(t)).ipynb) · " *
                       "[nbviewer]($NBVIEWER/$(stem(t)).ipynb)\n"); count=1)
end

notebooks_only = "notebooks" in ARGS
nbdir = notebooks_only ? joinpath(ROOT, "notebooks") : joinpath(@__DIR__, "src", "notebooks")
for t in available
  file = joinpath(ROOT, "src", t.file)
  Literate.notebook(file, nbdir; name=stem(t), execute=false, credit=false)
  notebooks_only && continue
  Literate.markdown(file, joinpath(@__DIR__, "src", "tutorials"); name=stem(t), execute=false, credit=false,
                    documenter=true, codefence="```julia" => "```", preprocess=add_links(t))
end
notebooks_only && exit()

using Documenter

const GROUPS = [:basics => "Basics", :coupled => "Coupled problems", :contact => "Contact",
                :optimisation => "Optimisation", :fluids => "Fluids", :solvers => "Solvers",
                :dynamics => "Dynamics"]
page(t) = "$(lpad(t.number, 2, '0')) · $(t.title)" => joinpath("tutorials", stem(t) * ".md")
pages = Any["Home" => "index.md"]
for (g, label) in GROUPS
  entries = Any[page(t) for t in available if t.group === g]
  g === :optimisation && push!(entries, "07 · Topology optimisation (in preparation)" => "E07_Topology_optimization.md")
  isempty(entries) || push!(pages, label => entries)
end
push!(pages, "Running and testing" => "running.md")

makedocs(;
  sitename = "HyperFEM tutorials",
  authors = "MultiSimOLab",
  repo = Remotes.GitHub("MultiSimOLab", "HyperFEMTutorials.jl"),
  format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true", canonical = SITE,
                           edit_link = "main", size_threshold_warn = 400 * 2^10, size_threshold = 800 * 2^10,
                           mathengine = Documenter.MathJax3()),
  pages,
  warnonly = [:missing_docs, :cross_references],
  checkdocs = :none,
)

deploydocs(; repo = "github.com/MultiSimOLab/HyperFEMTutorials.jl.git", devbranch = "main", push_preview = false)
