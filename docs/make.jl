using Documenter
using Literate
using CFsOnSphere

# Headless GR for figure generation in tutorials (no display needed on CI).
ENV["GKSwstype"] = "100"

const LITERATE_DIR = joinpath(@__DIR__, "literate")
const TUTORIAL_OUT = joinpath(@__DIR__, "src", "tutorials")

# Regenerate the tutorial Markdown from the Literate sources on every build.
isdir(TUTORIAL_OUT) && rm(TUTORIAL_OUT; recursive=true)
mkpath(TUTORIAL_OUT)
for src in sort(filter(f -> endswith(f, ".jl"), readdir(LITERATE_DIR; join=true)))
    Literate.markdown(src, TUTORIAL_OUT; documenter=true)
end

tutorial_md(name) = joinpath("tutorials", name)

makedocs(
    sitename = "CFsOnSphere.jl",
    authors  = "Mytraya Gattu and contributors",
    modules  = [CFsOnSphere],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://mytraya-gattu.github.io/CompositeFermions",
        assets     = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Physics background" => "physics.md",
        "Tutorials" => [
            "1 — Ground state & density"          => tutorial_md("01_ground_state.md"),
            "2 — Quasiholes & quasiparticles"     => tutorial_md("02_excitations.md"),
            "3 — Higher fillings (outer Jastrow)" => tutorial_md("03_higher_fillings.md"),
            "4 — Unprojected + Sherman–Morrison"  => tutorial_md("04_unprojected_fast.md"),
            "5 — Parton states"                   => tutorial_md("05_partons.md"),
            "6 — Observables & energies"          => tutorial_md("06_observables.md"),
            "7 — Under the hood"                  => tutorial_md("07_under_the_hood.md"),
        ],
        "API reference"  => "api.md",
        "Architecture"   => "architecture.md",
        "Validation"     => "validation.md",
        "Theory & citation" => "theory.md",
    ],
    warnonly = false,
)

deploydocs(
    repo = "github.com/mytraya-gattu/CompositeFermions.git",
    devbranch = "main",
    push_preview = true,
)
