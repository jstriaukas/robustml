using robustml
using Documenter

DocMeta.setdocmeta!(robustml, :DocTestSetup, :(using robustml); recursive=true)

makedocs(;
    modules=[robustml],
    authors="Jonas Striaukas, Seyed Abbas Jazaeri",
    repo="https://github.com/sajazaerica/robustml.jl/blob/{commit}{path}#{line}",
    sitename="robustml.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sajazaerica.github.io/robustml.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sajazaerica/robustml.jl",
)
