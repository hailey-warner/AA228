using Graphs, GraphPlot, Compose, Cairo, Fontconfig
using Printf
using DataFrames, CSV
using Random, LinearAlgebra, SpecialFunctions

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

function read_gph(filename)
"""
    read_gph(filename)

Takes a csv file path and returns a data matrix, variables vector, instantiations vector, and Dict of column indices to variables.
"""
    df = CSV.read(filename, DataFrame)
    D = transpose(Matrix(df))
    vars = names(df)
    ranges = [maximum(df[!, var]) for var in vars]
    idx2names = Dict(i => var for (i, var) in enumerate(vars))
    return D, vars, ranges, idx2names
end

function make_random_graph(vars)
"""
    make_random_graph(vars)

Takes variable name vector (length N) and returns a randomly generated DAG with 0 to N(N-1)/2 edges.
"""
    G = SimpleDiGraph(length(vars))
    num_edges = rand(0:(length(vars)*(length(vars)-1))/2) # (N)(N-1)/2 edges => fully-connected graph
    for i in 1:num_edges
        v1 = Int(rand(1:num_edges))
        v2 = Int(rand(1:num_edges))
        add_edge!(G, v1, v2)
        if is_cyclic(G)
            rem_edge!(G, v1, v2)
        end
    end
    return G
end

##################

function sub2ind(siz, x)
"""
    sub2ind(siz, x)

See Algorithms for Decision Making, p. 75
"""
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

function statistics(ranges, G, D) 
"""
    statistics(ranges, G, D)

See Algorithms for Decision Making, p. 75
"""
    n = size(D, 1)
    q = [prod([ranges[j] for j in inneighbors(G,i)]) for i in 1:n]
    M = [zeros(q[i], ranges[i]) for i in 1:n]
    for col in eachcol(D)
        for i in 1:n
            k = col[i]
            parents = inneighbors(G,i)
            j=1
            if !isempty(parents)
                j = sub2ind(ranges[parents], col[parents])
            end 
            M[i][j,k] += 1.0
        end
    end
    return M
end

function prior(vars, ranges, G)
"""
    prior(vars, ranges, G)

See Algorithms for Decision Making, p. 81
"""
    n = length(vars)
    q = [prod([ranges[j] for j in inneighbors(G,i)]) for i in 1:n]
    return [ones(q[i], ranges[i]) for i in 1:n]
end

function bayesian_score_component(M, a)
"""
    bayesian_score_component(M, a)

See Algorithms for Decision Making, p. 98
"""
    p = sum(loggamma.(a + M))
    p -= sum(loggamma.(a))
    p += sum(loggamma.(sum(a,dims=2)))
    p -= sum(loggamma.(sum(a,dims=2) + sum(M,dims=2)))
    return p
end

function bayesian_score(vars, ranges, G, D)
"""
    bayesian_score(vars, ranges, G, D)

See Algorithms for Decision Making, p. 75
"""
    n = length(vars)
    M = statistics(ranges, G, D)
    a = prior(vars, ranges, G)
    return sum(bayesian_score_component(M[i], a[i]) for i in 1:n)
end

function rand_graph_neighbor(G)
"""
    rand_graph_neighbor(G)

See Algorithms for Decision Making, p. 102
"""
    n = nv(G)
    i = rand(1:n)
    j = mod1(i + rand(2:n)-1, n)
    G_new = copy(G)
    has_edge(G, i, j) ? rem_edge!(G_new, i, j) : add_edge!(G_new, i, j)
    return G_new
end

#####################

function compute(infile, outfile)
"""
    compute(infile, outfile)

See Algorithms for Decision Making, p. 101
"""
    k_max = 1000
    D, vars, ranges, idx2names = read_gph(infile)
    G = make_random_graph(vars)
    y = bayesian_score(vars, ranges, G, D)
    for k in 1:k_max
        G_new = rand_graph_neighbor(G)
        y_new = is_cyclic(G_new) ? -Inf : bayesian_score(vars, ranges, G_new, D)
        if y_new > y
            y, G = y_new, G_new
        end
    end
    print(y)
    write_gph(G, idx2names, outfile)
    p = gplot(G; nodelabel=vars)
    draw(PDF(replace(outfile, r"\.gph$" => "")*".pdf", 16cm, 16cm), p)
end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

@time begin
compute(inputfilename, outputfilename)
end