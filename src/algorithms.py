
# get undirected graph from list of edges
def get_graph_from_edges(edges):
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
    return graph

# get directed graph from list of edges
def get_directed_graph_from_edges(edges):
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        graph[u].append(v)
    return graph

# get connected components in graph
def get_connected_components(graph):
    visited = set()
    components = []
    for node in graph:
        if node not in visited:
            component = set()
            component.add(node)
            stack = [node]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    if node in graph:
                        stack.extend(graph[node])
                    component.add(node)
            components.append(component)
    return components

# check if exists any cycle in graph
def is_cycle_graph(graph, component):
    n_edges = 0
    for node in component:
        if node in graph:
            n_edges += len(graph[node])
    return n_edges != (len(component) - 1)

def get_nodes_not_in_cycle(edges):
    graphs = get_graph_from_edges(edges)
    components = get_connected_components(graphs)
    directed_graphs = get_directed_graph_from_edges(edges)
    list_nodes = []
    for component in components:
        if not is_cycle_graph(directed_graphs, component):
            list_nodes.extend(component)
    return list_nodes

# edges = [
#     ('A', 'B'),
#     ('B', 'C'),
#     ('D', 'C'),
#     ('C', 'E'),
#     ('E', 'F'),
#     ('G', 'D'),
#     ('F', 'D'),
#     ('T', 'U'),
#     ('U', 'V'),
#     ('V', 'T'),
#     ('X', 'Y'),
#     ('Y', 'Z')
# ]

# print(get_nodes_not_in_cycle(edges))