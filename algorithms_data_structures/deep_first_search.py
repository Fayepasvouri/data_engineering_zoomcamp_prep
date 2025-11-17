# deep first search is a graph from smallest left to biggest right and HEAD vertex. it has vertices and edges and its Big O complexity is O(V + E)

# Simple implementation

def deep_first_search(graph, start, visited=None):
    
    if visited is None:
        visited = set()
    
    visited.add(start)
    
    for next_val in graph[start] - visited:
        deep_first_search(graph, next_val, visited) # dfs uses recursion
    print(visited)
    return visited

graph = {
    1: set([2, 3]),
    2: set([4, 5]),
    3: set([6, 7]),
    4: set([]),
    5: set([]),
    6: set([]),
    7: set([])
}

deep_first_search(graph, 1)
