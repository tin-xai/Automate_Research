import numpy as np
import copy

def has_negative_cycle(graph):
    V = len(graph)
    for i in range(V):
        if graph[i][i] < 0:
            return True
    return False

def matrix_multiplication(graph):
    # Get the number of vertices in the graph
    V = len(graph)
    
    # Initialize the distance matrix with the same values as the graph
    dist = [row[:] for row in graph]
    dist_copied = copy.deepcopy(dist)

    for i in range(V):
        choosen_row = dist[i]
        for k in range(V):
            choosen_column = []
            for row in dist:
                choosen_column.append(row[k])

            min_values = []
            for c_val, r_val in zip(choosen_column, choosen_row):
                if dist[i][k] > c_val + r_val:
                    min_values.append(c_val + r_val)
            if len(min_values) != 0:
                dist_copied[i][k] = min(min_values)
    
    return dist_copied

# Example usage:
inf = float('inf')
graph = [
    [0, 3, 5, 2],
    [1, 0, -2, 4],
    [-3, 5, 0, 6],
    [1, 2, -1, 0]
]

graph = [
    [0, 5, 6, inf, inf, 5,7,2],
    [4,0,4,inf,9,inf,inf,8],
    [8,9,0,10,8,inf,2,inf],
    [6,8,7,0,4,8,7,6],
    [inf,inf,8,4,0,5,7,inf],
    [9,7,inf,11,6,0,4,8],
    [inf,inf,10,3,9,2,0,inf],
    [5,6,3,inf,inf,inf,8,0]
]

step = 0
while not has_negative_cycle(graph):
    print(f"Step {step}")
    graph = matrix_multiplication(graph)
    print(np.array(graph))
    step += 1
    if step == 1:
        break
