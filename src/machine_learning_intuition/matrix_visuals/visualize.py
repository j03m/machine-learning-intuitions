import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_cube(ax, center, size, color):
    # Generate the vertices of the cube
    r = [-size / 2, size / 2]
    vertices = np.array([[x, y, z] for x in r for y in r for z in r])
    vertices[:, 0] += center[0]
    vertices[:, 1] += center[1]
    vertices[:, 2] += center[2]

    # Generate the edges of the cube
    edges = [[vertices[i], vertices[j]] for i in range(len(vertices)) for j in range(len(vertices)) if np.linalg.norm(vertices[i] - vertices[j]) == size]

    # Plot the edges
    for edge in edges:
        ax.plot3D(*zip(*edge), color=color)

def visualize_matrix_dot_product(A, B):
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions for dot product"

    # Calculate the dot product
    C = np.dot(A, B)

    # Create a 3D subplot
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw matrix A
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                draw_cube(ax, center=(i, j, k), size=A[i, j, k], color='red')

    # Draw matrix B with an offset
    offset = np.max(A.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            for k in range(B.shape[2]):
                draw_cube(ax, center=(i + offset, j, k), size=B[i, j, k], color='green')

    # Draw matrix C with another offset
    offset += np.max(B.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            for k in range(C.shape[2]):
                draw_cube(ax, center=(i + 2 * offset, j, k), size=C[i, j, k], color='blue')

    # Setting the plot limits
    max_dim = max(A.shape[0], B.shape[1], C.shape[0], C.shape[1]) + 2 * offset
    ax.set_xlim([0, max_dim])
    ax.set_ylim([0, max_dim])
    ax.set_zlim([0, max_dim])

    plt.show()

# Example matrices (replace these with actual 3D matrices)
A = np.random.rand(3, 3, 3)
B = np.random.rand(3, 3, 3)

visualize_matrix_dot_product(A, B)
