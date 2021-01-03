import numpy as np
import polytope
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull


def plot_polytope_3d(poly):
    V = polytope.extreme(poly)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    hull = ConvexHull(V, qhull_options='Qs QJ')
    ax.plot(hull.points[hull.vertices, 0],
            hull.points[hull.vertices, 1],
            hull.points[hull.vertices, 2], 'ko', markersize=4)

    s = ax.plot_trisurf(hull.points[:, 0], hull.points[:, 1], hull.points[:, 2], triangles=hull.simplices,
                        color='red', alpha=0.2, edgecolor='k')

    plt.show()
    return ax