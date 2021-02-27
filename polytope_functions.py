import numpy as np
import polytope
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull

def get_Cartesian_polytope(jacobian,joint_space_vrep):


    Pv=np.zeros([np.shape(joint_space_vrep)[0],np.shape(jacobian)[0]])

    for row,i in zip(joint_space_vrep,range(np.shape(joint_space_vrep)[0])):
        Pv[i,:]=np.matmul(jacobian,row)
    return Pv


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

def point_in_hull(point, hull, tolerance=1e-12):
    #    https: // stackoverflow.com / questions / 16750618 / whats - an - efficient - way - to - find - if -a - point - lies - in -the - convex - hull - of - a - point - cl / 42165596  # 42165596
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)