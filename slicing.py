import numpy as np
import polytope
import robot_functions
import polytope_functions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
from scipy.spatial import Delaunay, ConvexHull

def plot_it(desired_twist,c,ax):
    VV = polytope.extreme(desired_twist)

    # ax = fig.gca(projection='3d')

    #ax = fig.gca()
    hull = ConvexHull(VV, qhull_options='Qs QJ')
    ax.plot(hull.points[hull.vertices, 0],
            hull.points[hull.vertices, 1],
            hull.points[hull.vertices, 2], 'ko', markersize=4)

    s = ax.plot_trisurf(hull.points[:, 0], hull.points[:, 1], hull.points[:, 2], triangles=hull.simplices,
                        color=c, alpha=0.2, edgecolor='k')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    A_desired = np.vstack((np.random.rand(6, 3), -1 * np.eye(6, 3)))

    B_desired = np.random.rand(12,1)*2.0

    desired_twist = polytope.Polytope(A_desired, B_desired)

    V = polytope.extreme(desired_twist)
    print("V", V)

    A2_desired = np.vstack((np.eye(3, 3), -1 * np.eye(3, 3)))
    B2_desired = np.ones([6,])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_it(desired_twist,'red',ax)


    A2_desired = np.vstack((np.eye(3, 3), -1 * np.eye(3, 3)))
    B2_desired = np.ones([6,1])*5.0
    B2_desired[1]= 0.001
    B2_desired[4] = 0.001
    desired_twist2 = polytope.Polytope(A2_desired, B2_desired)
    V2 = polytope.extreme(desired_twist2)
    print("V2",V2)
    plot_it(desired_twist2,'blue',ax)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    A3_desired=np.vstack([A_desired,A2_desired])
    print(A3_desired)
    B3_desired = np.vstack([B_desired, B2_desired])
    desired_twist3 = polytope.Polytope(A3_desired, B3_desired)
    V3 = polytope.extreme(desired_twist3)
    plot_it(desired_twist3, 'green', ax2)
    plot_it(desired_twist3, 'green', ax)
    plt.show()
