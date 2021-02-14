import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import polytope
import numpy as np

def calculate_cable_length(wTb,wTp):
    """

    :param wTb: Transformation matrix of base attachment point in world
    :param wTp: Transformation matrix of platform attachment point in world
    :return: normalized_cable_vectors,cable_vectors,cable_lengths
    """
    cable_lengths=[]
    cable_vectors=[]
    normalized_cable_vectors=[]
    for pulley in wTb:
        cable_vec=(wTp-pulley)
        cable_vectors.append(cable_vec)
        cable_lengths.append(np.linalg.norm((cable_vec)))
        normalized_cable_vectors.append(cable_vec/np.linalg.norm((cable_vec)))
    return normalized_cable_vectors,cable_vectors,cable_lengths

def wrench_matrix(normalized_cable_vectors):
    """
    :param normalized_cable_vectors
    :return: wrench matrix
    """
    W=np.zeros([3,4])
    for i in range(len(normalized_cable_vectors)):
        W[:,i]=normalized_cable_vectors[i]
    return W

def tension_space_polytope(t_min,t_max):
    """
    :param t_min: minimum allowable tension value
    :param t_max: maxmimum allowable tension value
    :return: tension space polytope
    """
    A_desired = np.vstack((np.eye(4, 4), -1 * np.eye(4, 4)))
    B_desired=np.vstack((t_max*np.ones([4,1]), -t_min*np.ones([4,1])))
    tension_space_Hrep = polytope.Polytope(A_desired, B_desired)
    tension_space_Vrep = polytope.extreme(tension_space_Hrep)
    return tension_space_Vrep


def get_Cartesian_polytope(W,tension_space_Vrep):
    """
    :param W: Wrench matrix
    :param tension_space_Vrep: tension space poltope vertices
    :return: Wrench space vertices
    """

    # for each vertex in the tension space project to Wrench space using the wrench matrix
    Pv=np.zeros([np.shape(tension_space_Vrep)[0],np.shape(W)[0]])

    for row,i in zip(tension_space_Vrep,range(np.shape(tension_space_Vrep)[0])):
        Pv[i,:]=np.matmul(W,row)
    hull = ConvexHull(Pv, qhull_options='Qs QJ')
    return hull




def point_in_hull(point, hull, tolerance=1e-12):
    #    https: // stackoverflow.com / questions / 16750618 / whats - an - efficient - way - to - find - if -a - point - lies - in -the - convex - hull - of - a - point - cl / 42165596  # 42165596
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

def plot_polytope_3d(hull):
    """
    :param hull:  Wrench space convex hull
    :return: the plot and figure objects
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(hull.points[hull.vertices, 0],
            hull.points[hull.vertices, 1],
            hull.points[hull.vertices, 2], 'ko', markersize=4)

    s = ax.plot_trisurf(hull.points[:, 0], hull.points[:, 1], hull.points[:, 2], triangles=hull.simplices,
                        color='red', alpha=0.2, edgecolor='k')
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')

    return ax

# Simple workspace analysis for 4 cable suspended Cartesian CDPR robot
if __name__ == '__main__':
    # Base pulley points let's say a van of dimensions 3x2x
    x_max=3.5
    x_min=0.0
    y_max=2.0
    y_min=0.0
    z_max=2.2
    z_min=0.0
    A1 = np.array([x_min,y_min,z_max])
    A2 = np.array([x_min, y_max, z_max])
    A3 = np.array([x_max, y_min, z_max])
    A4 = np.array([x_max, y_max, z_max])
    wTb = np.array([A1, A2, A3, A4])  # The locations of the attachment points
    minimum_wrench = np.array([0.5, 0.5, -3.0*9.81]) # lets say platform must support it's weight of 2kg

    # max and min cable limits
    t_max = 20 # maximum cable tension found form motors
    t_min = 2 # need to ensure cable tension is above zero
    tension_space_Vrep = tension_space_polytope(t_min, t_max) # this defines a 'box' in tension space that has all feasible tensions


    # An example
    wTp = [0.5, 0.5, 0.5]  # Where the platform is at the moment
    normalized_cable_vectors, cable_vectors, cable_lengths = calculate_cable_length(wTb, wTp)
    W = wrench_matrix(normalized_cable_vectors)     # Wt + we=0 https://hal.archives-ouvertes.fr/hal-01941785/document
    AWS=get_Cartesian_polytope(W,tension_space_Vrep)
    # This returns plot all the forces that the platform can support in it's current position
    # Uncomment to plot
    # ax = plot_polytope_3d(AWS)
    print("Is the platform able to support itself?",point_in_hull(minimum_wrench, AWS))
    # Uncomment to plot
    # ax.plot([minimum_wrench[0]], [minimum_wrench[1]], [minimum_wrench[2]], markerfacecolor = 'k', markeredgecolor = 'k', marker = 'o', markersize = 10)

    # Plot shows wrench space capacities and the minimum wrench
    #plt.show()

    # =================== Plot the workspace ========================= #
    # Check if force is within convex hull
    tolerance=0.05 # can't have zero cable lentgh
    x_space = np.linspace(x_min+tolerance,x_max-tolerance,10)
    y_space = np.linspace(y_min+tolerance, y_max-tolerance, 10)
    z_space = np.linspace(z_min+tolerance, z_max-tolerance, 20)
    feasible_points = np.empty((0, 3), float)
    for x in x_space:
        for y in y_space:
            for z in z_space:
                wTp = np.array([x, y, z])  # Where the platform is at the moment
                normalized_cable_vectors, cable_vectors, cable_lengths = calculate_cable_length(wTb, wTp)
                W = wrench_matrix(normalized_cable_vectors)
                AWS = get_Cartesian_polytope(W, tension_space_Vrep)
                if( point_in_hull(minimum_wrench, AWS) ):
                    feasible_points=np.vstack([feasible_points,wTp])

    print(np.shape(feasible_points))

    # wrkspace_hull = ConvexHull(feasible_points, qhull_options='Qs QJ')
    # ax2=plot_polytope_3d(wrkspace_hull)

    A1 = np.array([x_min,y_min,z_max])
    A2 = np.array([x_min, y_max, z_max])
    A3 = np.array([x_max, y_min, z_max])
    A4 = np.array([x_max, y_max, z_max])

    fig = plt.figure()
    ax2 = fig.gca(projection='3d')
    ax2.plot([A1[0],A2[0],A4[0],A3[0],A1[0]],
              [A1[1],A2[1],A4[1],A3[1],A1[1]],
              [A1[2],A2[2],A4[2],A3[2],A1[2]], linewidth=5,markerfacecolor = 'b', markeredgecolor = 'k', marker = 'o', markersize = 10 )

    my_cmap = plt.get_cmap('hsv')
    ax2.scatter3D(feasible_points[:,0],feasible_points[:,1],feasible_points[:,2],
                  marker='x',
                  c = (feasible_points[:,0]+feasible_points[:,1]+feasible_points[:,2]),
                  cmap = my_cmap)
    plt.title("Workspace analysis")
    ax2.set_xlabel('X-axis', fontweight='bold')
    ax2.set_ylabel('Y-axis', fontweight='bold')
    ax2.set_zlabel('Z-axis', fontweight='bold')


    plt.show()
    # Now map the space