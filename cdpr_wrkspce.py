import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
from scipy.spatial import Delaunay, ConvexHull
import polytope
import numpy as np
#from tf import transformations



def calculate_cable_length(wld_position_base_pts_, wld_pose_tool_, tool_position_plat_pts_):
    """

    :param wld_position_base_pts_: positions of base attachment point in world
    :param wld_pose_tool_: Transformation matrix of platform attachment point in world
    :param tool_position_plat_pts_: positions of base attachment point in tool frame
    :return: normalized_cable_vectors,cable_vectors,cable_lengths
    """

    cable_lengths_ = []
    cable_vectors_ = []
    normalized_cable_vectors_ = []
    p = wld_pose_tool_[0:3, 3]

    for a, r in zip(wld_position_base_pts_, tool_position_plat_pts_):
        # https://hal.archives-ouvertes.fr/hal-01941785/document eq. 1
        cable_vec_ = (a - p - np.matmul(wld_pose_tool_[0:3, 0:3], r))
        cable_vectors_.append(cable_vec_)
        cable_lengths_.append(np.linalg.norm(cable_vec_))
        normalized_cable_vectors_.append(cable_vec_ / np.linalg.norm(cable_vec_))
    return normalized_cable_vectors_, cable_vectors_, cable_lengths_


def wrench_matrix(wld_position_base_pts_, wld_pose_tool_, tool_position_plat_pts_):
    """

    :param wld_position_base_pts_: positions of base attachment point in world
    :param wld_pose_tool_: Transformation matrix of platform attachment point in world
    :param tool_position_plat_pts_: positions of base attachment point in tool frame
    :return: Wrench Matrix
    """

    p = wld_pose_tool_[0:3, 3]
    W = np.zeros([6, 8])
    for count, (a, r) in enumerate(zip(wld_position_base_pts_, tool_position_plat_pts_)):
        # https://hal.archives-ouvertes.fr/hal-01941785/document eq. 1
        world_position_plat_pts_ = np.matmul(wld_pose_tool_[0:3, 0:3], r)
        cable_vec_ = (a - p - world_position_plat_pts_)
        cable_vec_normalized = cable_vec_ / np.linalg.norm(cable_vec_)
        W[0:3, count] = cable_vec_normalized
        W[3:6, count] = np.cross(world_position_plat_pts_, cable_vec_normalized)
    return W

def sub_wrench_matrix(wld_position_base_pts_, wld_pose_tool_):
    """

    :param wld_position_base_pts_: positions of base attachment point in world
    :param wld_pose_tool_: Transformation matrix of platform attachment point in world
    :param tool_position_plat_pts_: positions of base attachment point in tool frame
    :return: Wrench Matrix
    """

    p = wld_pose_tool_[0:3, 3]
    W = np.zeros([3, 8])
    for count, a in enumerate(wld_position_base_pts_):
        # https://hal.archives-ouvertes.fr/hal-01941785/document eq. 1
        cable_vec_ = (a - p )
        cable_vec_normalized = cable_vec_ / np.linalg.norm(cable_vec_)
        W[0:3, count] = cable_vec_normalized
    return W

def tension_space_polytope(t_min_, t_max_, n):
    """
    :param n: number of cables
    :param t_min_: minimum allowable tension value
    :param t_max_: maxmimum allowable tension value
    :return: tension space polytope
    """
    A_desired = np.vstack((np.eye(n, n), -1 * np.eye(n, n)))
    B_desired = np.vstack((t_max_ * np.ones([n, 1]), -t_min_ * np.ones([n, 1])))
    tension_space_Hrep = polytope.Polytope(A_desired, B_desired)
    tension_space_Vrep = polytope.extreme(tension_space_Hrep)
    return tension_space_Vrep

def get_Cartesian_polytope2(W, tension_space_Vrep):
    """
    :param W: Wrench matrix
    :param tension_space_Vrep: tension space poltope vertices
    :return: Wrench space vertices
    """

    # for each vertex in the tension space project to Wrench space using the wrench matrix
    Pv = np.zeros([np.shape(tension_space_Vrep)[0], np.shape(W)[0]])

    for row, i in zip(tension_space_Vrep, range(np.shape(tension_space_Vrep)[0])):
        Pv[i, :] = np.matmul(W, row)
    polyhull = polytope.qhull(Pv)
    return polyhull

def get_Cartesian_polytope(W, tension_space_Vrep):
    """
    :param W: Wrench matrix
    :param tension_space_Vrep: tension space poltope vertices
    :return: Wrench space vertices
    """

    # for each vertex in the tension space project to Wrench space using the wrench matrix
    Pv = np.zeros([np.shape(tension_space_Vrep)[0], np.shape(W)[0]])

    for row, i in zip(tension_space_Vrep, range(np.shape(tension_space_Vrep)[0])):
        Pv[i, :] = np.matmul(W, row)
    hull = ConvexHull(Pv, qhull_options='Qs QJ')
    return hull

def point_in_hull2(point, polyhull, tolerance=1e-12):
    #    https: // stackoverflow.com / questions / 16750618 / whats - an - efficient - way - to - find - if -a - point - lies - in -the - convex - hull - of - a - point - cl / 42165596  # 42165596
    return all(
        (np.dot(eq, point) - polyhull.b[count] <= tolerance)
        for count, eq in enumerate(polyhull.A))

def point_in_hull(point, hull, tolerance=1e-7):
    #    https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to- find-if-a-point-lies - in -the - convex - hull - of - a - point - cl / 42165596  # 42165596
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


def get_sub_base_attachment_points(platform_origin_x,platform_origin_y,platform_origin_z,
        platform_width,platform_breadth):
    # Base pulley points let's say a van of dimensions 3x2x

    A1 = np.array([platform_origin_x, platform_origin_y, platform_origin_z])
    A2 = np.array([platform_origin_x, platform_origin_y+platform_breadth, platform_origin_z])
    A3 = np.array([platform_origin_x+platform_width, platform_origin_y+platform_breadth, platform_origin_z])
    A4 = np.array([platform_origin_x+platform_width, platform_origin_y, platform_origin_z])

    return np.array([A1, A2, A3, A4])

def get_base_attachment_points(x_max,y_max,z_max):
    # Base pulley points let's say a van of dimensions 3x2x


    x_min = 0.0
    x_delta = 0.2

    y_min = 0.0
    y_delta = 0.1

    z_min = 0.0
    A1 = np.array([x_min, y_min, z_max])
    A2 = np.array([x_min, y_max, z_max])
    A3 = np.array([x_max, y_min, z_max])
    A4 = np.array([x_max, y_max, z_max])

    A5 = np.array([x_min + x_delta, y_min + y_delta, z_max])
    A6 = np.array([x_min + x_delta, y_max - y_delta, z_max])
    A7 = np.array([x_max - x_delta, y_min + y_delta, z_max])
    A8 = np.array([x_max - x_delta, y_max - y_delta, z_max])
    return np.array([A1, A2, A3, A4, A5, A6, A7, A8])


def get_platform_attachment_points(x_width, y_width, z_width):
    # Base pulley points let's say a van of dimensions 3x2x

    A1 = np.array([0., 0., 0.])
    A2 = np.array([x_width, 0., z_width])
    A3 = np.array([x_width, 0., 0.0])
    A4 = np.array([0., 0., z_width])

    A5 = np.array([0., y_width, z_width])
    A6 = np.array([x_width, y_width, z_width])
    A7 = np.array([0., y_width, 0.0])
    A8 = np.array([x_width, y_width, 0.0])
    return np.array([A1, A2, A3, A4, A5, A6, A7, A8])


def rotate_platform_points(world_pose_tool, tool_position_plat_pts):
    t = np.zeros([np.shape(tool_position_plat_pts)[0], 3])
    for count, tool_t_p_i in enumerate(tool_position_plat_pts):
        t[count, :] = np.matmul(world_pose_tool[0:3, 0:3], tool_t_p_i)
    return t


# Simple workspace analysis for 8 cable suspended Cartesian CDPR robot
if __name__ == '__main__':
    x_max = 5.5
    y_max = 2.5
    z_max = 2.1
    wld_position_base_pts = get_base_attachment_points(x_max,y_max,z_max)  # The positions of the base attachment points
    tool_position_plat_pts = get_platform_attachment_points(0.4, 0.3,
                                                            0.2)  # The positions of the end effector attachment points
    minimum_wrench = np.array([0.0, 0.0, 10.0 * 9.81])  # lets say platform must support it's weight of 10kg includes docking forces

    minimum_wrench_sub = np.array(
        [0.0, 0.0, 5.0 * 9.81])  # lets say platform must support it's weight of 10kg includes docking forces


    # max and min cable limits
    t_max = 100  # maximum cable tension found form motors
    t_min = 2  # need to ensure cable tension is above zero
    tension_space_Vrep = tension_space_polytope(t_min,
                                                t_max,8)  # this defines a 'box' in tension space that has all feasible tensions

    tension_space_Vrep_sub = tension_space_polytope(t_min,
                                                t_max,4)  # this defines a 'box' in tension space that has all feasible tensions
    # An end effector position
     wld_pose_tool = np.identity(4)#.euler_matrix(0.0, 0.0, 0.2)
     wld_pose_tool[0:3, 3] = [0.5, 0.5, 0.5]  # Where the platform is at the moment
    #
    # W = wrench_matrix(wld_position_base_pts,
    #                   wld_pose_tool,
    #                   tool_position_plat_pts)  # Wt + we=0 https://hal.archives-ouvertes.fr/hal-01941785/document



   # AWS = get_Cartesian_polytope(W[0:3,:], tension_space_Vrep)

   # polyhull = get_Cartesian_polytope2(W[0:3, :], tension_space_Vrep)
   # print("Is the platform able to support itself?", point_in_hull2(minimum_wrench, polyhull))


    # # This returns plot all the forces that the platform can support in it's current position
    # # Uncomment to plot
    #ax = plot_polytope_3d(AWS)
    #print("Is the platform able to support itself?", point_in_hull(minimum_wrench, AWS))
    # # Uncomment to plot
    #ax.plot([minimum_wrench[0]], [minimum_wrench[1]], [minimum_wrench[2]], markerfacecolor = 'k', markeredgecolor = 'k', marker = 'o', markersize = 10)
    #
    # # Plot shows wrench space capacities and the minimum wrench
    #plt.show()
    #
    # # =================== Plot the workspace ========================= #


    x_min = 0.2
    y_min = 0.2
    z_min = 0.0

    # # Check if force is within convex hull
    tolerance = 0.075  # can't have zero cable lentgh
    x_space = np.linspace(x_min , x_max - 0.1, 14)
    y_space = np.linspace(y_min , y_max - 0.3, 12)
    z_space = np.linspace(z_min , z_max - 0.1, 20)
    feasible_points = np.empty((0, 3), float)
    for x in x_space:
        for y in y_space:
            for z in z_space:
                wld_pose_tool[0:3, 3] =  np.array([x, y, z])  # Where the platform is at the moment

                W = wrench_matrix(wld_position_base_pts,
                                  wld_pose_tool,
                                  tool_position_plat_pts)  # Wt + we=0 https://hal.archives-ouvertes.fr/hal-01941785/document
                AWS = get_Cartesian_polytope(W[0:3,:], tension_space_Vrep)
                if (point_in_hull(minimum_wrench, AWS)):
                    feasible_points = np.vstack([feasible_points, wld_pose_tool[0:3, 3] ])

    print(np.shape(feasible_points))

    # wrkspace_hull = ConvexHull(feasible_points, qhull_options='Qs QJ')
    # ax2=plot_polytope_3d(wrkspace_hull)

    A1 = np.array([x_min, y_min, z_max])
    A2 = np.array([x_min, y_max, z_max])
    A3 = np.array([x_max, y_min, z_max])
    A4 = np.array([x_max, y_max, z_max])

    fig = plt.figure()
    ax2 = fig.gca(projection='3d')

    ax2.plot([A1[0], A2[0], A4[0], A3[0], A1[0]],
             [A1[1], A2[1], A4[1], A3[1], A1[1]],
             [A1[2], A2[2], A4[2], A3[2], A1[2]], 'k', linewidth=5,markerfacecolor='y', markeredgecolor='k', marker='o',
             markersize=5)



    my_cmap = plt.get_cmap('hsv')
    ax2.scatter3D(feasible_points[:, 0], feasible_points[:, 1], feasible_points[:, 2],
                  marker='x'
                  )
    plt.title("Wrench Feasible Workspace")
    ax2.set_xlabel('X-axis', fontweight='bold')
    ax2.set_ylabel('Y-axis', fontweight='bold')
    ax2.set_zlabel('Z-axis', fontweight='bold')


    # Now do the sub robot
    docked_platform_origin_x=2.5
    docked_platform_origin_y=1.5
    docked_platform_origin_z=1.5
    platform_width=   0.6
    platform_breadth= 0.6
    platform_height= 0.2

    ax2.plot([A1[0], A2[0], A4[0], A3[0], A1[0]],
         [A1[1], A2[1], A4[1], A3[1], A1[1]],
         [A1[2], A2[2], A4[2], A3[2], A1[2]], 'k', linewidth=5, markerfacecolor='y', markeredgecolor='k', marker='o',
         markersize=5)

    wld_position_base_pts = get_sub_base_attachment_points(docked_platform_origin_x,docked_platform_origin_y,docked_platform_origin_z,
        platform_width,platform_breadth,platform_height)

    x_space = np.linspace(docked_platform_origin_x, docked_platform_origin_x + platform_width, 12)
    y_space = np.linspace(docked_platform_origin_y, docked_platform_origin_y + platform_breadth, 12)
    z_space = np.linspace(z_min, docked_platform_origin_z - 0.1, 20)

    feasible_points_sub = np.empty((0, 3), float)
    for x in x_space:
        for y in y_space:
            for z in z_space:
                wld_pose_tool[0:3, 3] = np.array([x, y, z])  # Where the platform is at the moment

                W = sub_wrench_matrix(wld_position_base_pts,
                                  wld_pose_tool
                                  )  # Wt + we=0 https://hal.archives-ouvertes.fr/hal-01941785/document
                AWS = get_Cartesian_polytope(W, tension_space_Vrep_sub)
                if (point_in_hull(minimum_wrench, AWS)):
                    feasible_points_sub = np.vstack([feasible_points, wld_pose_tool[0:3, 3]])



    plt.show()
    # # Now map the space
