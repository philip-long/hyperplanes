import numpy as np
import polytope
import polytope_functions
import robot_functions
import matplotlib.pyplot as plt


def test_plot():
    A_desired = np.vstack((np.eye(3, 3), -1 * np.eye(3, 3)))
    B_desired = np.ones([6, 1])
    print(A_desired)
    print(B_desired)
    desired_twist = polytope.Polytope(A_desired, B_desired)
    print(desired_twist.volume)
    V = polytope.extreme(desired_twist)
    print("polytope vertices=", polytope.extreme(desired_twist))
    print("desired_twist vertices=", V)
    polytope_functions.plot_polytope_3d(desired_twist)


def test_sigmoid_gradient(limits, step, a):
    x_n = 0
    test_domain = np.arange(limits[0], limits[1], step)

    numerical_gradient_list = []
    analytical_gradient_list = []
    error = []
    for z in test_domain:
        x_nm1 = x_n
        x_n = robot_functions.sigmoid(z, a)
        analytical_gradient = robot_functions.sigmoid_gradient(z, a)
        numerical_gradient = ((x_n - x_nm1) / step)
        analytical_gradient_list.append(analytical_gradient)
        numerical_gradient_list.append(numerical_gradient)
        error.append(analytical_gradient - numerical_gradient)
    plt.plot(error)
    plt.plot(analytical_gradient_list, 'r')
    plt.plot(numerical_gradient_list, 'g')
    plt.show()
    return max(error)


def test_cross_product_gradient(limits, step):
    v1_cross_v2 = np.zeros(3, )

    rands = np.random.randn(5, ) * 10

    test_domain = np.arange(limits[0], limits[1], step)

    numerical_gradient_list = np.empty(3, )
    analytical_gradient_list = np.empty(3, )
    error = np.empty(3, )

    first_iteration = True
    for z in test_domain:
        v1_cross_v2_last = v1_cross_v2

        v1 = np.array([(z ** 2) + (rands[1] * z), (-6 * (z ** 2)) + (12 * z), rands[0] * z])
        dv1 = np.array([(2 * z) + rands[1], (-12 * z) + 12, rands[0]])
        v2 = np.array([(z ** 3) + 2 * z, -2 * z, -rands[4] * z])
        dv2 = np.array([(3 * (z ** 2)) + 2, -2, -rands[4]])
        v1_cross_v2 = np.cross(v1, v2)

        analytical_gradient = robot_functions.gradient_cross_product(v1, v2, dv1, dv2)
        numerical_gradient = ((v1_cross_v2 - v1_cross_v2_last) / step)

        if not first_iteration:
            analytical_gradient_list = np.vstack([analytical_gradient_list, analytical_gradient])
            numerical_gradient_list = np.vstack([numerical_gradient_list, numerical_gradient])
            error = np.vstack([error, analytical_gradient - numerical_gradient])

        first_iteration = False

    print("error = ", error)
    print("analytical_gradient = ", analytical_gradient_list)
    print("numerical_gradient = ", numerical_gradient_list)
    for i in range(3):
        plt.figure(i)
        plt.plot(error[:, i], 'r')
        plt.plot(analytical_gradient_list[:, i], 'g')
        plt.plot(numerical_gradient_list[:, i], 'b')

    plt.show()
    return np.max(np.absolute(error), axis=0)  # max of each column


def test_vector_norm_gradient(limits, step):
    rands = np.random.randn(5, ) * 10
    test_domain = np.arange(limits[0], limits[1], step)
    numerical_gradient_list = []
    analytical_gradient_list = []
    error = []
    norm_single_vector = 0.0
    for z in test_domain:
        norm_single_vector_last = norm_single_vector
        v1 = np.array([(z ** 2) + (rands[1] * z), (-rands[3] * (z ** 2)) + (12 * z), rands[0] * z])
        dv1 = np.array([(2 * z) + rands[1], (-(2 * rands[3]) * z) + 12, rands[0]])
        norm_single_vector = np.linalg.norm(v1)
        analytical_gradient = robot_functions.gradient_vector_norm(v1, dv1)
        numerical_gradient = (norm_single_vector - norm_single_vector_last) / step

        analytical_gradient_list.append(analytical_gradient)
        numerical_gradient_list.append(numerical_gradient)
        error.append(analytical_gradient - numerical_gradient)

    plt.plot(error[1:], 'r')
    plt.plot(analytical_gradient_list[1:], 'g')
    plt.plot(numerical_gradient_list[1:], 'b')
    plt.show()
    return max(error)


def test_normalized_cross_product_gradient(limits, step):
    v1_cross_v2_normalized = np.zeros(3, )
    rands = np.random.randn(5, ) * 10

    test_domain = np.arange(limits[0], limits[1], step)

    numerical_gradient_list = np.empty(3, )
    analytical_gradient_list = np.empty(3, )
    error = np.empty(3, )

    first_iteration = True
    for z in test_domain:
        v1_cross_v2_normalized_last = v1_cross_v2_normalized

        v1 = np.array([(z ** 2) + (rands[1] * z), (-6 * (z ** 2)) + (12 * z), rands[0] * z])
        dv1 = np.array([(2 * z) + rands[1], (-12 * z) + 12, rands[0]])
        v2 = np.array([(z ** 3) + 2 * z, -2 * z, -rands[4] * z])
        dv2 = np.array([(3 * (z ** 2)) + 2, -2, -rands[4]])

        v1_cross_v2_normalized = robot_functions.cross_product_normalized(v1, v2)
        analytical_gradient = robot_functions.gradient_cross_product_normalized(v1, v2, dv1, dv2)
        numerical_gradient = ((v1_cross_v2_normalized - v1_cross_v2_normalized_last) / step)

        if not first_iteration:
            analytical_gradient_list = np.vstack([analytical_gradient_list, analytical_gradient])
            numerical_gradient_list = np.vstack([numerical_gradient_list, numerical_gradient])
            error = np.vstack([error, analytical_gradient - numerical_gradient])

        first_iteration = False

    print("error = ", error)
    print("analytical_gradient = ", analytical_gradient_list)
    print("numerical_gradient = ", numerical_gradient_list)
    for i in range(3):
        plt.figure(i)
        plt.plot(error[:, i], 'r')
        plt.plot(analytical_gradient_list[:, i], 'g')
        plt.plot(numerical_gradient_list[:, i], 'b')

    plt.show()
    return np.max(np.absolute(error), axis=0)  # max of each column


if __name__ == '__main__':
    # test_plot()
    # print("Max error sigmoid gradient",test_sigmoid_gradient([-5.0,5.0],0.001,10))
    # print("Max error of cross product gradient",test_cross_product_gradient([-1.0, 1.0], 0.0001))
    # print("Max error of norm vector gradient",test_vector_norm_gradient([-2.0, 1.0], 0.0001))
    print("Max error of normalized cross product gradient", test_normalized_cross_product_gradient([-1.0, 1.0], 0.0001))
