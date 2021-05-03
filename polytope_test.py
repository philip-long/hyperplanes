import numpy as np
import polytope
import polytope_functions
import sawyer_functions
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

def test_smooth_min_gradient(limits, step):
    coeffs_rands = np.random.randn(5, ) * 10
    test_domain = np.arange(limits[0], limits[1], step)
    num_max=0
    smooth_max=0

    analytical_gradient_list=[]
    numerical_gradient_list=[]
    error=[]

    first_iteration = True
    for z in test_domain:
        num_max_m1=num_max
        smooth_max_m1=smooth_max

        v1 = np.array([(z ** 2) + (coeffs_rands[1] * z), (-6 * (z ** 2)) + (12 * z), coeffs_rands[0] * z])
        dv1 = np.array([(2 * z) + coeffs_rands[1], (-12 * z) + 12, coeffs_rands[0]])

        num_max=np.min(v1)
        smooth_max=robot_functions.smooth_max(-v1)

        if not first_iteration:
            analytical_gradient = robot_functions.exp_normalize(-v1)
            analytical_gradient_scalar=analytical_gradient[np.argmin(v1)]*dv1[np.argmin(v1)]

            numerical_gradient = ((num_max- num_max_m1) / step)

            analytical_gradient_list.append(analytical_gradient_scalar)
            numerical_gradient_list.append(numerical_gradient)
            error.append(analytical_gradient_scalar - numerical_gradient)
        first_iteration = False

    plt.plot(error)
    plt.plot(analytical_gradient_list, 'r')
    plt.plot(numerical_gradient_list, 'g')
    plt.show()

def test_smooth_max_gradient(limits, step):
    coeffs_rands = np.random.randn(5, ) * 10
    test_domain = np.arange(limits[0], limits[1], step)
    num_max=0
    smooth_max=0

    analytical_gradient_list=[]
    numerical_gradient_list=[]
    error=[]

    first_iteration = True
    for z in test_domain:
        num_max_m1=num_max
        smooth_max_m1=smooth_max

        v1 = np.array([(z ** 2) + (coeffs_rands[1] * z), (-6 * (z ** 2)) + (12 * z), coeffs_rands[0] * z])
        dv1 = np.array([(2 * z) + coeffs_rands[1], (-12 * z) + 12, coeffs_rands[0]])

        num_max=np.max(v1)
        smooth_max=robot_functions.smooth_max(v1)
        print("smooth_max",smooth_max,num_max)

        if not first_iteration:
            analytical_gradient = robot_functions.exp_normalize(v1)
            analytical_gradient_scalar=analytical_gradient[np.argmax(v1)]*dv1[np.argmax(v1)]

            numerical_gradient = ((np.max(v1) - num_max_m1) / step)

            analytical_gradient_list.append(analytical_gradient_scalar)
            numerical_gradient_list.append(numerical_gradient)
            error.append(analytical_gradient_scalar - numerical_gradient)
        first_iteration = False

    plt.plot(error)
    plt.plot(analytical_gradient_list, 'r')
    plt.plot(numerical_gradient_list, 'g')
    plt.show()


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

    coeffs_rands = np.random.randn(5, ) * 10

    test_domain = np.arange(limits[0], limits[1], step)

    numerical_gradient_list = np.empty(3, )
    analytical_gradient_list = np.empty(3, )
    error = np.empty(3, )

    first_iteration = True
    for z in test_domain:
        v1_cross_v2_last = v1_cross_v2

        v1 = np.array([(z ** 2) + (coeffs_rands[1] * z), (-6 * (z ** 2)) + (12 * z), coeffs_rands[0] * z])
        dv1 = np.array([(2 * z) + coeffs_rands[1], (-12 * z) + 12, coeffs_rands[0]])
        v2 = np.array([(z ** 3) + 2 * z, -2 * z, -coeffs_rands[4] * z])
        dv2 = np.array([(3 * (z ** 2)) + 2, -2, -coeffs_rands[4]])
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

def test_hessian(limits,step):
    q=np.random.randn(7)
    joint=np.random.randint(7)
    test_domain = np.arange(limits[0], limits[1], step)
    J=sawyer_functions.jacobianE0(q)
    error = []
    nm_tes=[]
    for z in test_domain:
        J_last=J

        q[joint]=z
        J = sawyer_functions.jacobianE0(q)
        H=robot_functions.getHessian(J)
        numerical_gradient = ((J-J_last) / step)
        nm_tes.append(numerical_gradient[2,3])
        error.append( np.linalg.norm(numerical_gradient-H[:,:,joint]))

    plt.plot(nm_tes[1:], 'b')
    plt.plot(error[1:], 'r')
    plt.show()
    return max(error)

def test_gamma_versus_gammahat(limits,step):
    q=np.random.randn(7)
    #q = np.array([0.9, 0.5, 0.5, 0.3, -0.13, 0.6, 0.9])
    qdot_max = np.ones([7,])*100.0
    qdot_min = np.ones([7,])*-50.0
    joint=np.random.randint(7)
    test_domain = np.arange(limits[0], limits[1], step)

    J=sawyer_functions.jacobianE0(q)
    H = robot_functions.getHessian(J)
    JE=J[0:3,:]

    vertices = np.array([[0.50000, 0.50000, 0.50000],
                         [0.50000, -0.10000, 0.50000],
                         [0.50000, 0.50000, -0.60000],
                         [0.50000, -0.10000, -0.60000],
                         [-0.30000, 0.50000, 0.50000],
                         [-0.30000, -0.10000, 0.50000],
                         [-0.30000, 0.50000, -0.60000],
                         [-0.30000, 0.450000, -0.62000],
                         [-0.30000, -0.10000, -0.60000]])



    m = np.shape(JE)[0]  # number of task space degrees of freedom
    number_of_joints = np.shape(JE)[1]  # number of joints
    active_joints = np.arange(number_of_joints)  # listing our the joints
    N, Nnot = robot_functions.getDofCombinations(active_joints, m)
    number_of_combinations = np.shape(N)[0]

    gamma_hat_last = 0
    Gamma_plus_last = np.zeros([number_of_combinations , np.shape(vertices)[0]])
    Gamma_minus_last = np.zeros([number_of_combinations, np.shape(vertices)[0]])
    gamma_err=[]
    gh_err=[]
    for z in test_domain:
        J_last=JE

        q[joint]=z
        JE = sawyer_functions.jacobianE0(q)
        H=robot_functions.getHessian(JE)
        JE=JE[0:3,:]

        gamma_hat,d_gamma_dq,gamma_all= polytope_functions.get_gamma_hat(JE, H, qdot_max, qdot_min, vertices, 200)
        print("this should be positive gamma_hat", gamma_hat)
        print("this should be positive gamma_all", gamma_all)
        print("Error", gamma_all-gamma_hat)
        gamma_hat_last=gamma_hat

#    plt.show()
    return 1

def test_gamma_hat_gradient(limits,step):
    q=np.random.randn(7)
    #q = np.array([0.9, 0.5, 0.5, 0.3, -0.13, 0.6, 0.9])
    qdot_max = np.ones([7,])*100.0
    qdot_min = np.ones([7,])*-50.0
    joint=np.random.randint(7)
    test_domain = np.arange(limits[0], limits[1], step)

    J=sawyer_functions.jacobianE0(q)
    H = robot_functions.getHessian(J)
    JE=J[0:3,:]

    vertices = np.array([[0.50000, 0.50000, 0.50000],
                         [0.50000, -0.10000, 0.50000],
                         [0.50000, 0.50000, -0.60000],
                         [0.50000, -0.10000, -0.60000],
                         [-0.30000, 0.50000, 0.50000],
                         [-0.30000, -0.10000, 0.50000],
                         [-0.30000, 0.50000, -0.60000],
                         [-0.30000, 0.450000, -0.62000],
                         [-0.30000, -0.10000, -0.60000]])



    m = np.shape(JE)[0]  # number of task space degrees of freedom
    number_of_joints = np.shape(JE)[1]  # number of joints
    active_joints = np.arange(number_of_joints)  # listing our the joints
    N, Nnot = robot_functions.getDofCombinations(active_joints, m)
    number_of_combinations = np.shape(N)[0]

    gamma_hat_last = 0
    Gamma_plus_last = np.zeros([number_of_combinations , np.shape(vertices)[0]])
    Gamma_minus_last = np.zeros([number_of_combinations, np.shape(vertices)[0]])
    gamma_err=[]
    gh_err=[]
    for z in test_domain:
        J_last=JE

        q[joint]=z
        JE = sawyer_functions.jacobianE0(q)
        H=robot_functions.getHessian(JE)
        JE=JE[0:3,:]

        gamma_hat,d_gamma_dq,gamma_all= polytope_functions.get_gamma_hat(JE, H, qdot_max, qdot_min, vertices, 200)
        print("this should be positive gamma_hat", gamma_hat)
        print("this should be positive gamma_all", gamma_all)
        print("Gradient", d_gamma_dq)
        print("numerical gradient", ((gamma_hat - gamma_hat_last) / step))
        print("Gradient error", d_gamma_dq[joint] - ((gamma_hat - gamma_hat_last) / step))

        #gh_err.append(np.max(d_gamma_dq[joint] - ((gamma_hat - gamma_hat_last) / step)))
        #gamma_err.append(gamma_hat-gamma_all)

        gamma_hat_last=gamma_hat

#    plt.show()
    return 1

def test_hyperplanes(limits,step):
    q=np.random.randn(7)
    joint=np.random.randint(7)
    test_domain = np.arange(limits[0], limits[1], step)
    JE=sawyer_functions.jacobianE0_trans(q)

    m = np.shape(JE)[0]  # number of task space degrees of freedom
    number_of_joints = np.shape(JE)[1]  # number of joints
    active_joints = np.arange(number_of_joints)  # listing our the joints
    N, Nnot = robot_functions.getDofCombinations(active_joints, m)
    number_of_combinations = np.shape(N)[0]

    n_last = np.zeros([number_of_combinations, m])
    hplus_last = np.zeros([number_of_combinations, ])
    hminus_last = np.zeros([number_of_combinations, ])

    qdot_max = np.ones([7,])*10.0
    qdot_min = np.ones([7,])*-5.0
    deltaq = qdot_max - qdot_min
    error=0
    JE = sawyer_functions.jacobianE0(q)
    H = robot_functions.getHessian(JE)
    JE = JE[0:3, :]
    n, hplus, hminus, d_n_dq, d_hplus_dq, d_hminus_dq = polytope_functions.get_hyperplane_parameters(JE, H, deltaq)
    ng_err=[]
    hp_err=[]
    hm_err=[]
    for z in test_domain:
        J_last=JE

        q[joint]=z
        JE = sawyer_functions.jacobianE0(q)
        H=robot_functions.getHessian(JE)
        JE=JE[0:3,:]
        n, hplus, hminus, d_n_dq, d_hplus_dq, d_hminus_dq = polytope_functions.get_hyperplane_parameters(JE, H, deltaq)

        ng_err.append(np.max(d_n_dq[:, :, joint] - ((n-n_last) / step)   ))
        hp_err.append(np.max(d_hplus_dq[:, joint] - ((hplus - hplus_last) / step)))
        hm_err.append(np.max(d_hminus_dq[:, joint] - ((hminus - hminus_last) / step)))
        n_last = n
        hplus_last = hplus
        hminus_last = hminus


    plt.plot(ng_err[1:], 'b')
    plt.plot(hp_err[1:], 'r')
    plt.plot(hm_err[1:], 'g')
    plt.show()
    return max([ng_err,hp_err,hm_err])


def test_gammas_gradient(limits,step):
    q=np.random.randn(7)
    qdot_max = np.ones([7,])*100.0
    qdot_min = np.ones([7,])*-50.0
    joint=np.random.randint(7)

    joint=np.random.randint(7)
    test_domain = np.arange(limits[0], limits[1], step)
    JE=sawyer_functions.jacobianE0_trans(q)

    m = np.shape(JE)[0]  # number of task space degrees of freedom
    number_of_joints = np.shape(JE)[1]  # number of joints
    active_joints = np.arange(number_of_joints)  # listing our the joints
    N, Nnot = robot_functions.getDofCombinations(active_joints, m)
    number_of_combinations = np.shape(N)[0]
    vertices = np.array([[0.50000, 0.50000, 0.50000],
                         [0.50000, -0.10000, 0.50000],
                         [0.50000, 0.50000, -0.60000],
                         [0.50000, -0.10000, -0.60000],
                         [-0.30000, 0.50000, 0.50000],
                         [-0.30000, -0.10000, 0.50000],
                         [-0.30000, 0.50000, -0.60000],
                         [-0.30000, 0.450000, -0.62000],
                         [-0.30000, -0.10000, -0.60000]])
    n_last = np.zeros([number_of_combinations, m])
    hplus_last = np.zeros([number_of_combinations, ])
    hminus_last = np.zeros([number_of_combinations, ])
    Gamma_plus_last= np.zeros([np.shape(hplus_last)[0], np.shape(vertices)[0]])
    Gamma_minus_last = np.zeros([np.shape(hplus_last)[0], np.shape(vertices)[0]])
    qdot_max = np.ones([7,])*10.0
    qdot_min = np.ones([7,])*-5.0
    deltaq = qdot_max - qdot_min
    error=0
    JE = sawyer_functions.jacobianE0(q)
    H = robot_functions.getHessian(JE)
    JE = JE[0:3, :]

    gp_err=[]
    gm_err=[]
    hm_err=[]
    for z in test_domain:
        J_last=JE

        q[joint]=z
        JE = sawyer_functions.jacobianE0(q)
        H=robot_functions.getHessian(JE)
        JE=JE[0:3,:]
        Gamma_plus, Gamma_minus, d_Gamma_plus_dq, d_Gamma_minus_dq\
            =polytope_functions.get_gamma(JE, H, qdot_max, qdot_min, vertices,200)

      #  print("max gradient error Gamma_plus",np.max(d_Gamma_plus_dq[:, :, joint] - ((Gamma_plus - Gamma_plus_last) / step)))
      #  print("max gradient error Gamma_minus", np.max(d_Gamma_minus_dq[:, :, joint] - ((Gamma_minus - Gamma_plus_last) / step)))
        gp_err.append(np.max(d_Gamma_plus_dq[:, :, joint] - ((Gamma_plus - Gamma_plus_last) / step)))
        gm_err.append(np.max(d_Gamma_minus_dq[:, :, joint] - ((Gamma_minus - Gamma_minus_last) / step)))
        Gamma_plus_last=Gamma_plus
        Gamma_minus_last = Gamma_minus

    plt.plot(gp_err[1:], 'b')
    plt.plot(gm_err[1:], 'r')
    plt.show()




if __name__ == '__main__':
    # test_plot()
    # print("Max error sigmoid gradient",test_sigmoid_gradient([-5.0,5.0],0.001,10))
    # print("Max error of cross product gradient",test_cross_product_gradient([-1.0, 1.0], 0.0001))
    # print("Max error of norm vector gradient",test_vector_norm_gradient([-2.0, 1.0], 0.0001))
    # print("Max error of normalized cross product gradient", test_normalized_cross_product_gradient([-1.0, 1.0], 0.0001))
    # print("Max error of Hessian", test_hessian([0.0, np.pi], 0.005))
    # print("Max error of smooth max", test_smooth_max_gradient([-5.0, 5.0],0.0001))
    # print("Max error of smooth max", test_smooth_min_gradient([-5.0, 5.0],0.0001))
    # print("Max error of test_hyperplanes ", test_hyperplanes([-1.0, 1.0],0.001))
    # print("Comparison Gamma versus Gamma hat ",test_gamma_versus_gammahat([-10.0, 10.0],0.0001))
    print("Testing gamma gradient", test_gammas_gradient([-2.0, 2.0], 0.01))
    #  Something not right with gamma hat gradient -> check gammas hat gradient

    # Applications
