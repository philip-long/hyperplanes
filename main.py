import numpy as np
import robot_functions
import polytope_functions
import sawyer_functions
import polytope

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    sigmoid_slope = 200
    q = np.array([0.9,0.5,0.5,0.3,-0.13,0.6,0.9])
    q=np.random.random(7,)
    qdot_max = np.ones([7,])*200.0
    qdot_min = np.ones([7,])*-50.0
    deltaq=qdot_max-qdot_min
    print(deltaq)


    J=sawyer_functions.jacobianE0(q)
    H = robot_functions.getHessian(J)
    JE=J[0:3,:]
    m=np.shape(JE)[0] # number of task space degrees of freedom
    number_of_joints =np.shape(JE)[1] # number of joints
    active_joints = np.arange(number_of_joints)  # listing our the joints

    n,hplus, hminus, d_n_dq, d_hplus_dq, d_hminus_dq\
        = polytope_functions.get_hyperplane_parameters(JE, H, deltaq, sigmoid_slope)
    n2,hplus2, hminus2= polytope_functions.get_reduced_hyperplane_parameters(JE, deltaq, active_joints, sigmoid_slope)

    A_desired = np.vstack((np.eye(3, 3), -1 * np.eye(3, 3)))
    B_desired = np.array([0.5, 0.5, 0.5, 0.3, 0.1, 0.6])

    desired_twist = polytope.Polytope(A_desired, B_desired)

  #  vertices = polytope.extreme(desired_twist)
    vertices = np.array([[0.50000, 0.50000, 0.50000],
                         [0.50000, -0.10000, 0.50000],
                         [0.50000, 0.50000, -0.60000],
                         [0.50000, -0.10000, -0.60000],
                         [-0.30000, 0.50000, 0.50000],
                         [-0.30000, -0.10000, 0.50000],
                         [-0.30000, 0.50000, -0.60000],
                         [-0.30000, 0.450000, -0.62000],
                         [-0.30000, -0.10000, -0.60000]])

    Gamma_plus,Gamma_minus,d_Gamma_plus_dq,d_Gamma_minus_dq=polytope_functions.get_gamma(JE,H,qdot_max,qdot_min,vertices,sigmoid_slope)
    gamma_hat=polytope_functions.get_gamma_hat(JE,H,qdot_max,qdot_min,vertices,sigmoid_slope)
    Gamma_all = np.vstack([-Gamma_plus, Gamma_minus])  # Stacking
    ind = np.unravel_index(np.argmax(Gamma_all, axis=None), Gamma_all.shape)  # Get's the index of the minimum value
    print("Gamma_all[ind]=",Gamma_all[ind])
    print("Gamma_all[ind]=", np.min(-Gamma_all))
    print("gamma_hat=", gamma_hat)

    #print(d_Gamma_plus_dq2,np.shape(d_Gamma_plus_dq2))
