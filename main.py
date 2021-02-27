import numpy as np
import robot_functions
import sawyer_functions
import itertools

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    q=np.random.randn(7)
    q=np.array([0.5,0.5,0.5,0.3,0.1,0.6,0.9])

    J=sawyer_functions.jacobianE0(q)
    H = robot_functions.getHessian(J)
    JE=J[0:3,:]
    m=np.shape(JE)[0] # number of task space degrees of freedom
    number_of_joints =np.shape(JE)[1] # number of joints
    active_joints = np.arange(number_of_joints) + 1 # listing our the joints
    N, Nnot=robot_functions.getDofCombinations(active_joints,m)

    print(N)
    print(Nnot)

    # Hyperplanes and all that
    number_of_combinations=np.shape(N)[0]
    n = np.zeros([number_of_combinations, m]);

    hplus = np.zeros([number_of_combinations, 1]);
    hminus = np.zeros([number_of_combinations, 1]);

    d_n_dq = np.zeros([number_of_joints,3])
    d_hplus_dq = np.zeros([number_of_joints,3])
    d_hminus_dq = np.zeros([number_of_joints,3])

    for i in range(np.shape(N)[0]):
        v1 = JE[:,N[i,0]]
        v2 = JE[:, N[i, 2]]
        n[i,:]=robot_functions.cross_product_normalized(v1,v2)
