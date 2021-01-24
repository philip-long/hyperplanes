import numpy as np



def sigmoid(z, a=1.0):
    # Continuous Sigmoid function returns 0 if z is negative and 1 if positive.
    # The slope is determined by a
    z = np.asarray(z)
    scalar_input = False
    if z.ndim == 0:
        z = z[None]  # Makes x 1D
        scalar_input = True
    z = 1.0 / (1.0 + np.exp(-z * a))
    if scalar_input:
        return z.item()
    return z

def skew(u):
    # skew symmetric matrix performing the cross product
    uskew=np.zeros((3,3))
    uskew[0, :]  = [0.0, -u[2], u[1]]
    uskew[1, :] =  [u[2], 0.0, -u[0]]
    uskew[2, :] =  [-u[1], u[0], 0.0]
    return uskew

def screw_transform(L):
    lhat = skew(L)
    m1=np.hstack([np.eye(3),-lhat])
    m2=np.hstack([np.zeros([3,3]), np.eye(3)])
    s=np.vstack([m1,m2])
    return s

def sigmoid_gradient(z, a=1.0):
    # Gradient of sigmoid function
    g = a * sigmoid(z, a) * (1 - sigmoid(z, a))
    return g

def cross_product_normalized(v1,v2):
    return np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))

def gradient_cross_product_normalized(v1,v2,dv1,dv2):
    # gradient of a cross product of two vector v1 v2 divided by its norm
    # dv1, dv2 are gradient of each vector
    # n = cross(v1,v2) / norm(cross(v1,v2))
    # n=   u/v
    # dn  =  du v - u dv
    # dq     dq       dq
    #       ---------------
    #             v'*v

    u=np.cross(v1,v2)
    v=np.linalg.norm(np.cross(v1,v2))

    dudq=gradient_cross_product(v1,v2,dv1,dv2)
    dvdq=gradient_vector_norm(u,dudq)

    dndq = ((dudq * v) - (u * dvdq)) / (v ** 2)
    return dndq


def gradient_vector_norm(v1,dv1):
    # gradient of the norm of a vector v1 with vector gradient dv1
    return np.dot(dv1,v1) / (np.dot(v1,v1)**0.5)

def gradient_cross_product(v1,v2,dv1,dv2):
    # gradient of the cross product of two vector v1 v2
    # dv1, dv2 are gradient of each vector
    return np.matmul(-skew(v2), dv1) + np.matmul(skew(v1), dv2)


