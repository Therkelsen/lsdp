from lxml import etree
import numpy as np

def main():
    # load .xml file with camera intrinsics using etree
    intrinsic = etree.parse('lecture_2/3D_reconstruction/camrea.xml')
    f = -float(intrinsic.find('f').text)
    print("Focal length: ", f)
    
    rotation_matrix_elements = np.loadtxt("lecture_2/3D_reconstruction/EP.txt", skiprows=1, usecols=range(4, 13))
    # print("Rotation matrix elements: ", rotation_matrix_elements)
    rotation_matrices = rotation_matrix_elements.reshape(-1, 3, 3)
    rl = rotation_matrices[0].T
    rr = rotation_matrices[1].T
    
    image_points = np.loadtxt("lecture_2/3D_reconstruction/EP.txt", skiprows=1, usecols=range(1, 4))
    image_points = image_points.astype(float)
    
    Pl = image_points[0]
    Pr = image_points[1]
    
    print("Image points 1: ", image_points[0])
    print("Image points 2: ", image_points[1])
    
    # my points
    # pl = (3301, 635)
    # pr = (3307, 1175)
    # anders points
    # pl = (806, 3465)
    # pr = (1323, 3475)
    # alex peter points
    pl = (3209, 1360)
    pr = (3216, 1910)
    print("Point 1: ", pl)
    print("Point 2: ", pr)
    
    A = np.array([[pl[0]*rl[0,2] + f*rl[0,0], pl[0]*rl[1,2] + f*rl[1,0], pl[0]*rl[2,2] + f*rl[2,0]],
                  [pl[1]*rl[0,2] + f*rl[0,1], pl[1]*rl[1,2] + f*rl[1,1], pl[1]*rl[2,2] + f*rl[2,1]],
                  [pr[0]*rr[0,2] + f*rr[0,0], pr[0]*rr[1,2] + f*rr[1,0], pr[0]*rr[2,2] + f*rr[2,0]],
                  [pr[1]*rr[0,2] + f*rr[0,1], pr[1]*rr[1,2] + f*rr[1,1], pr[1]*rr[2,2] + f*rr[2,1]]])
    print("Matrix A:\n", A)
    
    L = np.array([[(pl[0]*rl[0,2] + f*rl[0,0])*Pl[0] + (pl[0]*rl[1,2] + f*rl[1,0])*Pl[1] + (pl[0]*rl[2,2] + f*rl[2,0])*Pl[2]],
                  [(pl[1]*rl[0,2] + f*rl[0,1])*Pl[0] + (pl[1]*rl[1,2] + f*rl[1,1])*Pl[1] + (pl[1]*rl[2,2] + f*rl[2,1])*Pl[2]],
                  [(pr[0]*rr[0,2] + f*rr[0,0])*Pr[0] + (pr[0]*rr[1,2] + f*rr[1,0])*Pr[1] + (pr[0]*rr[2,2] + f*rr[2,0])*Pr[2]],
                  [(pr[1]*rr[0,2] + f*rr[0,1])*Pr[0] + (pr[1]*rr[1,2] + f*rr[1,1])*Pr[1] + (pr[1]*rr[2,2] + f*rr[2,1])*Pr[2]]])
    print("Matrix L:\n", L)
    
    Xhat = np.linalg.inv(A.T @ A) @ A.T @ L

    print("Xhat:\n", Xhat)
    
if __name__ == "__main__":
    main()