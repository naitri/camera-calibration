import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import scipy.optimize 




def get_img_points(image,nx,ny,num):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_points = []
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        corners = corners.reshape(-1, 2)
        # Append image points
        corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        img_points.append(corners)

        # Draw and display the corners
        # cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
        # plt.imshow(image)
        # plt.savefig(num+'.png')
        
        return np.array(img_points)


def get_world_points(img,square_size,nx,ny):

    x, y = np.meshgrid(np.linspace(0,nx-1,nx), np.linspace(0,ny-1,ny))
    x = np.flip((x.reshape(54, 1) * square_size), axis=0)
    y = (y.reshape(54,1)*square_size)
    M = np.float32(np.hstack((y,x)))
    
  
    return M

def get_homography(points1,points2):
    H,_ = cv2.findHomography(points1,points2)
    return H

def compute_Vij(H, i,j):
    i,j = i-1,j-1
    v_ij = np.array([H[0, i]*H[0, j],
                    H[0, i]*H[1, j] + H[1, i]*H[0, j],
                    H[1, i]*H[1, j],
                    H[2, i]*H[0, j] + H[0, i]*H[2, j],
                    H[2, i]*H[1, j] + H[1, i]*H[2, j],
                    H[2, i]*H[2, j] 
                    ])
    return v_ij

def compute_V(H):
    #calculating eqtn 8 in paper
    V = []
    for h in H:
        v12 = compute_Vij(h,1,2).T
        v11 = compute_Vij(h,1,1)
        v22 = compute_Vij(h,2,2)
        v11_v22 = (v11-v22).T
        V.append(v12)
        V.append(v11_v22)
       

    return np.array(V)

def compute_B(V):
    u, sigma, v = np.linalg.svd(V)
    b = v[-1, :]
    return b

def compute_K(H):
    #compute V
    V = compute_V(H)
   
    #compute b
    b = compute_B(V)
    
    
    b11, b12, b22, b13, b23, b33 = b[0],b[1],b[2],b[3],b[4],b[5]

    v0 = (b12*b13 - b11*b23)/(b11*b22 - b12**2)
    lamda = b33 - (b13**2 + v0*(b12*b13 - b11*b23))/b11
    alpha = np.sqrt(lamda/b11)
    beta = np.sqrt(lamda*b11 /(b11*b22 - b12**2))
    gamma = -b12*(alpha**2)*beta/lamda
    u0 = gamma*v0/beta -b13*(alpha**2)/lamda

    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,      1]])
    return K

def compute_Rt(K,H):
    extrinsic = []
    for h in H:
        h1,h2,h3 = h.T # get the column vectors

        K_inv = np.linalg.inv(K)
        lamda = 1/np.linalg.norm(K_inv.dot(h1),ord =2 )
        r1 = lamda*K_inv.dot(h1)
        r2 = lamda*K_inv.dot(h2)
        r3 = np.cross(r1,r2)
        t = lamda*K_inv.dot(h3)
        RT = np.vstack((r1, r2, r3, t)).T
        extrinsic.append(RT)
    return extrinsic

def reprojection_error(initial_params,world_points,img_points_set,RT):
    final_error = []
    error = []
    for i,RT3 in enumerate(RT):
        mi_hat = projection(initial_params,world_points[i],img_points_set[i],RT3)
        
        mi = img_points_set[i].reshape(54,2)


        for m, m_ in  zip(mi, mi_hat.squeeze()):
            
            e = np.linalg.norm(m - m_, ord=2) # compute L2 norm

            error.append(e)

        err = np.mean(error)
       
        final_error.append(err)

    return final_error

def projection(initial_params,world_points,mi,RT):
    alpha, beta, gamma,u0,v0,k1,k2=initial_params

    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,      1]])
    kc = (k1,k2)

    m_i_ = []
    error = []
    
    
    for M in world_points:

        M = np.float64(np.hstack((M,0,1)))

        projected_pt = np.dot(RT,M)
        projected_pt = projected_pt/projected_pt[-1]

            #compute radius of distortion
        x = projected_pt[0]
        y = projected_pt[1]
      
        r = x**2 + y**2
      
        

            #projected image coordinates
                   

        uv = np.dot(K,projected_pt)
        u = uv[0]/uv[-1]
        v = uv[1]/uv[-1]
      
        #eqtn 11 and 12 from the paper
        u_hat = u+ (u-u0)*(k1*r + k2*(r**2))
        v_hat = v + (v-v0)*(k1*r + k2*(r**2))
       
        m_ = np.hstack((u_hat,v_hat))
        
        m_i_.append(m_)
    return np.array(m_i_)

def loss(initial_params,world_points,img_points_set,RT):
    
    final_error = []
    error = []
    for i,RT3 in enumerate(RT):
        mi_hat = projection(initial_params,world_points[i],img_points_set[i],RT3)
        mi = img_points_set[i].reshape(54,2)
   

        for m, m_ in  zip(mi, mi_hat.squeeze()):
            e = np.linalg.norm(m - m_, ord=2) # compute L2 norm
           
            error.append(e)
        err = np.sum(error)
       
        
       
      

        final_error.append(err)
 
    return final_error



def optimize(initial_params,world_points_set,img_points_set,RT):
    opt = scipy.optimize.least_squares(fun = loss, x0 = initial_params, method="lm", args = [world_points_set, img_points_set, RT])
    params = opt.x

  
    alpha, beta, gamma, u0, v0, k1 ,k2 = params
    K_new= np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,      1]])
    kc = (k1,k2)

    return K_new,kc



def main():

    nx = 9 #number of inside corners in x 
    ny = 6# number of inside corners in y
    square_size = 21.5 #21.5mm
    img_points_set = []
    world_points_set = []
    H_matrix_set = []
    num = 0

    #read all images
    for image in sorted(glob.glob("Calibration_Imgs/*.jpg")):
        img = cv2.imread(image)
        img_points = get_img_points(img,nx,ny,str(num))
        world_points =  get_world_points(img,square_size,nx,ny)
        H = get_homography(world_points,img_points[0])
        img_points_set.append(img_points)
        world_points_set.append(world_points)
        H_matrix_set.append(H)
        num +=1
    

   #Estimating K
    #Vb= 0 solving this will give K

    K_init = compute_K(H_matrix_set)
    print("The intrinsic matrix K is:\n",K_init)

    # #Compute Extrinsic parameters
    RT = compute_Rt(K_init,H_matrix_set)
    print("The extrinsic matrix [R|t]is \n:",RT[0])



    #Optimize
    alpha, beta, gamma,u0,v0 = K_init[0, 0], K_init[1, 1], K_init[0, 1] ,K_init[0, 2], K_init[1, 2]
    k1,k2=0,0
    initial_params = [alpha, beta, gamma,u0,v0,k1,k2]
    projection_error = reprojection_error(initial_params,world_points_set,img_points_set,RT)
    print("projection error:\n",np.mean(projection_error))

    K_new, kc = optimize(initial_params,world_points_set,img_points_set,RT)
    print("The new intrinsic matrix K is:\n",K_new)
    print("kc is:\n", kc)
   
  
    RT_new = compute_Rt(K_new,H_matrix_set)
    print("The new extrinsic matrix [R|t]is \n:",RT_new[0])

    #Get new image points
    new_img_points = []
    for i,rt in enumerate(RT_new):
        world_point = np.column_stack((world_points_set[i], np.ones(len(world_points_set[i]))))
        r1,r2,r3,t = rt.T
        R = np.stack((r1,r2,r3), axis=1)
        t = t.reshape(-1,1)
        img_pt, _ = cv2.projectPoints(world_point, R, t, K_new, (kc[0],kc[1], 0, 0))
        new_img_points.append(img_pt.squeeze())


    #Calculate Mean reprojection error
    alpha, beta, gamma,u0,v0 = K_new[0, 0], K_new[1, 1], K_new[0, 1] ,K_new[0, 2], K_new[1, 2]
    final_params = [alpha, beta, gamma,u0,v0,kc[0],kc[1]]
    projection_error_new = reprojection_error(final_params,world_points_set,new_img_points,RT_new)
    print("reprojection error:",np.mean(projection_error_new))
    
    distortion = np.array([kc[0],kc[1],0,0,0],dtype=float)
    i = 0
    for image in sorted(glob.glob("Calibration_Imgs/*.jpg")):
        img = cv2.imread(image)
        img = cv2.undistort(img,K_new,distortion)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,(9,6),None)
        out= cv2.drawChessboardCorners(img,(9,6),corners,ret)
        plt.imshow(out)
        plt.savefig('result'+str(i)+'.png')
        i+=1
    
  

if __name__=="__main__":
    main()
