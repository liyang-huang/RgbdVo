import sys
import numpy as np
import cv2 as cv
from dataclasses import dataclass
import math
import sophus as sp
from scipy.spatial.transform import Rotation as R

#np.set_printoptions(threshold=np.inf)

@dataclass
class RgbdFrame:
    image: np.array
    depth: np.array
    mask: np.array
    normals: np.array

def frame_copy(dst, src):
    dst.image = src.image.copy()
    dst.depth = src.depth.copy()
    dst.mask = src.mask.copy()
    dst.normals = src.normals.copy()

def write_results(filename, timestamps, Rts):
    if(len(timestamps) != len(Rts)):
        print('Write result fail !!!')
        exit()

    with open(filename, 'w') as f:
        for i in range(len(Rts)):
            Rt_curr = np.around(Rts[i], 6)
            R = Rt_curr[0:3,0:3]

            rvec, jacobian = cv.Rodrigues(R)
            alpha = cv.norm(rvec)
            if(alpha!=0):
                rvec = rvec/alpha
            cos_alpha2 = math.cos(0.5*alpha)
            sin_alpha2 = math.sin(0.5*alpha)
            rvec = rvec*sin_alpha2
            rvec = np.squeeze(rvec)
            rvec = np.around(rvec, 6)
            cos_alpha2 = np.around(cos_alpha2, 6)
            f.write(timestamps[i]+' '+str(Rt_curr[0][3])+' '+str(Rt_curr[1][3])+' '+str(Rt_curr[2][3])+' '+str(rvec[0])+' '+str(rvec[1])+' '+str(rvec[2])+' '+str(cos_alpha2)+'\n')
    

def buildPyramidImage(img, levelCount):
    pyImg = []
    pyImg.append(img)
    for i in range(levelCount):
        imgDn = cv.pyrDown(pyImg[-1])
        pyImg.append(imgDn)

    return pyImg  

def buildPyramidCameraMatrix(cameraMatrix, level):
    pyCameraMat = []
    pyCameraMat.append(cameraMatrix)
    for i in range(level):
        matDn = pyCameraMat[-1] * 0.5
        matDn[2][2] = 1
        pyCameraMat.append(matDn)

    return pyCameraMat

def buildPyramidCloud(pyramidDepth, cameraMatrix):
    pyramidCameraMatrix= buildPyramidCameraMatrix(cameraMatrix, len(pyramidDepth)-1)
    
    pyramidCloud = []
    for i in range(len(pyramidDepth)):
        cloud = cv.rgbd.depthTo3d(pyramidDepth[i], pyramidCameraMatrix[i])
        pyramidCloud.append(cloud)

    return pyramidCloud

def normalComputer(cloud):
    row, col, ch = np.shape(cloud)
    ori = cloud[0:-1, 0:-1, :] 
    x_shift = cloud[0:-1, 1:, :]
    y_shift = cloud[1:, 0:-1, :]
    du = x_shift - ori
    dv = y_shift - ori
    cloud_cross = np.cross(du,dv)
    norm = np.linalg.norm(cloud_cross, axis=2)
    norms = np.repeat(norm[:, :, np.newaxis], 3, axis=2)
    normals = cloud_cross / norms 
    normals = np.pad(normals , ((0,1),(0,1),(0,0)), 'constant', constant_values = (0,0))

    return normals

def buildPyramidNormal(pyramidCloud):
    normals = normalComputer(pyramidCloud[0])
    level = len(pyramidCloud)
    pyramidNormal = buildPyramidImage(normals, level-1)

    # renormalize normals
    for i in range(level-1):
        currNormal = pyramidNormal[i+1]
        currNorm = np.linalg.norm(currNormal, axis=2)
        currNorms = np.repeat(currNorm[:, :, np.newaxis], 3, axis=2)
        pyramidNormal[i+1] = currNormal / currNorms 

    return pyramidNormal

def buildPyramidMask(pyramidDepth, minDepth, maxDepth, pyramidNormal):
    pyramidMask = []
    for i in range(len(pyramidDepth)):
        levelDepth = pyramidDepth[i]
        levelNormal = pyramidNormal[i]
        #levelMask = (levelDepth > minDepth) and (levelDepth < maxDepth)
        levelDepth = cv.patchNaNs(levelDepth, 0)
        minMask = levelDepth < minDepth
        maxMask = levelDepth > maxDepth
        nanMask = np.logical_not(np.isnan(levelDepth))
        levelMask = np.logical_or(levelDepth < minDepth, levelDepth > maxDepth)
        levelMask = np.logical_or(levelMask, nanMask)

        channelMask_0, channelMask_1, channelMask_2 = cv.split(levelNormal)
        channelMask_0 = np.isnan(channelMask_0)
        channelMask_1 = np.isnan(channelMask_1)
        channelMask_2 = np.isnan(channelMask_2)
        print(minMask)
        print(maxMask)
        print(nanMask)
        print(levelMask)
        print(levelNormal)
        print(np.shape(levelNormal))
        print(channelMask_0)
        print(np.shape(channelMask_0))
        print(validNormalMask)
        print(np.shape(validNormalMask))
        exit()



class Odometry:
    def __init__(self, odometryType):
        self.odometryType = odometryType
        self.initial_pose = sp.SE3()
        self.max_it = 100
    
    def setDefaultIterCounts(self):
        self.iterCounts = [7, 7, 7, 10]

    def setCameraMatrix(self, cameraMatrix):
        self.cameraMatrix = cameraMatrix

    def setDepthThreshold(self, minDepth, maxDepth):
        self.minDepth = minDepth
        self.maxDepth = maxDepth

    def compute(self, srcImage, srcDepth, srcMask, dstImage, dstDepth, dstMask):
        self.setDefaultIterCounts()
        levelCount = len(self.iterCounts)-1
        print('#level',levelCount)
        self.pyramidImageSrc = buildPyramidImage(srcImage, levelCount)
        self.pyramidImageDst = buildPyramidImage(dstImage, levelCount)
        self.pyramidDepthSrc = buildPyramidImage(srcDepth, levelCount)
        self.pyramidDepthDst = buildPyramidImage(dstDepth, levelCount)

        self.pyramidCloudSrc = buildPyramidCloud(self.pyramidDepthSrc, self.cameraMatrix)
        self.pyramidCloudDst = buildPyramidCloud(self.pyramidDepthDst, self.cameraMatrix)

        self.pyramidNormalDst = buildPyramidNormal(self.pyramidCloudDst)

        
        self.pyramidMaskDst = buildPyramidMask(self.pyramidDepthDst, self.minDepth, self.maxDepth, self.pyramidNormalDst)

        print(self.pyramidCloudSrc[0])
        print(np.shape(self.pyramidCloudSrc[0]))
        exit()

    def calc_residual(self, p3d, p2d, pose):
        fx = self.cameraMatrix[0][0]
        fy = self.cameraMatrix[1][1]
        cx = self.cameraMatrix[0][2]
        cy = self.cameraMatrix[1][2]
        residuals = np.zeros(np.shape(p3d)[0])
        res_std = []
        for j in range(np.shape(p3d)[0]):
            p3d_c = pose * p3d[j]
            px = fx * p3d_c[0] / p3d_c[2] + cx
            py = fy * p3d_c[1] / p3d_c[2] + cy
            proj = np.array([px, py])
            reproj_error = p2d[j] - proj
            residuals[j] = np.linalg.norm(reproj_error)
            if(np.isnan(p3d_c[2])==False):
                res_std.append(np.linalg.norm(reproj_error))
        return residuals, np.std(res_std)

    def BA_GN(self, p3d, p2d, it, pose, robust=None):
        fx = self.cameraMatrix[0][0]
        fy = self.cameraMatrix[1][1]
        cx = self.cameraMatrix[0][2]
        cy = self.cameraMatrix[1][2]
        last_cost = 0
        res, res_std = self.calc_residual(p3d, p2d, pose)
        huber_k = 1.345 * res_std
        if (robust=='Huber'):
            print('With Huber kernel')
            weight = []
            for i in res:
                if (i<=huber_k):
                    weight.append(1.0)
                else:
                    weight.append(huber_k/i)
        else:
            print('Without robust kernel')
            weight = np.ones(len(res))
        for i in range(it):
            H = np.zeros((6,6))
            b = np.zeros((6,))
            cost = 0
            for j in range(np.shape(p3d)[0]):
                p3d_c = pose * p3d[j]
                if(np.isnan(p3d_c[2])==False):
                    z_inv = 1.0 / p3d_c[2]
                    z2_inv = z_inv * z_inv
                    px = fx * p3d_c[0] / p3d_c[2] + cx
                    py = fy * p3d_c[1] / p3d_c[2] + cy
                    proj = np.array([px, py])
                    reproj_error = p2d[j] - proj
                    #print('3d:',p3d_c)
                    #print('proj 2d:',proj)
                    #print('reproj:',reproj_error)
                    #exit()
                    cost = cost + np.linalg.norm(reproj_error)
                    J = np.zeros((2,6))
                    J[0][0] = -fx * z_inv
                    J[0][1] = 0
                    J[0][2] = fx * p3d_c[0] * z2_inv
                    J[0][3] = fx * p3d_c[0] * p3d_c[1] * z2_inv
                    J[0][4] = -fx - fx * p3d_c[0] * p3d_c[0] * z2_inv
                    J[0][5] = fx * p3d_c[1] * z_inv
                    J[1][0] = 0
                    J[1][1] = -fy * z_inv
                    J[1][2] = fy * p3d_c[1] * z2_inv
                    J[1][3] = fy + fy * p3d_c[1] * p3d_c[1] * z2_inv
                    J[1][4] = -fy * p3d_c[0] * p3d_c[1] * z2_inv
                    J[1][5] = -fy * p3d_c[0] * z_inv
                    H = H + J.transpose() @ (J * weight[j])
                    b = b - J.transpose() @ (reproj_error * weight[j])

            #print('H',H)
            #print('b',b)
            dx = np.linalg.solve(H, b)
            print('iter:',i)
            print('dx:',dx)
            print('exp dx:',sp.SE3.exp(dx))
            print('Cost:',cost)
            if(np.isnan(dx[0])):
                print('NaN~~~')
                exit()
            if(i>0 and cost>last_cost):
                print('early terminate!!!  iter:',i)
                break

            pose = sp.SE3.exp(dx) * pose
            last_cost = cost

        return pose

    def process_feature(self, img_pre, img_cur):
        #img_pre = cv.undistort(img_pre, self.K, self.dist)
        #img_cur = cv.undistort(img_cur, self.K, self.dist)
        orb = cv.ORB_create()
        kp_pre, des_pre = orb.detectAndCompute(img_pre,None)
        kp_cur, des_cur = orb.detectAndCompute(img_cur,None)         
        bf = cv.BFMatcher(cv.NORM_HAMMING)   
        matches = bf.knnMatch(des_pre,des_cur,k=2)
        gmatches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                gmatches.append(m)
        print('matchs:',len(gmatches))
        points2D_pre = np.empty((0,2))
        points2D_cur = np.empty((0,2))
        kp_pre_match = []
        kp_cur_match = []
        for mat in gmatches:
            pre_idx = mat.queryIdx
            cur_idx = mat.trainIdx
            points2D_pre = np.vstack((points2D_pre,np.array(kp_pre[pre_idx].pt)))
            points2D_cur = np.vstack((points2D_cur,np.array(kp_cur[cur_idx].pt)))
            kp_pre_match.append(kp_pre[pre_idx])
            kp_cur_match.append(kp_cur[cur_idx])
        return points2D_pre, points2D_cur, kp_pre_match, kp_cur_match

    def compute_reproj(self, srcImage, srcDepth, srcMask, dstImage, dstDepth, dstMask):
        points2D_pre, points2D_cur, kp_pre, kp_cur = self.process_feature(srcImage, dstImage)
        fx = self.cameraMatrix[0][0]
        fy = self.cameraMatrix[1][1]
        cx = self.cameraMatrix[0][2]
        cy = self.cameraMatrix[1][2]
        #print(np.shape(points2D_cur))
        #print(np.shape(kp_cur))
        #print(type(points2D_cur))
        points3D = []
        for i, j in enumerate(points2D_pre):
            #print(kp_pre[i].pt)
            row = int(kp_pre[i].pt[1])
            col = int(kp_pre[i].pt[0])
            z = srcDepth[row][col]
            p3d_x = (j[0]-cx)*z/fx
            p3d_y = (j[1]-cy)*z/fy
            p3d = np.array([p3d_x, p3d_y, z])
            points3D.append(p3d)
        points3D = np.array(points3D)
        #print(np.shape(points3D))
        #print('pre 2d:',points2D_pre[0])
        #print('cur 2d:',points2D_cur[0])
        #print('pre 3d:',points3D[0])
        #BA_pose = self.BA_GN(points3D, points2D_cur, self.max_it, self.initial_pose)
        #BA_pose = self.BA_GN(points3D, points2D_cur, self.max_it, sp.SE3())
        #BA_pose = self.BA_GN(points3D, points2D_cur, self.max_it, sp.SE3(), 'Huber')
        BA_pose = self.BA_GN(points3D, points2D_cur, self.max_it, self.initial_pose, 'Huber')
        #retval, rvec_est, tvec_est, inliers = cv.solvePnPRansac(points3D, points2D_cur, self.cameraMatrix, None)
        #rvec_est =  R.from_rotvec(rvec_est.reshape(3,)).as_matrix()
        #BA_pose = sp.SE3(rvec_est, tvec_est)
        self.initial_pose = BA_pose
        #return BA_pose.matrix()
        return BA_pose.matrix()



if __name__ == '__main__':

    if(len(sys.argv) != 4):
        print(len(sys.argv))
        print('Argc is wrong!!!')
        exit()

    filename = sys.argv[1]

    dirpos = filename.rfind('/')
    dirname = filename[0:dirpos+1]

    if (filename.find('fr1')!=-1):
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
        #fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
        print('Set CameraMatrix Freiburg1 !!!')
    elif (filename.find('fr2')!=-1):
        fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
        print('Set CameraMatrix Freiburg2 !!!')
    else: 
        fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
        print('Set CameraMatrix Default !!!')

    frame_prev = RgbdFrame(np.array([]), np.array([]), np.array([]), np.array([]));
    frame_curr = RgbdFrame(np.array([]), np.array([]), np.array([]), np.array([]));

    odometry = cv.rgbd.Odometry_create(sys.argv[3]+'Odometry')
    odometry2 = Odometry(sys.argv[3]+'Odometry')
    print(odometry,' is created')
    cameraMatrix = np.eye(3,3)
    cameraMatrix[0][0] = fx 
    cameraMatrix[1][1] = fy 
    cameraMatrix[0][2] = cx 
    cameraMatrix[1][2] = cy 
    odometry.setCameraMatrix(cameraMatrix)
    odometry2.setCameraMatrix(cameraMatrix)
    odometry2.setDepthThreshold(10,100)

    Rts = []
    timestamps = []
    tt = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            s = line.split(' ')
            timestamp_rgb = s[0]
            img_rgb_path = dirname+s[1]
            timestamp_depth = s[2]
            img_depth_path = dirname+s[3]
     
            img_rgb = cv.imread(img_rgb_path)
            img_depth = cv.imread(img_depth_path, cv.IMREAD_ANYDEPTH).astype(np.float32)

            img_depth = img_depth / 5000.0
            depth_index = np.where(img_depth==0)
            img_depth[depth_index] = np.nan

            img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY);
            frame_curr.image = img_gray
            frame_curr.depth = img_depth
     
            if(len(Rts)!=0):
                #res, Rt = odometry.compute(frame_curr.image, frame_curr.depth, frame_curr.mask, frame_prev.image, frame_prev.depth, frame_prev.mask)        
                #res, Rt = odometry.compute(frame_curr.image, frame_curr.depth, None, frame_prev.image, frame_prev.depth, None)        
                Rt = odometry2.compute_reproj(frame_curr.image, frame_curr.depth, None, frame_prev.image, frame_prev.depth, None)        
                #odometry2.compute(frame_curr.image, frame_curr.depth, None, frame_prev.image, frame_prev.depth, None)

            if(len(Rts)==0):
                Rts.append(np.eye(4,4))
            else:
                prevRt = Rts[-1]
                print('Rt',Rt)
                Rts.append(np.dot(prevRt, Rt))

            timestamps.append(timestamp_rgb)

            frame_copy(frame_prev,frame_curr)

            
            print('frame',tt)
            tt = tt+1
            #if (tt==2):
            #    break
    write_results(sys.argv[2], timestamps, Rts)
