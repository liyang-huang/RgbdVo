import sys
import numpy as np
import cv2 as cv
from dataclasses import dataclass
import math
import sophus as sp
from scipy.spatial.transform import Rotation as R
from BundleAdjustment import BundleAdjustment
import g2o
import time

#np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore', invalid='ignore')

@dataclass
class RgbdFrame:
    image: np.array
    depth: np.array
    mask: np.array
    normals: np.array

@dataclass
class CameraParameter:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float=0.0

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
        levelDepth = cv.patchNaNs(levelDepth, 0)
        minMask = levelDepth > minDepth
        maxMask = levelDepth < maxDepth
        levelMask = np.logical_and(minMask, maxMask)

        levelNormal = pyramidNormal[i]
        validNormalMask = levelNormal==levelNormal
        #channelMask_0, channelMask_1, channelMask_2 = cv.split(validNormalMask)
        channelMask_0 = validNormalMask[:,:,0]
        channelMask_1 = validNormalMask[:,:,1]
        channelMask_2 = validNormalMask[:,:,2]
        channelMask_01 = np.logical_and(channelMask_0, channelMask_1)
        validNormalMask = np.logical_and(channelMask_01, channelMask_2)
        levelMask = np.logical_and(levelMask, validNormalMask)
        
        pyramidMask.append(levelMask)
    return pyramidMask

def buildPyramidSobel(pyramidImage, dx, dy, sobelSize):
    pyramidSobel = []
    for i in range(len(pyramidImage)):
        levelImage = pyramidImage[i]
        levelSobel = cv.Sobel(levelImage, cv.CV_16S, dx, dy, sobelSize)
        pyramidSobel.append(levelSobel)
    
    return pyramidSobel

def randomSubsetOfMask(mask, part):
    minPointsCount = 1000
    nonzeros = np.sum(mask)
    mask_total = np.shape(mask)[0] * np.shape(mask)[1]
    needCount = np.max((minPointsCount, int(mask_total * part)))
    if (needCount < nonzeros):
        subsetSize = 0
        subset = np.zeros(np.shape(mask), dtype=np.bool)
        while (subsetSize < needCount):
            y = np.random.randint(np.shape(mask)[0])
            x = np.random.randint(np.shape(mask)[1])
            if (mask[y][x] == True):
                subset[y][x] = True
                mask[y][x] = False
                subsetSize = subsetSize + 1
        else:
            mask_final = subset
    else:
        mask_final = mask

    return mask_final

def buildPyramidTexturedMask(pyramid_dI_dx, pyramid_dI_dy, minGradientMagnitudes, pyramidMaskDst, maxPointsPart, sobelScale):
    sobelScale2_inv = 1.0 / (sobelScale * sobelScale)
    pyramidTexturedMask = []
    for i in range(len(pyramidMaskDst)):
        minScaledGradMagnitude2 = minGradientMagnitudes[i] * minGradientMagnitudes[i] * sobelScale2_inv
        leveldx = pyramid_dI_dx[i]
        leveldy = pyramid_dI_dy[i]
        levelpyramidMask = pyramidMaskDst[i]
        magnitude2 = np.square(leveldx, dtype=np.int32) + np.square(leveldy, dtype=np.int32)
        texturedMask = magnitude2 >= minScaledGradMagnitude2
        texturedMask_final = np.logical_and(texturedMask, levelpyramidMask)
         
        texturedMask_random = randomSubsetOfMask(texturedMask_final, maxPointsPart)
        pyramidTexturedMask.append(texturedMask_random)

    return pyramidTexturedMask

def buildPyramidNormalsMask(pyramidNormals, pyramidMask, maxPointsPart):
    pyramidNormalsMask = []
    for i in range(len(pyramidNormals)):
        levelMask = pyramidMask[i]
        levelMask_random = randomSubsetOfMask(levelMask, maxPointsPart)
        pyramidNormalsMask.append(levelMask_random)

    return pyramidNormalsMask


def computeCorresps(K, T, depth0, validMask0, depth1, selectMask1, maxDepthDiff):
    K_inv = np.linalg.inv(K)
    R = T.matrix()[:3,:3]
    t = T.matrix()[:3,3]
    Kt = K @ t
    KRK_inv = K @ R @ K_inv

    rows = np.shape(depth0)[0]
    cols = np.shape(depth0)[1]
    KRK_inv0_u1 = np.empty(cols)
    KRK_inv3_u1 = np.empty(cols)
    KRK_inv6_u1 = np.empty(cols)
    KRK_inv1_v1_plus_KRK_inv2 = np.empty(rows)
    KRK_inv4_v1_plus_KRK_inv5 = np.empty(rows)
    KRK_inv7_v1_plus_KRK_inv8 = np.empty(rows)
    for u1 in range(cols):
        KRK_inv0_u1[u1] = KRK_inv[0][0] * u1
        KRK_inv3_u1[u1] = KRK_inv[1][0] * u1
        KRK_inv6_u1[u1] = KRK_inv[2][0] * u1
    for v1 in range(rows):
        KRK_inv1_v1_plus_KRK_inv2[v1] = (KRK_inv[0][1] * v1 + KRK_inv[0][2])
        KRK_inv4_v1_plus_KRK_inv5[v1] = (KRK_inv[1][1] * v1 + KRK_inv[1][2])
        KRK_inv7_v1_plus_KRK_inv8[v1] = (KRK_inv[2][1] * v1 + KRK_inv[2][2])

    corresps = np.full((rows, cols, 2), -1)
    correspCount = 0
    for v1 in range(rows):
        for u1 in range(cols):
            if (selectMask1[v1][u1] == True):
                d1 = depth1[v1][u1]
                transformed_d1 = d1 * (KRK_inv6_u1[u1] + KRK_inv7_v1_plus_KRK_inv8[v1]) + Kt[2]
                if (transformed_d1 > 0):
                    transformed_d1_inv = 1.0 / transformed_d1
                    u0 = round(transformed_d1_inv * (d1 * (KRK_inv0_u1[u1] + KRK_inv1_v1_plus_KRK_inv2[v1]) + Kt[0]))
                    v0 = round(transformed_d1_inv * (d1 * (KRK_inv3_u1[u1] + KRK_inv4_v1_plus_KRK_inv5[v1]) + Kt[1]))
                    if ((u0 < cols) and (v0 < rows) and (u0 >= 0) and (v0 >= 0)):
                        d0 = depth0[v0][u0]
                        if ((validMask0[v0][u0] == True) and (np.abs(transformed_d1 - d0) <= maxDepthDiff)):
                            c = corresps[v0][u0]
                            if (c[0] != -1):
                                exist_u1 = c[0]
                                exist_v1 = c[1]
                                exist_d1 = depth1[exist_v1][exist_u1] * (KRK_inv6_u1[exist_u1] + KRK_inv7_v1_plus_KRK_inv8[exist_v1]) + Kt[2]
                                if(transformed_d1 > exist_d1):
                                    continue
                            else:
                                correspCount = correspCount + 1
                            corresps[v0][u0][0] = u1
                            corresps[v0][u0][1] = v1
    corresps_final = []
    for v0 in range(rows):
        for u0 in range(cols):
            c = corresps[v0][u0]
            if (c[0] != -1):
                corresps_final.append([u0, v0, c[0], c[1]])

    #print('correspCount',correspCount)
    #print(np.shape(corresps_final))
    return corresps_final, correspCount

def calcRgbdLsmMatrices(image0, cloud0, T, image1, dI_dx1, dI_dy1, corresps, fx, fy, sobelScaleIn):
    res = []
    sigma = 0
    R = T.matrix()[:3,:3]
    t = T.matrix()[:3,3]
    for i in corresps:
        u0 = i[0]   
        v0 = i[1]   
        u1 = i[2]   
        v1 = i[3]  
        diffs = int(image0[v0][u0]) - int(image1[v1][u1])
        sigma = sigma + diffs * diffs
        res.append(diffs)
    sigma_final = np.sqrt(sigma / len(corresps))
    
    H = np.zeros((6,6))
    b = np.zeros((6,))
    for j, i in enumerate(corresps):
        u0 = i[0]   
        v0 = i[1]   
        u1 = i[2]   
        v1 = i[3]
        w = sigma_final + np.abs(res[j])
        w_tmp = 1.0 / w if w > np.finfo(np.float32).eps else 1.0
        w_sobelScale = w_tmp * sobelScaleIn
        p0 = cloud0[v0][u0]
        Tp0 = R @ p0 + t
        dIdx = dI_dx1[v1][u1] * w_sobelScale
        dIdy = dI_dy1[v1][u1] * w_sobelScale
        invz = 1.0 / Tp0[2]
        c0 = dIdx * fx * invz
        c1 = dIdy * fy * invz
        c2 = -(c0 * Tp0[0] + c1 * Tp0[1]) * invz
        A = np.empty(6)
        A[3] = -Tp0[2] * c1 + Tp0[1] * c2
        A[4] =  Tp0[2] * c0 - Tp0[0] * c2
        A[5] = -Tp0[1] * c0 + Tp0[0] * c1
        A[0] = c0
        A[1] = c1
        A[2] = c2
        #H = H + A.transpose() @ A
        #b = b + A.transpose() * w_tmp * res[j]
        #for i in range(6):
        #    for j in range(6):
        #        H[i][j] = H[i][j] + A[i] * A[j]
        #    b[i] = b[i] + A[i] * w_tmp * res[j]
        A1 = np.vstack((A,A,A,A,A,A))
        A_inv = A.reshape(6,1)
        A2 = np.hstack((A_inv,A_inv,A_inv,A_inv,A_inv,A_inv))
        H = H + A1 * A2
        b = b + A * w_tmp * res[j]
    #for i in range(6):
    #    for j in range(i+1, 6):
    #        H[j][i] = H[i][j]
    return H, b

def calcICPLsmMatrices(cloud0, T, cloud1, normals1, corresps):
    res = []
    tps0 = []
    sigma = 0
    R = T.matrix()[:3,:3]
    t = T.matrix()[:3,3]
    for i in corresps:
        u0 = i[0]   
        v0 = i[1]   
        u1 = i[2]   
        v1 = i[3]
        p0 = cloud0[v0][u0]  
        Tp0 = R @ p0 + t
        n1 = normals1[v1][u1]
        v = cloud1[v1][u1] - Tp0
        diffs = n1[0] * v[0] + n1[1] * v[1] + n1[2] * v[2]
        sigma = sigma + diffs * diffs
        res.append(diffs)
        tps0.append(Tp0)
    sigma_final = np.sqrt(sigma / len(corresps))
    
    H = np.zeros((6,6))
    b = np.zeros((6,))
    for j, i in enumerate(corresps):
        u1 = i[2]   
        v1 = i[3]
        w = sigma_final + np.abs(res[j])
        w_tmp = 1.0 / w if w > np.finfo(np.float32).eps else 1.0
        wn1 = normals1[v1][u1] * w_tmp
        p0 = tps0[j]
        A = np.empty(6)
        A[3] = -p0[2] * wn1[1] + p0[1] * wn1[2]
        A[4] =  p0[2] * wn1[0] - p0[0] * wn1[2]
        A[5] = -p0[1] * wn1[0] + p0[0] * wn1[1]
        A[0] = wn1[0]
        A[1] = wn1[1]
        A[2] = wn1[2]
        A1 = np.vstack((A,A,A,A,A,A))
        A_inv = A.reshape(6,1)
        A2 = np.hstack((A_inv,A_inv,A_inv,A_inv,A_inv,A_inv))
        H = H + A1 * A2
        b = b + A * w_tmp * res[j]
        
    return H, b

class Odometry:
    def __init__(self, odometryType):
        self.odometryType = odometryType
        self.initial_pose = sp.SE3()
        self.max_it = 100
        self.sobelSize = 3
        self.sobelScale = 1.0 / 8.0
        self.maxPointsPart = 0.07 #in [0, 1] 
        self.maxDepthDiff = 0.07 #in meters
    
    def setDefaultIterCounts(self):
        self.iterCounts = [7, 7, 7, 10]

    def setDefaultMinGradientMagnitudes(self):
        self.minGradientMagnitudes  = [10, 10, 10, 10]

    def setCameraMatrix(self, cameraMatrix):
        self.cameraMatrix = cameraMatrix

    def setDepthThreshold(self, minDepth, maxDepth):
        self.minDepth = minDepth
        self.maxDepth = maxDepth

    def compute(self, srcImage, srcDepth, srcMask, dstImage, dstDepth, dstMask):
        self.setDefaultIterCounts()
        self.setDefaultMinGradientMagnitudes()
        levelCount = len(self.iterCounts)-1
        print('#level',levelCount)
        self.pyramidImageSrc = buildPyramidImage(srcImage, levelCount)
        self.pyramidImageDst = buildPyramidImage(dstImage, levelCount)
        self.pyramidDepthSrc = buildPyramidImage(srcDepth, levelCount)
        self.pyramidDepthDst = buildPyramidImage(dstDepth, levelCount)

        self.pyramidCloudSrc = buildPyramidCloud(self.pyramidDepthSrc, self.cameraMatrix)
        self.pyramidCloudDst = buildPyramidCloud(self.pyramidDepthDst, self.cameraMatrix)

        self.pyramidNormalSrc = buildPyramidNormal(self.pyramidCloudSrc)
        self.pyramidNormalDst = buildPyramidNormal(self.pyramidCloudDst)
        
        self.pyramidMaskSrc = buildPyramidMask(self.pyramidDepthSrc, self.minDepth, self.maxDepth, self.pyramidNormalSrc)
        self.pyramidMaskDst = buildPyramidMask(self.pyramidDepthDst, self.minDepth, self.maxDepth, self.pyramidNormalDst)

        self.pyramid_dI_dx = buildPyramidSobel(self.pyramidImageDst, 1, 0, self.sobelSize)
        self.pyramid_dI_dy = buildPyramidSobel(self.pyramidImageDst, 0, 1, self.sobelSize)

        self.pyramidTexturedMask = buildPyramidTexturedMask(self.pyramid_dI_dx, self.pyramid_dI_dy, self.minGradientMagnitudes, self.pyramidMaskDst, self.maxPointsPart, self.sobelScale)
        self.pyramidNormalsMask = buildPyramidNormalsMask(self.pyramidNormalDst, self.pyramidMaskDst, self.maxPointsPart)

        pyramidCameraMatrix= buildPyramidCameraMatrix(self.cameraMatrix, levelCount)

        minCorrespsCount = 20 * 6
        pose_tmp = self.initial_pose
        tt1 = time.time()
        for level in range(levelCount, 0-1, -1):
            levelCameraMatrix = pyramidCameraMatrix[level]
            srcLevelDepth = self.pyramidDepthSrc[level]
            dstLevelDepth = self.pyramidDepthDst[level]
            fx = levelCameraMatrix[0][0]
            fy = levelCameraMatrix[1][1]
            

            for i in range(self.iterCounts[level]):
                t1 = time.time()
                corresps_rgbd, corresps_rgbd_count = computeCorresps(levelCameraMatrix, pose_tmp.inverse(), srcLevelDepth, self.pyramidMaskSrc[level], dstLevelDepth, self.pyramidTexturedMask[level], self.maxDepthDiff)
                t2 = time.time()
                corresps_icp, corresps_icp_count = computeCorresps(levelCameraMatrix, pose_tmp.inverse(), srcLevelDepth, self.pyramidMaskSrc[level], dstLevelDepth, self.pyramidNormalsMask[level], self.maxDepthDiff)
                if (corresps_rgbd_count < minCorrespsCount):
                    break
                t3 = time.time()
                AtA_rgbd, AtB_rgbd = calcRgbdLsmMatrices(self.pyramidImageSrc[level], self.pyramidCloudSrc[level], pose_tmp, self.pyramidImageDst[level], self.pyramid_dI_dx[level], self.pyramid_dI_dy[level], corresps_rgbd, fx, fy, self.sobelScale) 
                t4 = time.time()
                AtA_icp, AtB_icp = calcICPLsmMatrices(self.pyramidCloudSrc[level], pose_tmp, self.pyramidCloudDst[level], self.pyramidNormalDst[level], corresps_icp)
                #dx = np.linalg.solve(AtA_rgbd, AtB_rgbd)
                H = AtA_rgbd + AtA_icp
                b = AtB_rgbd + AtB_icp
                dx = np.linalg.solve(H, b)
                pose_tmp = sp.SE3.exp(dx) * pose_tmp
                #print('cor:',t2-t1)
                #print('lsm:',t4-t3)
            #print(levelCameraMatrix)
            #exit()
        #tt2 = time.time()
        #print('total:',tt2-tt1)
        #exit()

        self.initial_pose = pose_tmp
        return pose_tmp.matrix()

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
        return BA_pose.matrix()

    def compute_reproj_g2o(self, srcImage, srcDepth, srcMask, dstImage, dstDepth, dstMask):
        pose_pre = g2o.SE3Quat(sp.SE3().matrix()[:3,:3], sp.SE3().matrix()[:3,3])
        pose_cur = g2o.SE3Quat(self.initial_pose.matrix()[:3,:3], self.initial_pose.matrix()[:3,3])
        cam = CameraParameter(self.cameraMatrix[0][0], self.cameraMatrix[1][1], self.cameraMatrix[0][2], self.cameraMatrix[1][2])
        BA = BundleAdjustment()
        BA.add_pose(0, pose_pre, cam, fixed=True )
        BA.add_pose(1, pose_cur, cam, fixed=False )

        points2D_pre, points2D_cur, kp_pre, kp_cur = self.process_feature(srcImage, dstImage)
        points3D = []
        for i, j in enumerate(points2D_pre):
            row = int(kp_pre[i].pt[1])
            col = int(kp_pre[i].pt[0])
            z = srcDepth[row][col]
            p3d_x = (j[0]-cx)*z/fx
            p3d_y = (j[1]-cy)*z/fy
            p3d = np.array([p3d_x, p3d_y, z])
            points3D.append(p3d)
        points3D = np.array(points3D)

        for i in range(np.shape(points3D)[0]):
            p3d = points3D[i]
            if(np.isnan(p3d[2])==False):
                print(p3d)
                BA.add_point(i, points3D[i], fixed=True)
                BA.add_edge(i, 0, i, points2D_pre[i])
                BA.add_edge(i, 1, i+1, points2D_cur[i])

        BA.optimize(max_iterations = self.max_it)
        BA_pose = sp.SE3(BA.get_pose(1).to_homogeneous_matrix())
        self.initial_pose = BA_pose
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
    odometry2.setDepthThreshold(0,4.0)

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
                #Rt = odometry2.compute_reproj(frame_curr.image, frame_curr.depth, None, frame_prev.image, frame_prev.depth, None)        
                #Rt = odometry2.compute_reproj_g2o(frame_curr.image, frame_curr.depth, None, frame_prev.image, frame_prev.depth, None)        
                Rt = odometry2.compute(frame_curr.image, frame_curr.depth, None, frame_prev.image, frame_prev.depth, None)

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
