import sys
import numpy as np
import cv2 as cv
from dataclasses import dataclass
import math

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


class Odometry:
    def __init__(self, odometryType):
        self.odometryType = odometryType
    
    def setDefaultIterCounts(self):
        self.iterCounts = [7, 7, 7, 10]

    def setCameraMatrix(self,cameraMatrix):
        self.cameraMatrix = cameraMatrix

    def compute(self, srcImage, srcDepth, srcMask, dstImage, dstDepth, dstMask):
        self.setDefaultIterCounts()
        levelCount = len(self.iterCounts)-1
        print(levelCount)
        self.pyramidImageSrc = buildPyramidImage(srcImage, levelCount)
        self.pyramidImageDst = buildPyramidImage(dstImage, levelCount)
        self.pyramidDepthSrc = buildPyramidImage(srcDepth, levelCount)
        self.pyramidDepthDst = buildPyramidImage(dstDepth, levelCount)

        self.pyramidCloudSrc = buildPyramidCloud(self.pyramidDepthSrc, self.cameraMatrix)
        self.pyramidCloudDst = buildPyramidCloud(self.pyramidDepthDst, self.cameraMatrix)

        print(self.pyramidCloudSrc[0])
        print(np.shape(self.pyramidCloudSrc[0]))
        exit()




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
                res, Rt = odometry.compute(frame_curr.image, frame_curr.depth, None, frame_prev.image, frame_prev.depth, None)        
                odometry2.compute(frame_curr.image, frame_curr.depth, None, frame_prev.image, frame_prev.depth, None)

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
            if (tt==2):
                break
    write_results(sys.argv[2], timestamps, Rts)
