#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <gmpxx.h>
#include <gmp.h>

#include "vo.hpp"
#include "fixed_point_util.hpp"

using namespace cv;
using namespace std;
//
const int sign = 1;
const int bit_width = 61; //have to less than FIXP_INT_SCALAR_TYPE?
const int shift = 24;
const int shift_half = 12;
const FixedPointConfig fpconfig(sign, bit_width, shift);

const int sobelSize = 3;
const double sobelScale = 1./8.;

static inline
void setDefaultIterCounts(Mat& iterCounts)
{
    iterCounts = Mat(Vec4i(7,7,7,10));
}

static inline
void setDefaultMinGradientMagnitudes(Mat& minGradientMagnitudes)
{
    minGradientMagnitudes = Mat(Vec4f(10,10,10,10));
}
/*
static
//vector<FixedPointVector> normalsComputer(const vector<FixedPointVector>& p3d_vec, int rows, int cols) 
vector<FixedPointVector> normalsComputer(vector<FixedPointVector>& p3d_vec, int rows, int cols) 
{
  FixedPointScalar zero_value((FIXP_SCALAR_TYPE)0, fpconfig);
  //FixedPointScalar zero_value((FIXP_SCALAR_TYPE)std::numeric_limits<double>::quiet_NaN(), fpconfig);
  FixedPointVector zero_vec(zero_value, zero_value, zero_value);
  vector<FixedPointVector> normals(rows*cols, zero_vec);
  for (int y = 0; y < rows - 1; ++y)
  {
    for (int x = 0; x < cols - 1; ++x)
    {
        //if(!(cvIsNaN(p3d_vec[y*cols + x].x.value_floating) || cvIsNaN(p3d_vec[y*cols + (x+1)].x.value_floating) || cvIsNaN(p3d_vec[(y+1)*cols + x].x.value_floating)))
        {
            FixedPointScalar du_x = p3d_vec[y*cols + (x+1)].x - p3d_vec[y*cols + x].x;
            FixedPointScalar dv_x = p3d_vec[(y+1)*cols + x].x - p3d_vec[y*cols + x].x;
            FixedPointScalar du_y = p3d_vec[y*cols + (x+1)].y - p3d_vec[y*cols + x].y;
            FixedPointScalar dv_y = p3d_vec[(y+1)*cols + x].y - p3d_vec[y*cols + x].y;
            FixedPointScalar du_z = p3d_vec[y*cols + (x+1)].z - p3d_vec[y*cols + x].z;
            FixedPointScalar dv_z = p3d_vec[(y+1)*cols + x].z - p3d_vec[y*cols + x].z;
            FixedPointScalar n_x = (du_y * dv_z) - (du_z * dv_y);
            FixedPointScalar n_y = (du_z * dv_x) - (du_x * dv_z);
            FixedPointScalar n_z = (du_x * dv_y) - (du_y * dv_x);
            FixedPointScalar n2_x = n_x*n_x;
            FixedPointScalar n2_y = n_y*n_y;
            FixedPointScalar n2_z = n_z*n_z;
            FixedPointScalar norm_pre = n2_x + n2_y + n2_z;
            FixedPointScalar norm = (norm_pre).sqrt();
            //if(!cvIsNaN(norm.value_floating))
            if(!(norm.value==0))
            {
                FixedPointScalar n_x_final = n_x / norm;
                cout << "n.x: " << n_x.value_floating << endl;
                cout << "n.y: " << n_y.value_floating << endl;
                cout << "n.z: " << n_z.value_floating << endl;
                //cout << "norm+++: " << norm.value_floating << endl;
                //cout << "n_final.x: " << n_x_final.value_floating << endl;
                FixedPointScalar n_y_final = n_y / norm;
                FixedPointScalar n_z_final = n_z / norm;
                FixedPointVector normal(n_x_final, n_y_final, n_z_final);   
                normals[y*cols + x] = normal;
                //cout << "n_final.y: " << n_y_final.value_floating << endl;
                //cout << "n_final.z: " << n_z_final.value_floating << endl;
                //cout << "liyang test"<< norm_pre.value_floating << endl;
            }
            if((x==590)&&(y==478))
            {
                cout << "norm: " << norm.value_floating << endl;
                cout << "norm: " << norm.value << endl;
                cout << "norm_pre: " << norm_pre.value_floating << endl;
                cout << "norm_pre: " << norm_pre.value << endl;
                cout << "n.x: " << n_x.value_floating << endl;
                cout << "n.y: " << n_y.value_floating << endl;
                cout << "n.z: " << n_z.value_floating << endl;
                //cout << "n_final.x: " << n_x_final.value_floating << endl;
                //cout << "n_final.y: " << n_y_final.value_floating << endl;
                //cout << "n_final.z: " << n_z_final.value_floating << endl;
                cout << normals[y*cols + x].x.value_floating << endl;
                cout << normals[y*cols + x].y.value_floating << endl;
                cout << normals[y*cols + x].z.value_floating << endl;
                cout << p3d_vec[y*cols + x].x.value_floating << endl;
                cout << p3d_vec[y*cols + x].y.value_floating << endl;
                cout << p3d_vec[y*cols + x].z.value_floating << endl;
                cout << "du.x: " << du_x.value_floating << endl;
                cout << "du.y: " << du_y.value_floating << endl;
                cout << "du.z: " << du_z.value_floating << endl;
                cout << "dv.x: " << dv_x.value_floating << endl;
                cout << "dv.y: " << dv_y.value_floating << endl;
                cout << "dv.z: " << dv_z.value_floating << endl;
                exit(1);
            }
        }
    }
  }
  return normals;
}
*/
static
void normalsComputer(const Mat& points3d, int rows, int cols, Mat & normals) 
{
  normals.create(points3d.size(), CV_MAKETYPE(points3d.depth(), 3));
  //for (int y = 0; y < rows - 1; ++y)
  for (int y = 0; y < rows ; ++y)
  {
    //for (int x = 0; x < cols - 1; ++x)
    for (int x = 0; x < cols ; ++x)
    {
    	Vec3f du = points3d.at<Vec3f>(y,x+1) - points3d.at<Vec3f>(y,x);
    	Vec3f dv = points3d.at<Vec3f>(y+1,x) - points3d.at<Vec3f>(y,x);
        //normals.at<Vec3f>(y,x) = du.cross(dv);
        normals.at<Vec3f>(y,x)[0] = (du[1]*dv[2]) - (du[2]*dv[1]);
        normals.at<Vec3f>(y,x)[1] = (du[2]*dv[0]) - (du[0]*dv[2]);
        normals.at<Vec3f>(y,x)[2] = (du[0]*dv[1]) - (du[1]*dv[0]);
    	float norm = sqrt(normals.at<Vec3f>(y,x)[0]*normals.at<Vec3f>(y,x)[0] + normals.at<Vec3f>(y,x)[1]*normals.at<Vec3f>(y,x)[1] +normals.at<Vec3f>(y,x)[2]*normals.at<Vec3f>(y,x)[2]);
        normals.at<Vec3f>(y,x)[0] = normals.at<Vec3f>(y,x)[0] / norm;
        normals.at<Vec3f>(y,x)[1] = normals.at<Vec3f>(y,x)[1] / norm;
        normals.at<Vec3f>(y,x)[2] = normals.at<Vec3f>(y,x)[2] / norm;
        //if((x==590)&&(y==478))
        //{
        //    cout << "norm: " << norm << endl;
        //    cout << normals.at<Vec3f>(y,x)[0] << endl;
        //    cout << normals.at<Vec3f>(y,x)[1] << endl;
        //    cout << normals.at<Vec3f>(y,x)[2] << endl;
        //    cout << points3d.at<Vec3f>(y,x)[0] << endl;
        //    cout << points3d.at<Vec3f>(y,x)[1] << endl;
        //    cout << points3d.at<Vec3f>(y,x)[2] << endl;
        //    cout << points3d.at<Vec3f>(y,x+1)[0] << endl;
        //    cout << points3d.at<Vec3f>(y,x+1)[1] << endl;
        //    cout << points3d.at<Vec3f>(y,x+1)[2] << endl;
        //    cout << points3d.at<Vec3f>(y+1,x)[0] << endl;
        //    cout << points3d.at<Vec3f>(y+1,x)[1] << endl;
        //    cout << points3d.at<Vec3f>(y+1,x)[2] << endl;
        //    cout << "du.x: " << du[0] << endl;
        //    cout << "du.y: " << du[1] << endl;
        //    cout << "du.z: " << du[2] << endl;
        //    cout << "dv.x: " << dv[0] << endl;
        //    cout << "dv.y: " << dv[1] << endl;
        //    cout << "dv.z: " << dv[2] << endl;
        //    exit(1);
        //}
        if((x==cols-1)||(y==rows-1))
        {
           normals.at<Vec3f>(y,x)[0] = 0.0;
           normals.at<Vec3f>(y,x)[1] = 0.0;
           normals.at<Vec3f>(y,x)[2] = 0.0;
        }
    }
  }
}

RgbdFrame::RgbdFrame() : ID(-1)
{}

RgbdFrame::RgbdFrame(const Mat& image_in, const Mat& depth_in, const Mat& mask_in, const Mat& normals_in, int ID_in)
    : ID(ID_in), image(image_in), depth(depth_in), mask(mask_in), normals(normals_in)
{}

RgbdFrame::~RgbdFrame()
{}

void RgbdFrame::release()
{
    ID = -1;
    image.release();
    depth.release();
    mask.release();
    normals.release();
}

OdometryFrame::OdometryFrame() : RgbdFrame()
{}

OdometryFrame::OdometryFrame(const Mat& image_in, const Mat& depth_in, const Mat& mask_in, const Mat& normals_in, int ID_in)
    : RgbdFrame(image_in, depth_in, mask_in, normals_in, ID_in)
{}

void OdometryFrame::release()
{
    RgbdFrame::release();
    releasePyramids();
}

void OdometryFrame::releasePyramids()
{
    pyramidImage.clear();
    pyramidDepth.clear();
    pyramidMask.clear();

    pyramidCloud.clear();

    pyramid_dI_dx.clear();
    pyramid_dI_dy.clear();
    pyramidTexturedMask.clear();

    pyramidNormals.clear();
    pyramidNormalsMask.clear();
}


Odometry::Odometry() :
    minDepth(DEFAULT_MIN_DEPTH()),
    maxDepth(DEFAULT_MAX_DEPTH()),
    maxDepthDiff(DEFAULT_MAX_DEPTH_DIFF()),
    maxPointsPart(DEFAULT_MAX_POINTS_PART()),
    maxTranslation(DEFAULT_MAX_TRANSLATION()),
    maxRotation(DEFAULT_MAX_ROTATION())

{
    setDefaultIterCounts(iterCounts);
    setDefaultMinGradientMagnitudes(minGradientMagnitudes);
}

Odometry::Odometry(const Mat& _cameraMatrix,
                   float _minDepth, float _maxDepth, float _maxDepthDiff,
                   const std::vector<int>& _iterCounts,
                   const std::vector<float>& _minGradientMagnitudes,
                   float _maxPointsPart) :
                   minDepth(_minDepth), maxDepth(_maxDepth), maxDepthDiff(_maxDepthDiff),
                   iterCounts(Mat(_iterCounts).clone()),
                   minGradientMagnitudes(Mat(_minGradientMagnitudes).clone()),
                   maxPointsPart(_maxPointsPart),
                   cameraMatrix(_cameraMatrix),
                   maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    if(iterCounts.empty() || minGradientMagnitudes.empty())
    {
        setDefaultIterCounts(iterCounts);
        setDefaultMinGradientMagnitudes(minGradientMagnitudes);
    }
}

static
void preparePyramidImage(const Mat& image, std::vector<Mat>& pyramidImage, size_t levelCount)
{
    if(!pyramidImage.empty())
    {
        if(pyramidImage.size() < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidImage has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidImage[0].size() == image.size());
        for(size_t i = 0; i < pyramidImage.size(); i++)
            CV_Assert(pyramidImage[i].type() == image.type());
    }
    else
        buildPyramid(image, pyramidImage, (int)levelCount - 1);
}

static
void buildpy(const Mat& depth, std::vector<FixedPointMatrix>& pyramidDepth, size_t levelCount)
{
    if(pyramidDepth.empty())
    {
        Mat tmp;
        for( int i = 0; i < levelCount; i++ )
        {   
            Mat dwn;
            vector<FixedPointScalar> depth_fixp;
            if(i != 0)
            {
                pyrDown(tmp, dwn);
                depth_fixp = f_Mat2Vec(dwn, fpconfig);
                FixedPointMatrix depth_mtx(depth_fixp, dwn.rows, dwn.cols);
                pyramidDepth.push_back(depth_mtx);
                tmp = dwn;
            }
            else
            {
                depth_fixp = f_Mat2Vec(depth, fpconfig);
                FixedPointMatrix depth_mtx(depth_fixp, depth.rows, depth.cols);
                pyramidDepth.push_back(depth_mtx);
                tmp = depth;
            }

        }
    }
}

static
void preparePyramidDepth(const Mat& depth, std::vector<Mat>& pyramidDepth, size_t levelCount)
{
    if(!pyramidDepth.empty())
    {
        if(pyramidDepth.size() < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidDepth has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidDepth[0].size() == depth.size());
        for(size_t i = 0; i < pyramidDepth.size(); i++)
            CV_Assert(pyramidDepth[i].type() == depth.type());
    }
    else
        buildPyramid(depth, pyramidDepth, (int)levelCount - 1);
}

static
void buildPyramidCameraMatrix(const Mat& cameraMatrix, int levels, std::vector<Mat>& pyramidCameraMatrix)
{
    pyramidCameraMatrix.resize(levels);

    Mat cameraMatrix_dbl;
    //cameraMatrix.convertTo(cameraMatrix_dbl, CV_64FC1);
    cameraMatrix.convertTo(cameraMatrix_dbl, CV_32FC1);

    for(int i = 0; i < levels; i++)
    {
        Mat levelCameraMatrix = i == 0 ? cameraMatrix_dbl : 0.5f * pyramidCameraMatrix[i-1];
        //levelCameraMatrix.at<double>(2,2) = 1.;
        levelCameraMatrix.at<float>(2,2) = 1.;
        pyramidCameraMatrix[i] = levelCameraMatrix;
    }
}

inline void
rescaleDepth(InputArray in_in, int depth, OutputArray out_out)
{
  cv::Mat in = in_in.getMat();
  CV_Assert(in.type() == CV_64FC1 || in.type() == CV_32FC1 || in.type() == CV_16UC1 || in.type() == CV_16SC1);
  CV_Assert(depth == CV_64FC1 || depth == CV_32FC1);

  int in_depth = in.depth();

  out_out.create(in.size(), depth);
  cv::Mat out = out_out.getMat();
  if (in_depth == CV_16U)
  {
    in.convertTo(out, depth, 1 / 1000.0); //convert to float so that it is in meters
    cv::Mat valid_mask = in == std::numeric_limits<ushort>::min(); // Should we do std::numeric_limits<ushort>::max() too ?
    out.setTo(std::numeric_limits<float>::quiet_NaN(), valid_mask); //set a$
  }
  if (in_depth == CV_16S)
  {
    in.convertTo(out, depth, 1 / 1000.0); //convert to float so tha$
    cv::Mat valid_mask = (in == std::numeric_limits<short>::min()) | (in == std::numeric_limits<short>::max()); // Should we do std::numeric_limits<ushort>::max() too ?
    out.setTo(std::numeric_limits<float>::quiet_NaN(), valid_mask); //set a$
  }
  if ((in_depth == CV_32F) || (in_depth == CV_64F))
    in.convertTo(out, depth);
}

template<typename T>
void
rescaleDepthTemplated(const Mat& in, Mat& out);

template<>
inline void
rescaleDepthTemplated<float>(const Mat& in, Mat& out)
{
  rescaleDepth(in, CV_32F, out);
}

template<>
inline void
rescaleDepthTemplated<double>(const Mat& in, Mat& out)
{
  rescaleDepth(in, CV_64F, out);
}

template<typename T>
void
depthTo3dNoMask(const cv::Mat& in_depth, const cv::Mat_<T>& K, cv::Mat& points3d)
{
  const T inv_fx = T(1) / K(0, 0);
  const T inv_fy = T(1) / K(1, 1);
  const T ox = K(0, 2);
  const T oy = K(1, 2);

  // Build z
  cv::Mat_<T> z_mat;
  if (z_mat.depth() == in_depth.depth())
    z_mat = in_depth;
  else
    rescaleDepthTemplated<T>(in_depth, z_mat);

  //cout << "liyang test" << in_depth << endl;
  //exit(1);
  // Pre-compute some constants
  cv::Mat_<T> x_cache(1, in_depth.cols), y_cache(in_depth.rows, 1);
  T* x_cache_ptr = x_cache[0], *y_cache_ptr = y_cache[0];
  for (int x = 0; x < in_depth.cols; ++x, ++x_cache_ptr)
    *x_cache_ptr = (x - ox) * inv_fx;
  for (int y = 0; y < in_depth.rows; ++y, ++y_cache_ptr)
    *y_cache_ptr = (y - oy) * inv_fy;

  y_cache_ptr = y_cache[0];
  for (int y = 0; y < in_depth.rows; ++y, ++y_cache_ptr)
  {
    cv::Vec<T, 3>* point = points3d.ptr<cv::Vec<T, 3> >(y);
    const T* x_cache_ptr_end = x_cache[0] + in_depth.cols;
    const T* depth = z_mat[y];
    for (x_cache_ptr = x_cache[0]; x_cache_ptr != x_cache_ptr_end; ++x_cache_ptr, ++point, ++depth)
    {
      T z = *depth;
      (*point)[0] = (*x_cache_ptr) * z;
      (*point)[1] = (*y_cache_ptr) * z;
      (*point)[2] = z;
      //cout << "liyang test" << (*point)[0] << endl;
      //cout << "liyang test" << (*point)[1] << endl;
      //cout << "liyang test" << (*point)[2] << endl;
      //exit(1);
    }
  }
}
/*
void
depthTo3d(InputArray depth_in, InputArray K_in, OutputArray points3d_out)
{
  cv::Mat depth = depth_in.getMat();
  cv::Mat K = K_in.getMat();
  CV_Assert(K.cols == 3 && K.rows == 3 && (K.depth() == CV_64F || K.depth()==CV_32F));
  CV_Assert(
      depth.type() == CV_64FC1 || depth.type() == CV_32FC1 || depth.type() == CV_16UC1 || depth.type() == CV_16SC1);

  // TODO figure out what to do when types are different: convert or reject ?
  cv::Mat K_new;
  if ((depth.depth() == CV_32F || depth.depth() == CV_64F) && depth.depth() != K.depth())
  {
    K.convertTo(K_new, depth.depth());
  }
  else
    K_new = K;

  // Create 3D points in one go.
  points3d_out.create(depth.size(), CV_MAKETYPE(K_new.depth(), 3));
  cv::Mat points3d = points3d_out.getMat();
  if (K_new.depth() == CV_64F)
    depthTo3dNoMask<double>(depth, K_new, points3d);
  else
    depthTo3dNoMask<float>(depth, K_new, points3d);
}
*/

vector<FixedPointVector> depthTo3d(const FixedPointMatrix& depth, const Mat& K)
{
  vector<FixedPointScalar> cam_fixp;
  cam_fixp = f_Mat2Vec(K, fpconfig);
  FixedPointScalar fx =cam_fixp[0];
  FixedPointScalar fy =cam_fixp[4];
  FixedPointScalar cx =cam_fixp[2];
  FixedPointScalar cy =cam_fixp[5];
  vector<FixedPointScalar> depth_vec;
  //depth_vec = f_Mat2Vec(depth, fpconfig);
  depth_vec = depth.to_vector();

  // Create 3D points in one go.
  //Mat points3d;
  //points3d.create(depth.size(), CV_MAKETYPE(K.depth(), 3));

  vector<FixedPointVector> p3d_vec;
  for (int y = 0; y < depth.value_floating.rows(); ++y)
  {
    for (int x = 0; x < depth.value_floating.cols(); ++x)
    {
         //FIXP_INT_SCALAR_TYPE value_x = x << shift;
         //FixedPointScalar p_x(value_x, fpconfig);
         FixedPointScalar p_x((FIXP_SCALAR_TYPE)x, fpconfig);

         
         p_x = (p_x - cx);
         p_x = p_x * depth_vec[y*depth.value_floating.cols() + x];
         p_x = p_x / fx;
         

         //FIXP_INT_SCALAR_TYPE value_y = y << shift;
         //FixedPointScalar p_y(value_y, fpconfig);
         FixedPointScalar p_y((FIXP_SCALAR_TYPE)y, fpconfig);

         p_y = (p_y - cy) / fy;
         p_y = p_y * depth_vec[y*depth.value_floating.cols() + x];

         //FixedPointScalar p_z(depth_vec[y*depth.cols + x].value, fpconfig);
         FixedPointScalar p_z = depth_vec[y*depth.value_floating.cols() + x];

         FixedPointVector p3d(p_x, p_y, p_z);
         //FixedPointVector p3d;
         //p3d.set(p_x, p_y, p_z);
         p3d_vec.push_back(p3d);
         //if((y*depth.cols + x) == 307148)
         //{
         //    cout << "test:" << depth_vec[(y*depth.cols + x)].value << endl;
         //    cout << "x:" << x << endl;
         //    cout << "y:" << y << endl;
         //    cout << "p_z:" << p_z.value << endl;
         //}
    }
  }
  //int rows = depth.value_floating.rows();
  //int cols = depth.value_floating.cols();
  //points3d = PVec2Mat_f(p3d_vec, depth.rows, depth.cols);
  //FixedPointMatrixP points3d(p3d_vec, rows, cols);
  //cout << "test:" << depth.rows << endl;
  //cout << "test:" << depth_vec[307148].value << endl;
  //cout << "test:" << p3d_vec[307148].x.value << endl;
  //cout << "test:" << p3d_vec[307148].y.value << endl;
  //cout << "test:" << p3d_vec[307148].z.value << endl;
  //cout << "testi:" << points3d.at<Vec3i>(479,588)[0] << endl;
  //cout << "testi:" << points3d.at<Vec3i>(479,588)[1] << endl;
  //cout << "testi:" << points3d.at<Vec3i>(479,588)[2] << endl;
  //exit(1);
  //return points3d;
  return p3d_vec;
}

static
//void preparePyramidCloud(const std::vector<Mat>& pyramidDepth, const Mat& cameraMatrix, std::vector<Mat>& pyramidCloud)
//void preparePyramidCloud(const std::vector<FixedPointMatrix>& pyramidDepth, const Mat& cameraMatrix, std::vector<Mat>& pyramidCloud)
void preparePyramidCloud(const std::vector<FixedPointMatrix>& pyramidDepth, const Mat& cameraMatrix, std::vector<std::vector<FixedPointVector>>& pyramidCloud)
{
    if(!pyramidCloud.empty())
    {
        if(pyramidCloud.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidCloud.");
        
        //for(size_t i = 0; i < pyramidDepth.size(); i++)
        //{
        //    //CV_Assert(pyramidCloud[i].size() == pyramidDepth[i].size());
        //    //CV_Assert(pyramidCloud[i].type() == CV_32FC3);
        //}
    }
    else
    {
        std::vector<Mat> pyramidCameraMatrix;
        buildPyramidCameraMatrix(cameraMatrix, (int)pyramidDepth.size(), pyramidCameraMatrix);

        pyramidCloud.resize(pyramidDepth.size());
        for(size_t i = 0; i < pyramidDepth.size(); i++)
        {
            //Mat cloud;
            //depthTo3d(pyramidDepth[i], pyramidCameraMatrix[i], cloud);
            //cloud = depthTo3d(Vec2Mat_f(pyramidDepth[i].to_vector(), pyramidDepth[i].value_floating.rows(), pyramidDepth[i].value_floating.cols()), pyramidCameraMatrix[i]);
            vector<FixedPointVector> cloud = depthTo3d(pyramidDepth[i], pyramidCameraMatrix[i]);
            pyramidCloud[i] = cloud;
        }
    }
}

static
void preparePyramidNormals(const Mat& normals, const std::vector<Mat>& pyramidDepth, std::vector<Mat>& pyramidNormals)
{
    if(!pyramidNormals.empty())
    {
        if(pyramidNormals.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormals.");

        for(size_t i = 0; i < pyramidNormals.size(); i++)
        {
            CV_Assert(pyramidNormals[i].size() == pyramidDepth[i].size());
            CV_Assert(pyramidNormals[i].type() == CV_32FC3);
        }
    }
    else
    {
        buildPyramid(normals, pyramidNormals, (int)pyramidDepth.size() - 1);
        // renormalize normals
        for(size_t i = 1; i < pyramidNormals.size(); i++)
        {
            Mat& currNormals = pyramidNormals[i];
            for(int y = 0; y < currNormals.rows; y++)
            {
                Point3f* normals_row = currNormals.ptr<Point3f>(y);
                for(int x = 0; x < currNormals.cols; x++)
                {
                    double nrm = norm(normals_row[x]);
                    normals_row[x] *= 1./nrm;
                }
            }
        }
    }
}

static
void preparePyramidMask(const Mat& mask, const std::vector<Mat>& pyramidDepth, float minDepth, float maxDepth,
                        const std::vector<Mat>& pyramidNormal,
                        std::vector<Mat>& pyramidMask)
{
    minDepth = std::max(0.f, minDepth);

    if(!pyramidMask.empty())
    {
        if(pyramidMask.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Levels count of pyramidMask has to be equal to size of pyramidDepth.");

        for(size_t i = 0; i < pyramidMask.size(); i++)
        {
            CV_Assert(pyramidMask[i].size() == pyramidDepth[i].size());
            CV_Assert(pyramidMask[i].type() == CV_8UC1);
        }
    }
    else
    {
        Mat validMask;
        if(mask.empty())
            validMask = Mat(pyramidDepth[0].size(), CV_8UC1, Scalar(255));
        else
            validMask = mask.clone();

        //cout << "liyang test" << validMask << endl;
        //exit(1);
        buildPyramid(validMask, pyramidMask, (int)pyramidDepth.size() - 1);

        for(size_t i = 0; i < pyramidMask.size(); i++)
        {
            Mat levelDepth = pyramidDepth[i].clone();
            //patchNaNs(levelDepth, 0);

            Mat& levelMask = pyramidMask[i];
            levelMask &= (levelDepth > minDepth) & (levelDepth < maxDepth);

            if(!pyramidNormal.empty())
            {
                CV_Assert(pyramidNormal[i].type() == CV_32FC3);
                CV_Assert(pyramidNormal[i].size() == pyramidDepth[i].size());
                Mat levelNormal = pyramidNormal[i].clone();

                Mat validNormalMask = levelNormal == levelNormal; // otherwise it's Nan
                CV_Assert(validNormalMask.type() == CV_8UC3);

                std::vector<Mat> channelMasks;
                split(validNormalMask, channelMasks);
                validNormalMask = channelMasks[0] & channelMasks[1] & channelMasks[2];

                levelMask &= validNormalMask;
            }
        }
    }
}

static
void preparePyramidSobel(const std::vector<Mat>& pyramidImage, int dx, int dy, std::vector<Mat>& pyramidSobel)
{
    if(!pyramidSobel.empty())
    {
        if(pyramidSobel.size() != pyramidImage.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidSobel.");

        for(size_t i = 0; i < pyramidSobel.size(); i++)
        {
            CV_Assert(pyramidSobel[i].size() == pyramidImage[i].size());
            CV_Assert(pyramidSobel[i].type() == CV_16SC1);
        }
    }
    else
    {
        pyramidSobel.resize(pyramidImage.size());
        for(size_t i = 0; i < pyramidImage.size(); i++)
        {
            Sobel(pyramidImage[i], pyramidSobel[i], CV_16S, dx, dy, sobelSize);
        }
    }
}

static
void randomSubsetOfMask(Mat& mask, float part)
{
    const int minPointsCount = 1000; // minimum point count (we can process them fast)
    const int nonzeros = countNonZero(mask);
    const int needCount = std::max(minPointsCount, int(mask.total() * part));
    if(needCount < nonzeros)
    {
        RNG rng(420);
        Mat subset(mask.size(), CV_8UC1, Scalar(0));

        int subsetSize = 0;
        while(subsetSize < needCount)
        {
            int y = rng(mask.rows);
            int x = rng(mask.cols);
            if(mask.at<uchar>(y,x))
            {
                subset.at<uchar>(y,x) = 255;
                mask.at<uchar>(y,x) = 0;
                subsetSize++;
            }
        }
        mask = subset;
    }
}

static
void preparePyramidTexturedMask(const std::vector<Mat>& pyramid_dI_dx, const std::vector<Mat>& pyramid_dI_dy,
                                const std::vector<float>& minGradMagnitudes, const std::vector<Mat>& pyramidMask, double maxPointsPart,
                                std::vector<Mat>& pyramidTexturedMask)
{
    if(!pyramidTexturedMask.empty())
    {
        if(pyramidTexturedMask.size() != pyramid_dI_dx.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidTexturedMask.");

        for(size_t i = 0; i < pyramidTexturedMask.size(); i++)
        {
            CV_Assert(pyramidTexturedMask[i].size() == pyramid_dI_dx[i].size());
            CV_Assert(pyramidTexturedMask[i].type() == CV_8UC1);
        }
    }
    else
    {
        const float sobelScale2_inv = 1.f / (float)(sobelScale * sobelScale);
        pyramidTexturedMask.resize(pyramid_dI_dx.size());
        for(size_t i = 0; i < pyramidTexturedMask.size(); i++)
        {
            const float minScaledGradMagnitude2 = minGradMagnitudes[i] * minGradMagnitudes[i] * sobelScale2_inv;
            const Mat& dIdx = pyramid_dI_dx[i];
            const Mat& dIdy = pyramid_dI_dy[i];

            Mat texturedMask(dIdx.size(), CV_8UC1, Scalar(0));

            for(int y = 0; y < dIdx.rows; y++)
            {
                const short *dIdx_row = dIdx.ptr<short>(y);
                const short *dIdy_row = dIdy.ptr<short>(y);
                uchar *texturedMask_row = texturedMask.ptr<uchar>(y);
                for(int x = 0; x < dIdx.cols; x++)
                {
                    float magnitude2 = static_cast<float>(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x]);
                    if(magnitude2 >= minScaledGradMagnitude2)
                        texturedMask_row[x] = 255;
                }
            }
            pyramidTexturedMask[i] = texturedMask & pyramidMask[i];

            randomSubsetOfMask(pyramidTexturedMask[i], (float)maxPointsPart);
        }
    }
}

static
void preparePyramidNormalsMask(const std::vector<Mat>& pyramidNormals, const std::vector<Mat>& pyramidMask, double maxPointsPart,
                               std::vector<Mat>& pyramidNormalsMask)
{
    if(!pyramidNormalsMask.empty())
    {
        if(pyramidNormalsMask.size() != pyramidMask.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormalsMask.");

        for(size_t i = 0; i < pyramidNormalsMask.size(); i++)
        {
            CV_Assert(pyramidNormalsMask[i].size() == pyramidMask[i].size());
            CV_Assert(pyramidNormalsMask[i].type() == pyramidMask[i].type());
        }
    }
    else
    {
        pyramidNormalsMask.resize(pyramidMask.size());

        for(size_t i = 0; i < pyramidNormalsMask.size(); i++)
        {
            pyramidNormalsMask[i] = pyramidMask[i].clone();
            Mat& normalsMask = pyramidNormalsMask[i];
            for(int y = 0; y < normalsMask.rows; y++)
            {
                const Vec3f *normals_row = pyramidNormals[i].ptr<Vec3f>(y);
                uchar *normalsMask_row = pyramidNormalsMask[i].ptr<uchar>(y);
                for(int x = 0; x < normalsMask.cols; x++)
                {
                    Vec3f n = normals_row[x];
                    if(cvIsNaN(n[0]))
                    {
                        CV_DbgAssert(cvIsNaN(n[1]) && cvIsNaN(n[2]));
                        normalsMask_row[x] = 0;
                    }
                }
            }
            randomSubsetOfMask(normalsMask, (float)maxPointsPart);
        }
    }
}

static inline
void checkImage(const Mat& image)
{
    if(image.empty())
        CV_Error(Error::StsBadSize, "Image is empty.");
    if(image.type() != CV_8UC1)
        CV_Error(Error::StsBadSize, "Image type has to be CV_8UC1.");
}

static inline
void checkDepth(const Mat& depth, const Size& imageSize)
{
    if(depth.empty())
        CV_Error(Error::StsBadSize, "Depth is empty.");
    if(depth.size() != imageSize)
        CV_Error(Error::StsBadSize, "Depth has to have the size equal to the image size.");
    //if(depth.type() != CV_32FC1)
    //    CV_Error(Error::StsBadSize, "Depth type has to be CV_32FC1.");
}

static inline
void checkMask(const Mat& mask, const Size& imageSize)
{
    if(!mask.empty())
    {
        if(mask.size() != imageSize)
            CV_Error(Error::StsBadSize, "Mask has to have the size equal to the image size.");
        if(mask.type() != CV_8UC1)
            CV_Error(Error::StsBadSize, "Mask type has to be CV_8UC1.");
    }
}

static inline
void checkNormals(const Mat& normals, const Size& depthSize)
{
    if(normals.size() != depthSize)
        CV_Error(Error::StsBadSize, "Normals has to have the size equal to the depth size.");
    if(normals.type() != CV_32FC3)
        CV_Error(Error::StsBadSize, "Normals type has to be CV_32FC3.");
}


Size Odometry::prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const
{
    
    checkImage(frame->image);

    checkDepth(frame->depth, frame->image.size());
    
    if(frame->mask.empty() && !frame->pyramidMask.empty()) //pre_frame has pyramidMask
        frame->mask = frame->pyramidMask[0];
    checkMask(frame->mask, frame->image.size());
   
    preparePyramidImage(frame->image, frame->pyramidImage, iterCounts.total());

    //preparePyramidDepth(frame->depth, frame->pyramidDepth, iterCounts.total());

    vector<FixedPointMatrix> pyramidDepth_test;
    buildpy(frame->depth, pyramidDepth_test, iterCounts.total());
    if(frame->pyramidDepth.empty())
    {
        for( int i = 0; i < iterCounts.total(); i++ )
        {
            frame->pyramidDepth.push_back(Vec2Mat_f(pyramidDepth_test[i].to_vector(), pyramidDepth_test[i].value_floating.rows(), pyramidDepth_test[i].value_floating.cols()));
        }  
    }
    //cout << frame->pyramidDepth[0].size() << endl;
    //cout << frame->pyramidDepth[1].size() << endl;
    //cout << frame->pyramidDepth[2].size() << endl;
    //cout << frame->pyramidDepth[3].size() << endl;
    //cout << pyramidDepth_test[0].to_floating().size() << endl;
    //cout << pyramidDepth_test[0].value_floating.rows() << endl;
    //cout << pyramidDepth_test[0].value_floating.cols() << endl;
    //cout << pyramidDepth_test[1].value_floating.rows() << endl;
    //cout << pyramidDepth_test[1].value_floating.cols() << endl;
    //cout << pyramidDepth_test[2].value_floating.rows() << endl;
    //cout << pyramidDepth_test[2].value_floating.cols() << endl;
    //cout << pyramidDepth_test[3].value_floating.rows() << endl;
    //cout << pyramidDepth_test[3].value_floating.cols() << endl;
    //exit(1);

   

    //preparePyramidCloud(frame->pyramidDepth, cameraMatrix, frame->pyramidCloud);
    vector<vector<FixedPointVector>> pyramidCloud_test;
    preparePyramidCloud(pyramidDepth_test, cameraMatrix, pyramidCloud_test);
    if(frame->pyramidCloud.empty())
    {
        for( int i = 0; i < iterCounts.total(); i++ )
        {
            frame->pyramidCloud.push_back(PVec2Mat_f(pyramidCloud_test[i], pyramidDepth_test[i].value_floating.rows(), pyramidDepth_test[i].value_floating.cols()));
        }  
    }

    if(cacheType & OdometryFrame::CACHE_DST)
    {
        if(frame->normals.empty())
        {
            //if(!frame->pyramidNormals.empty())
            //    frame->normals = frame->pyramidNormals[0];
            //else
            //{
                normalsComputer(frame->pyramidCloud[0], frame->depth.rows, frame->depth.cols, frame->normals);
                //vector<FixedPointVector> normals_test = normalsComputer(pyramidCloud_test[0], frame->depth.rows, frame->depth.cols);
                //frame->normals = PVec2Mat_f(normals_test, frame->depth.rows, frame->depth.cols);
            //}
        }
        //cout << "normal done" << endl;
        checkNormals(frame->normals, frame->depth.size());
        //cout << frame->normals << endl;
        ////cout << frame->pyramidCloud[0] << endl;
        ////cout << frame->pyramidCloud_test[0] << endl;
        //cout << frame->normals.size() << endl;
        ////cout << frame->normals.at<Vec3f>(0,0) << endl;
        ////cout << frame->normals.at<Vec3f>(429,638) << endl;
        ////cout << frame->normals.at<Vec3f>(430,638) << endl;
        ////cout << frame->normals.at<Vec3f>(431,638) << endl;
        ////cout << frame->normals.at<Vec3f>(638,429) << endl;
        ////cout << frame->normals.at<Vec3f>(638,430) << endl;
        ////cout << frame->normals.at<Vec3f>(638,431) << endl;
        ////cout << frame->normals.at<Vec3f>(479,638) << endl;
        //cout << frame->normals.at<Vec3f>(478,638) << endl;
        //cout << frame->normals.at<Vec3f>(478,639) << endl;
        //cout << frame->normals.at<Vec3f>(478,590) << endl;
        //cout << frame->normals.at<Vec3f>(478,589) << endl;
        //cout << frame->normals.at<Vec3f>(478,588) << endl;
        //exit(1);

        preparePyramidNormals(frame->normals, frame->pyramidDepth, frame->pyramidNormals);

        preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals, frame->pyramidMask);

        //cout << "liyang test" << frame->mask << endl;
        //exit(1);
        preparePyramidSobel(frame->pyramidImage, 1, 0, frame->pyramid_dI_dx);
        preparePyramidSobel(frame->pyramidImage, 0, 1, frame->pyramid_dI_dy);
        preparePyramidTexturedMask(frame->pyramid_dI_dx, frame->pyramid_dI_dy,
                                   minGradientMagnitudes, frame->pyramidMask,
                                   maxPointsPart, frame->pyramidTexturedMask);

        preparePyramidNormalsMask(frame->pyramidNormals, frame->pyramidMask, maxPointsPart, frame->pyramidNormalsMask);
    }
    else
        preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals, frame->pyramidMask);

    return frame->image.size();
}

static
void computeCorresps(const Mat& K, const Mat& K_inv, const Mat& Rt,
                     const Mat& depth0, const Mat& validMask0,
                     const Mat& depth1, const Mat& selectMask1, float maxDepthDiff,
                     //const vector<FixedPointScalar>& depth0, const Mat& validMask0,
                     //const vector<FixedPointScalar>& depth1, const Mat& selectMask1, float maxDepthDiff,
                     Mat& _corresps)
{

  FixedPointScalar fx (K.at<float>(0,0), fpconfig);//float
  FixedPointScalar fy (K.at<float>(1,1), fpconfig);//float
  FixedPointScalar cx (K.at<float>(0,2), fpconfig);//float
  FixedPointScalar cy (K.at<float>(1,2), fpconfig);//float
  FixedPointScalar fx_inv (K_inv.at<float>(0,0), fpconfig);//float
  FixedPointScalar fy_inv (K_inv.at<float>(1,1), fpconfig);//float
  //int64_t fx_inv_tmp = (1LL << (2*shift)) / fx.value;
  //int64_t fy_inv_tmp = (1LL << (2*shift)) / fy.value;
  //FixedPointScalar fx_inv (fx_inv_tmp, fpconfig);
  //FixedPointScalar fy_inv (fy_inv_tmp, fpconfig);
  //int64_t cx_inv_tmp = (cx * fx_inv).value * -1;
  //int64_t cy_inv_tmp = (cy * fy_inv).value * -1;
  //FixedPointScalar cx_inv (cx_inv_tmp, fpconfig);
  //FixedPointScalar cy_inv (cy_inv_tmp, fpconfig);
  FixedPointScalar cx_inv (K_inv.at<float>(0,2), fpconfig);//float
  FixedPointScalar cy_inv (K_inv.at<float>(1,2), fpconfig);//float

  vector<FixedPointScalar> Rt_vec;
  Rt_vec = f_Mat2Vec(Rt, fpconfig);
  vector<FixedPointScalar> d0_vec;
  vector<FixedPointScalar> d1_vec;
  d0_vec = f_Mat2Vec(depth0, fpconfig);//float
  d1_vec = f_Mat2Vec(depth1, fpconfig);//float
  //d0_vec = depth0;
  //d1_vec = depth1;

  //FixedPointScalar RK_inv_00 ((Rt_vec[0]*fx_inv).value, fpconfig);
  //FixedPointScalar RK_inv_01 ((Rt_vec[1]*fy_inv).value, fpconfig);
  //FixedPointScalar RK_inv_02 ((Rt_vec[0]*cx_inv + Rt_vec[1]*cy_inv + Rt_vec[2]).value, fpconfig);
  //FixedPointScalar RK_inv_10 ((Rt_vec[4]*fx_inv).value, fpconfig);
  //FixedPointScalar RK_inv_11 ((Rt_vec[5]*fy_inv).value, fpconfig);
  //FixedPointScalar RK_inv_12 ((Rt_vec[4]*cx_inv + Rt_vec[5]*cy_inv + Rt_vec[6]).value, fpconfig);
  //FixedPointScalar RK_inv_20 ((Rt_vec[8]*fx_inv).value, fpconfig);
  //FixedPointScalar RK_inv_21 ((Rt_vec[9]*fy_inv).value, fpconfig);
  //FixedPointScalar RK_inv_22 ((Rt_vec[8]*cx_inv + Rt_vec[9]*cy_inv + Rt_vec[10]).value, fpconfig);
  FixedPointScalar RK_inv_00 = Rt_vec[0]*fx_inv;
  FixedPointScalar RK_inv_01 = Rt_vec[1]*fy_inv;
  FixedPointScalar RK_inv_02 = Rt_vec[0]*cx_inv + Rt_vec[1]*cy_inv + Rt_vec[2];
  FixedPointScalar RK_inv_10 = Rt_vec[4]*fx_inv;
  FixedPointScalar RK_inv_11 = Rt_vec[5]*fy_inv;
  FixedPointScalar RK_inv_12 = Rt_vec[4]*cx_inv + Rt_vec[5]*cy_inv + Rt_vec[6];
  FixedPointScalar RK_inv_20 = Rt_vec[8]*fx_inv;
  FixedPointScalar RK_inv_21 = Rt_vec[9]*fy_inv;
  FixedPointScalar RK_inv_22 = Rt_vec[8]*cx_inv + Rt_vec[9]*cy_inv + Rt_vec[10];
  
  //FixedPointScalar KRK_inv_00 ((fx*RK_inv_00 + cx*RK_inv_20).value, fpconfig);
  //FixedPointScalar KRK_inv_01 ((fx*RK_inv_01 + cx*RK_inv_21).value, fpconfig);
  //FixedPointScalar KRK_inv_02 ((fx*RK_inv_02 + cx*RK_inv_22).value, fpconfig);
  //FixedPointScalar KRK_inv_10 ((fy*RK_inv_10 + cy*RK_inv_20).value, fpconfig);
  //FixedPointScalar KRK_inv_11 ((fy*RK_inv_11 + cy*RK_inv_21).value, fpconfig);
  //FixedPointScalar KRK_inv_12 ((fy*RK_inv_12 + cy*RK_inv_22).value, fpconfig);
  FixedPointScalar KRK_inv_00 = fx*RK_inv_00 + cx*RK_inv_20;
  FixedPointScalar KRK_inv_01 = fx*RK_inv_01 + cx*RK_inv_21;
  FixedPointScalar KRK_inv_02 = fx*RK_inv_02 + cx*RK_inv_22;
  FixedPointScalar KRK_inv_10 = fy*RK_inv_10 + cy*RK_inv_20;
  FixedPointScalar KRK_inv_11 = fy*RK_inv_11 + cy*RK_inv_21;
  FixedPointScalar KRK_inv_12 = fy*RK_inv_12 + cy*RK_inv_22;
  //FixedPointScalar KRK_inv_20 ((RK_inv_20).value, fpconfig);
  //FixedPointScalar KRK_inv_21 ((RK_inv_21).value, fpconfig);
  //FixedPointScalar KRK_inv_22 ((RK_inv_22).value, fpconfig);
  FixedPointScalar KRK_inv_20 = RK_inv_20;
  FixedPointScalar KRK_inv_21 = RK_inv_21;
  FixedPointScalar KRK_inv_22 = RK_inv_22;
  //FixedPointScalar Kt_0 ((fx*Rt_vec[3] + cx*Rt_vec[11]).value, fpconfig);
  //FixedPointScalar Kt_1 ((fy*Rt_vec[7] + cy*Rt_vec[11]).value, fpconfig);
  //FixedPointScalar Kt_2 ((Rt_vec[11]).value, fpconfig);
  FixedPointScalar Kt_0 = fx*Rt_vec[3] + cx*Rt_vec[11];
  FixedPointScalar Kt_1 = fy*Rt_vec[7] + cy*Rt_vec[11];
  FixedPointScalar Kt_2 = Rt_vec[11];
  int rows = depth1.rows;
  int cols = depth1.cols;
  int correspCount = 0;
  Mat corresps(depth0.size(), CV_16SC2, Scalar::all(-1));
  Rect r(0, 0, cols, rows);
  for(int v1 = 0; v1 < rows; v1++)
  {
     for(int u1 = 0; u1 < cols; u1++)
     {
         if(selectMask1.at<uchar>(v1, u1))
         {
             FixedPointScalar d1 = d1_vec[v1*cols + u1];
             //FixedPointScalar u1_shift ((int64_t)(u1 * (1LL << shift)), fpconfig);
             //FixedPointScalar v1_shift ((int64_t)(v1 * (1LL << shift)), fpconfig);
             FixedPointScalar u1_shift ((FIXP_SCALAR_TYPE)u1, fpconfig);
             FixedPointScalar v1_shift ((FIXP_SCALAR_TYPE)v1, fpconfig);
             //FixedPointScalar transformed_d1_shift ((KRK_inv_20*u1_shift + KRK_inv_21*v1_shift + KRK_inv_22).value, fpconfig);
             FixedPointScalar transformed_d1_shift = KRK_inv_20*u1_shift + KRK_inv_21*v1_shift + KRK_inv_22;
             transformed_d1_shift = (d1*transformed_d1_shift) + Kt_2;
             if(transformed_d1_shift.value > 0)
             {
                 //FixedPointScalar u0_shift ((KRK_inv_00*u1_shift + KRK_inv_01*v1_shift + KRK_inv_02).value, fpconfig);
                 FixedPointScalar u0_shift = KRK_inv_00*u1_shift + KRK_inv_01*v1_shift + KRK_inv_02;
                 FixedPointScalar ttt = KRK_inv_00*u1_shift + KRK_inv_01*v1_shift;
                 FixedPointScalar test_shift = ttt + KRK_inv_02;
                 //FixedPointScalar v0_shift ((KRK_inv_10*u1_shift + KRK_inv_11*v1_shift + KRK_inv_12).value, fpconfig);
                 FixedPointScalar v0_shift = KRK_inv_10*u1_shift + KRK_inv_11*v1_shift + KRK_inv_12;
                 u0_shift = (d1*u0_shift) + Kt_0;
                 v0_shift = (d1*v0_shift) + Kt_1;
                 u0_shift = u0_shift / transformed_d1_shift;
                 v0_shift = v0_shift / transformed_d1_shift;
                 int u0 = (int)round(u0_shift.value_floating);
                 int v0 = (int)round(v0_shift.value_floating); 
                        //if(u1==18 && v1==7)
                        //{
                        //   cout << u0_shift.value_floating << endl;
                        //   cout << v0_shift.value_floating << endl;
                        //  cout << "u0 "  << u0 << endl;
                        //  cout << "v0 "  << v0 << endl;
                        //  cout << "u1 "  << u1 << endl;
                        //  cout << "v1 "  << v1 << endl;
                        //  cout << "KRK "  << KRK_inv_00.value << endl;
                        //}
                 if(r.contains(Point(u0,v0)))
                 {
                     FixedPointScalar d0 = d0_vec[v0*cols + u0];
                           //if(u1==71 && v1==8)
                           ////if(u0==71 && v0==8)
                           //{
                           //  cout << u0 << endl;
                           //  cout << v0 << endl;
                           //  cout << u1 << endl;
                           //  cout << v1 << endl;
                           //  cout << u0_shift.value_floating << endl;
                           //  cout << v0_shift.value_floating << endl;
                           //  cout << validMask0.at<uchar>(v0, u0) << endl;
                           //  cout << std::abs(transformed_d1_shift.value - d0.value) << endl;
                           //  cout << maxDepthDiff*(1LL<<shift) << endl;
                           //  exit(1);
                           //}
                     if(validMask0.at<uchar>(v0, u0) && std::abs(transformed_d1_shift.value - d0.value) <= (maxDepthDiff*(1LL<<shift)))
                     {
                            Vec2s& c = corresps.at<Vec2s>(v0,u0);
                            if(c[0] != -1)
                            {
                                int exist_u1 = c[0], exist_v1 = c[1];
                                FixedPointScalar exist_u1_shift ((int64_t)(exist_u1 * (1LL << shift)), fpconfig);
                                FixedPointScalar exist_v1_shift ((int64_t)(exist_v1 * (1LL << shift)), fpconfig);
                                FixedPointScalar exist_d1 = d1_vec[exist_v1*cols + exist_u1];
                                FixedPointScalar exist_d1_shift = KRK_inv_20*exist_u1_shift + KRK_inv_21*exist_v1_shift + KRK_inv_22;
                                exist_d1_shift = (exist_d1*exist_d1_shift) + Kt_2;
                                if(transformed_d1_shift.value > exist_d1_shift.value)
                                    continue;
                            }
                            else
                                correspCount++;

                            c = Vec2s((short)u1, (short)v1);
                            //if(u0==72 && v0==58)
                            //{
                            //cout << u0 << endl;
                            //cout << v0 << endl;
                            //cout << u1 << endl;
                            //cout << v1 << endl;
                            //cout << c << endl;
                            //exit(1);
                            //}

                     }
                 }
             }

         }
     }
  }

  _corresps.create(correspCount, 1, CV_32SC4);
  Vec4i * corresps_ptr = _corresps.ptr<Vec4i>();
  for(int v0 = 0, i = 0; v0 < corresps.rows; v0++)
  {
      const Vec2s* corresps_row = corresps.ptr<Vec2s>(v0);
      for(int u0 = 0; u0 < corresps.cols; u0++)
      {
          const Vec2s& c = corresps_row[u0];
          if(c[0] != -1)
              corresps_ptr[i++] = Vec4i(u0,v0,c[0],c[1]);
      }
  }
  //cout << _corresps.size() << endl;
  //exit(1);
}

/*
static
void computeCorresps(const Mat& K, const Mat& K_inv, const Mat& Rt,
                     const Mat& depth0, const Mat& validMask0,
                     const Mat& depth1, const Mat& selectMask1, float maxDepthDiff,
                     Mat& _corresps)
{
    //CV_Assert(K.type() == CV_64FC1);
    //CV_Assert(K_inv.type() == CV_64FC1);
    //CV_Assert(Rt.type() == CV_64FC1);
    CV_Assert(K.type() == CV_32FC1);
    CV_Assert(K_inv.type() == CV_32FC1);
    CV_Assert(Rt.type() == CV_32FC1);

    Mat corresps(depth1.size(), CV_16SC2, Scalar::all(-1));

    Rect r(0, 0, depth1.cols, depth1.rows);
    Mat Kt = Rt(Rect(3,0,1,3)).clone();
    Kt = K * Kt;
    //const double * Kt_ptr = Kt.ptr<const double>();
    const float * Kt_ptr = Kt.ptr<const float>();

    AutoBuffer<float> buf(3 * (depth1.cols + depth1.rows));
    float *KRK_inv0_u1 = buf;
    float *KRK_inv1_v1_plus_KRK_inv2 = KRK_inv0_u1 + depth1.cols;
    float *KRK_inv3_u1 = KRK_inv1_v1_plus_KRK_inv2 + depth1.rows;
    float *KRK_inv4_v1_plus_KRK_inv5 = KRK_inv3_u1 + depth1.cols;
    float *KRK_inv6_u1 = KRK_inv4_v1_plus_KRK_inv5 + depth1.rows;
    float *KRK_inv7_v1_plus_KRK_inv8 = KRK_inv6_u1 + depth1.cols;
    {
        Mat R = Rt(Rect(0,0,3,3)).clone();

        Mat KRK_inv = K * R * K_inv;
        //const double * KRK_inv_ptr = KRK_inv.ptr<const double>();
        const float * KRK_inv_ptr = KRK_inv.ptr<const float>();
        for(int u1 = 0; u1 < depth1.cols; u1++)
        {
            KRK_inv0_u1[u1] = (float)(KRK_inv_ptr[0] * u1);
            KRK_inv3_u1[u1] = (float)(KRK_inv_ptr[3] * u1);
            KRK_inv6_u1[u1] = (float)(KRK_inv_ptr[6] * u1);
        }

        for(int v1 = 0; v1 < depth1.rows; v1++)
        {
            KRK_inv1_v1_plus_KRK_inv2[v1] = (float)(KRK_inv_ptr[1] * v1 + KRK_inv_ptr[2]);
            KRK_inv4_v1_plus_KRK_inv5[v1] = (float)(KRK_inv_ptr[4] * v1 + KRK_inv_ptr[5]);
            KRK_inv7_v1_plus_KRK_inv8[v1] = (float)(KRK_inv_ptr[7] * v1 + KRK_inv_ptr[8]);
        }
    }

    int correspCount = 0;
    for(int v1 = 0; v1 < depth1.rows; v1++)
    {
        const float *depth1_row = depth1.ptr<float>(v1);
        const uchar *mask1_row = selectMask1.ptr<uchar>(v1);
        for(int u1 = 0; u1 < depth1.cols; u1++)
        {
            float d1 = depth1_row[u1];
            if(mask1_row[u1])
            {
                CV_DbgAssert(!cvIsNaN(d1));
                float transformed_d1 = static_cast<float>(d1 * (KRK_inv6_u1[u1] + KRK_inv7_v1_plus_KRK_inv8[v1]) +
                                                          Kt_ptr[2]);
                if(transformed_d1 > 0)
                {
                    float transformed_d1_inv = 1.f / transformed_d1;
                    int u0 = cvRound(transformed_d1_inv * (d1 * (KRK_inv0_u1[u1] + KRK_inv1_v1_plus_KRK_inv2[v1]) +
                                                           Kt_ptr[0]));
                    int v0 = cvRound(transformed_d1_inv * (d1 * (KRK_inv3_u1[u1] + KRK_inv4_v1_plus_KRK_inv5[v1]) +
                                                           Kt_ptr[1]));
                         if(u1==55 && v1==56)
                         {
                            cout << u0 << endl;
                            cout << v0 << endl;
                            cout << u1 << endl;
                            cout << v1 << endl;
                         }
                    if(r.contains(Point(u0,v0)))
                    {
                        float d0 = depth0.at<float>(v0,u0);
                        if(validMask0.at<uchar>(v0, u0) && std::abs(transformed_d1 - d0) <= maxDepthDiff)
                        {
                            CV_DbgAssert(!cvIsNaN(d0));
                            Vec2s& c = corresps.at<Vec2s>(v0,u0);
                            if(c[0] != -1)
                            {
                                int exist_u1 = c[0], exist_v1 = c[1];

                                float exist_d1 = (float)(depth1.at<float>(exist_v1,exist_u1) *
                                    (KRK_inv6_u1[exist_u1] + KRK_inv7_v1_plus_KRK_inv8[exist_v1]) + Kt_ptr[2]);

                                if(transformed_d1 > exist_d1)
                                    continue;
                            }
                            else
                                correspCount++;

                            c = Vec2s((short)u1, (short)v1);
                            //cout << u0 << endl;
                            //cout << v0 << endl;
                            //cout << u1 << endl;
                            //cout << v1 << endl;
                            //cout << c << endl;
                            //exit(1);
                        }
                    }
                }
            }
        }
    }

    _corresps.create(correspCount, 1, CV_32SC4);
    Vec4i * corresps_ptr = _corresps.ptr<Vec4i>();
    for(int v0 = 0, i = 0; v0 < corresps.rows; v0++)
    {
        const Vec2s* corresps_row = corresps.ptr<Vec2s>(v0);
        for(int u0 = 0; u0 < corresps.cols; u0++)
        {
            const Vec2s& c = corresps_row[u0];
            if(c[0] != -1)
                corresps_ptr[i++] = Vec4i(u0,v0,c[0],c[1]);
        }
    }
}
*/
typedef
void (*CalcRgbdEquationCoeffsPtr)(double*, double, double, const Point3f&, double, double);

typedef
//void (*CalcICPEquationCoeffsPtr)(double*, const Point3f&, const Vec3f&);
void (*CalcICPEquationCoeffsPtr)(float*, const Point3f&, const Vec3f&);


static
void calcRgbdLsmMatrices(const Mat& image0, const Mat& cloud0, const Mat& Rt,
               const Mat& image1, const Mat& dI_dx1, const Mat& dI_dy1,
               const Mat& corresps, double fx, double fy, double sobelScaleIn,
               Mat& AtA, Mat& AtB, CalcRgbdEquationCoeffsPtr func, int transformDim)
{
    AtA = Mat(transformDim, transformDim, CV_64FC1, Scalar(0));
    AtB = Mat(transformDim, 1, CV_64FC1, Scalar(0));
    double* AtB_ptr = AtB.ptr<double>();

    const int correspsCount = corresps.rows;

    CV_Assert(Rt.type() == CV_64FC1);
    const double * Rt_ptr = Rt.ptr<const double>();

    AutoBuffer<float> diffs(correspsCount);
    float* diffs_ptr = diffs;

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    double sigma = 0;
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
         const Vec4i& c = corresps_ptr[correspIndex];
         int u0 = c[0], v0 = c[1];
         int u1 = c[2], v1 = c[3];

         diffs_ptr[correspIndex] = static_cast<float>(static_cast<int>(image0.at<uchar>(v0,u0)) -
                                                      static_cast<int>(image1.at<uchar>(v1,u1)));
         //std::cout << "====================test=======================" << diffs_ptr[0] <<  std::endl;
         //std::cout << static_cast<int>(image0.at<uchar>(v0,u0)) <<  std::endl;
         //std::cout << static_cast<int>(image1.at<uchar>(v1,u1)) <<  std::endl;
	 //exit(1);
         sigma += diffs_ptr[correspIndex] * diffs_ptr[correspIndex];
    }
    sigma = std::sqrt(sigma/correspsCount);

    std::vector<double> A_buf(transformDim);
    double* A_ptr = &A_buf[0];

    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
         const Vec4i& c = corresps_ptr[correspIndex];
         int u0 = c[0], v0 = c[1];
         int u1 = c[2], v1 = c[3];

         double w = sigma + std::abs(diffs_ptr[correspIndex]);
         w = w > DBL_EPSILON ? 1./w : 1.;

         double w_sobelScale = w * sobelScaleIn;

         const Point3f& p0 = cloud0.at<Point3f>(v0,u0);
         Point3f tp0;
         tp0.x = (float)(p0.x * Rt_ptr[0] + p0.y * Rt_ptr[1] + p0.z * Rt_ptr[2] + Rt_ptr[3]);
         tp0.y = (float)(p0.x * Rt_ptr[4] + p0.y * Rt_ptr[5] + p0.z * Rt_ptr[6] + Rt_ptr[7]);
         tp0.z = (float)(p0.x * Rt_ptr[8] + p0.y * Rt_ptr[9] + p0.z * Rt_ptr[10] + Rt_ptr[11]);

         func(A_ptr,
              w_sobelScale * dI_dx1.at<short int>(v1,u1),
              w_sobelScale * dI_dy1.at<short int>(v1,u1),
              tp0, fx, fy);

        for(int y = 0; y < transformDim; y++)
        {
            double* AtA_ptr = AtA.ptr<double>(y);
            for(int x = y; x < transformDim; x++)
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];

            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
        }
    }

    for(int y = 0; y < transformDim; y++)
        for(int x = y+1; x < transformDim; x++)
            AtA.at<double>(x,y) = AtA.at<double>(y,x);
}


static
void calcICPLsmMatrices(const Mat& cloud0, const Mat& Rt,
                        const Mat& cloud1, const Mat& normals1,
                        const Mat& corresps,
                        //Mat& AtA, Mat& AtB, CalcICPEquationCoeffsPtr func, int transformDim)
                        vector<FixedPointScalar>& A_vec, vector<FixedPointScalar>& B_vec, CalcICPEquationCoeffsPtr func, int transformDim)
{
    
    //FixedPointScalar zero_fix((int64_t)0, fpconfig);
    //vector<FixedPointScalar> A_vec(transformDim*transformDim, zero_fix);
    //vector<FixedPointScalar> B_vec(transformDim, zero_fix);

    //FixedPointScalar correspsCount((int64_t)(corresps.rows*(1LL<<shift)), fpconfig);
    FixedPointScalar correspsCount((FIXP_SCALAR_TYPE)corresps.rows, fpconfig);

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    vector<FixedPointScalar> Rt_vec;
    Rt_vec = f_Mat2Vec(Rt, fpconfig); //float
    //vector<FixedPointVector> cloud0_vec;
    //cloud0_vec = f_PMat2Vec(cloud0, fpconfig); //float
    //vector<FixedPointVector> cloud1_vec;
    //cloud1_vec = f_PMat2Vec(cloud1, fpconfig); //float
    //vector<FixedPointVector> nor_vec;
    //nor_vec = f_PMat2Vec(normals1, fpconfig);  //float

    vector<FixedPointScalar> diffs_ptr;
    vector<FixedPointVector> tps0_ptr;
    //FixedPointScalar sigma((int64_t)0, fpconfig);
    FixedPointScalar sigma((FIXP_SCALAR_TYPE)0, fpconfig);
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1];
        int u1 = c[2], v1 = c[3];

        //FixedPointVector p0 = cloud0_vec[v0*cloud0.cols + u0];
        const Point3f& p0 = cloud0.at<Point3f>(v0,u0); // float
        FixedPointScalar p0x ((FIXP_SCALAR_TYPE)p0.x, fpconfig);
        FixedPointScalar p0y ((FIXP_SCALAR_TYPE)p0.y, fpconfig);
        FixedPointScalar p0z ((FIXP_SCALAR_TYPE)p0.z, fpconfig);


        //FixedPointScalar tp0x ((p0x * Rt_vec[0] + p0y * Rt_vec[1] + p0z * Rt_vec[2] + Rt_vec[3]).value, fpconfig);
        //FixedPointScalar tp0y ((p0x * Rt_vec[4] + p0y * Rt_vec[5] + p0z * Rt_vec[6] + Rt_vec[7]).value, fpconfig);
        //FixedPointScalar tp0z ((p0x * Rt_vec[8] + p0y * Rt_vec[9] + p0z * Rt_vec[10] + Rt_vec[11]).value, fpconfig);
        FixedPointScalar tp0x = p0x * Rt_vec[0] + p0y * Rt_vec[1] + p0z * Rt_vec[2] + Rt_vec[3];
        FixedPointScalar tp0y = p0x * Rt_vec[4] + p0y * Rt_vec[5] + p0z * Rt_vec[6] + Rt_vec[7];
        FixedPointScalar tp0z = p0x * Rt_vec[8] + p0y * Rt_vec[9] + p0z * Rt_vec[10] + Rt_vec[11];

        //FixedPointVector n1 = nor_vec[v1*cloud1.rows + u1];
        const Point3f& n1 = normals1.at<Point3f>(v1,u1); // float
        FixedPointScalar n1x ((FIXP_SCALAR_TYPE)n1.x, fpconfig);
        FixedPointScalar n1y ((FIXP_SCALAR_TYPE)n1.y, fpconfig);
        FixedPointScalar n1z ((FIXP_SCALAR_TYPE)n1.z, fpconfig);

        //FixedPointVector p1 = cloud1_vec[v1*cloud1.rows + u1];
        const Point3f& p1 = cloud1.at<Point3f>(v1,u1); // float
        FixedPointScalar p1x ((FIXP_SCALAR_TYPE)p1.x, fpconfig);
        FixedPointScalar p1y ((FIXP_SCALAR_TYPE)p1.y, fpconfig);
        FixedPointScalar p1z ((FIXP_SCALAR_TYPE)p1.z, fpconfig);

        FixedPointVector v (p1x - tp0x, p1y - tp0y, p1z - tp0z);

        FixedPointVector tp0(tp0x, tp0y, tp0z);
        tps0_ptr.push_back(tp0);
        FixedPointScalar diffs = n1x * v.x + n1y * v.y + n1z * v.z;
        diffs_ptr.push_back(diffs);
        //std::cout << "====================test=======================" << diffs_ptr[0] <<  std::endl;
        //exit(1);
        sigma += diffs * diffs;
    }

    //AtA = Mat(transformDim, transformDim, CV_32FC1, Scalar(0));
    //AtB = Mat(transformDim, 1, CV_32FC1, Scalar(0));
    //float* AtB_ptr = AtB.ptr<float>();
    //cout << sigma.value_floating << endl;
    //FixedPointScalar sigma_final( (int64_t)(sqrt((sigma/correspsCount).value) * (1LL<<shift_half)) ,  fpconfig);
    //FixedPointScalar sigma_final((sigma/correspsCount).sqrt().value ,  fpconfig);
    FixedPointScalar sigma_final = (sigma/correspsCount).sqrt();
    //cout << "test: " << sigma_final.value_floating << endl;
    //cout << sigma_final.to_floating() << endl;

    //std::vector<float> A_buf(transformDim);
    //Mat tps0_mat;
    //tps0_mat = PVec2Mat_f(tps0_ptr, corresps.rows, 1);
    //Point3f* tps0_mat_ptr = tps0_mat.ptr<Point3f>();
    //double* A_ptr = &A_buf[0];
    //float* A_ptr = &A_buf[0];
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u1 = c[2], v1 = c[3];
        
        //int64_t w_tmp = sigma_final.value + std::abs(diffs_ptr[correspIndex].value);
        //if(w_tmp == 0)
        //{
        //  w_tmp = 1LL<<shift;
        //}
        //else
        //{
        //  w_tmp = ((1LL<<(shift*2))/w_tmp);
        //  //w_tmp = ((1LL<<shift)/w_tmp)<<shift;
        //  //cout << w_tmp << endl;
        //  //exit(1);
        //}
        //FixedPointScalar w (w_tmp, fpconfig);
        

        FixedPointScalar w_tmp = sigma_final + diffs_ptr[correspIndex].abs();
        FixedPointScalar one_fix((FIXP_SCALAR_TYPE)1, fpconfig);
        FixedPointScalar w = one_fix;
        if(w_tmp.value == 0)
        {
          w = one_fix;
        }
        else
        {
          w = one_fix / w_tmp;
        }
        
        //cout << 'w' << w.value_floating << endl;
        //func(A_ptr, tps0_mat_ptr[correspIndex], normals1.at<Vec3f>(v1, u1) * w.value_floating);

        const Point3f& n1 = normals1.at<Point3f>(v1,u1); // float
        FixedPointScalar n1x ((FIXP_SCALAR_TYPE)n1.x, fpconfig);
        FixedPointScalar n1y ((FIXP_SCALAR_TYPE)n1.y, fpconfig);
        FixedPointScalar n1z ((FIXP_SCALAR_TYPE)n1.z, fpconfig);
        n1x = n1x * w;
        n1y = n1y * w;
        n1z = n1z * w;

        FixedPointVector tp0 = tps0_ptr[correspIndex];
        FixedPointScalar neg_one(-1.0f, fpconfig);
        FixedPointScalar zero_fix((FIXP_SCALAR_TYPE)0, fpconfig);
        vector<FixedPointScalar> C_vec(6, zero_fix);
        //C_vec.push_back(neg_one * tp0.z * n1y + tp0.y * n1z);
        //C_vec.push_back( tp0.z * n1x - tp0.x * n1z);
        //C_vec.push_back(neg_one * tp0.y * n1x + tp0.x * n1y);
        //C_vec.push_back(n1x);
        //C_vec.push_back(n1y);
        //C_vec.push_back(n1z);
        FixedPointScalar c0 = neg_one * tp0.z * n1y + tp0.y * n1z;
        FixedPointScalar c1 = tp0.z * n1x - tp0.x * n1z;
        FixedPointScalar c2 = neg_one * tp0.y * n1x + tp0.x * n1y;
        C_vec[0] = c0;
        C_vec[1] = c1;
        C_vec[2] = c2;
        C_vec[3] = n1x;
        C_vec[4] = n1y;
        C_vec[5] = n1z;

        //Mat AtC = Vec2Mat_f(C_vec, transformDim, 1); //float
        //cout << AtC << endl;
        //cout << "tp0z" << tp0.z.value_floating <<  endl;
        //cout << "tp0z" << tp0.z.value <<  endl;
        //cout << "tp0y" << tp0.y.value_floating <<  endl;
        //cout << "tp0y" << tp0.y.value <<  endl;
        //cout << "n1y" << n1y.value_floating <<  endl;
        //cout << "n1y" << n1y.value <<  endl;
        //cout << "n1z" << n1z.value_floating <<  endl;
        //cout << "n1z" << n1z.value <<  endl;
        //cout << "-1" << neg_one.value_floating <<  endl;
        //cout << "-1" << neg_one.value <<  endl;
        //FixedPointScalar lll = neg_one * tp0.z * n1y + tp0.y * n1z;
        //cout << "lll " << lll.value_floating <<  endl;
        //cout << "c_vec[0] " << C_vec[0].value_floating <<  endl;
        //cout << "c_vec[0] " << C_vec[0].value <<  endl;
        //exit(1);

        for(int y = 0; y < transformDim; y++)
        {
            //float* AtA_ptr = AtA.ptr<float>(y);
            for(int x = y; x < transformDim; x++)
            {
                FixedPointScalar  test = C_vec[y] * C_vec[x];
                //AtA_ptr[x] += (C_vec[y] * C_vec[x]).value_floating;
                //A_vec[y*transformDim + x] = A_vec[y*transformDim + x] + (C_vec[y] * C_vec[x]);
                //AtA_ptr[x] += test.value_floating;
                A_vec[y*transformDim + x] = A_vec[y*transformDim + x] + test;
            }
            //AtB_ptr[y] += C_vec[y].value_floating * w.value_floating * diffs_ptr[correspIndex].value_floating;
            B_vec[y] = B_vec[y] + (C_vec[y] * w * diffs_ptr[correspIndex]);
        }
    }
    //if(A_vec[0].value==0)
        //cout << "===========DIVVV===================== " << A_vec[0].value << endl;

    for(int y = 0; y < transformDim; y++)
        for(int x = y+1; x < transformDim; x++)
        {
            //AtA.at<float>(x,y) = AtA.at<float>(y,x);
            A_vec[x*transformDim + y] = A_vec[y*transformDim + x];
        }
    //Mat AtA = Vec2Mat_f(A_vec, transformDim, transformDim); //float
    //Mat AtB = Vec2Mat_f(B_vec, transformDim, 1); //float
    //cout << AtA << endl;
    //cout << AtB << endl;
    //exit(1);
}

/*
static
void calcICPLsmMatrices(const Mat& cloud0, const Mat& Rt,
                        const Mat& cloud1, const Mat& normals1,
                        const Mat& corresps,
                        Mat& AtA, Mat& AtB, CalcICPEquationCoeffsPtr func, int transformDim)
{
    //AtA = Mat(transformDim, transformDim, CV_64FC1, Scalar(0));
    //AtB = Mat(transformDim, 1, CV_64FC1, Scalar(0));
    //double* AtB_ptr = AtB.ptr<double>();
    AtA = Mat(transformDim, transformDim, CV_32FC1, Scalar(0));
    AtB = Mat(transformDim, 1, CV_32FC1, Scalar(0));
    float* AtB_ptr = AtB.ptr<float>();

    const int correspsCount = corresps.rows;

    //CV_Assert(Rt.type() == CV_64FC1);
    //const double * Rt_ptr = Rt.ptr<const double>();
    CV_Assert(Rt.type() == CV_32FC1);
    const float * Rt_ptr = Rt.ptr<const float>();

    AutoBuffer<float> diffs(correspsCount);
    float * diffs_ptr = diffs;

    AutoBuffer<Point3f> transformedPoints0(correspsCount);
    Point3f * tps0_ptr = transformedPoints0;

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    //double sigma = 0;
    float sigma = 0;
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1];
        int u1 = c[2], v1 = c[3];

        const Point3f& p0 = cloud0.at<Point3f>(v0,u0);
        Point3f tp0;
        tp0.x = (float)(p0.x * Rt_ptr[0] + p0.y * Rt_ptr[1] + p0.z * Rt_ptr[2] + Rt_ptr[3]);
        tp0.y = (float)(p0.x * Rt_ptr[4] + p0.y * Rt_ptr[5] + p0.z * Rt_ptr[6] + Rt_ptr[7]);
        tp0.z = (float)(p0.x * Rt_ptr[8] + p0.y * Rt_ptr[9] + p0.z * Rt_ptr[10] + Rt_ptr[11]);

        Vec3f n1 = normals1.at<Vec3f>(v1, u1);
        Point3f v = cloud1.at<Point3f>(v1,u1) - tp0;

        tps0_ptr[correspIndex] = tp0;
        diffs_ptr[correspIndex] = n1[0] * v.x + n1[1] * v.y + n1[2] * v.z;
        //std::cout << "====================test=======================" << diffs_ptr[0] <<  std::endl;
        //exit(1);
        sigma += diffs_ptr[correspIndex] * diffs_ptr[correspIndex];
    }

    cout << sigma << endl;
    sigma = std::sqrt(sigma/correspsCount);
    cout << sigma << endl;

    //std::vector<double> A_buf(transformDim);
    std::vector<float> A_buf(transformDim);
    //double* A_ptr = &A_buf[0];
    float* A_ptr = &A_buf[0];
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u1 = c[2], v1 = c[3];

        //double w = sigma + std::abs(diffs_ptr[correspIndex]);
        float w = sigma + std::abs(diffs_ptr[correspIndex]);
        w = w > DBL_EPSILON ? 1./w : 1.;
        cout << "w"  << w  << endl;

        func(A_ptr, tps0_ptr[correspIndex], normals1.at<Vec3f>(v1, u1) * w);

        for(int y = 0; y < transformDim; y++)
        {
            //double* AtA_ptr = AtA.ptr<double>(y);
            float* AtA_ptr = AtA.ptr<float>(y);
            for(int x = y; x < transformDim; x++)
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];

            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
        }
    }

    for(int y = 0; y < transformDim; y++)
        for(int x = y+1; x < transformDim; x++)
            //AtA.at<double>(x,y) = AtA.at<double>(y,x);
            AtA.at<float>(x,y) = AtA.at<float>(y,x);
    cout << AtA << endl;
    cout << AtB << endl;
    exit(1);
    //cout << "==================test====================" << AtA <<  endl;
    //cout << "==================test====================" << AtB <<  endl;
    //exit(1);
}
*/

static
void solveSystem(vector<FixedPointScalar>& A_vec, vector<FixedPointScalar>& B_vec, double detThreshold, Mat& x)
{
    //vector<FixedPointScalar> A_vec;
    //A_vec = i_Mat2Vec(AtA, fpconfig);
    //vector<FixedPointScalar> B_vec;
    //B_vec = i_Mat2Vec(AtB, fpconfig);
    FixedPointScalar zero_fix((FIXP_SCALAR_TYPE)0, fpconfig);
    vector<FixedPointScalar> A_vec2(6*6, zero_fix);
    vector<FixedPointScalar> B_vec2(6, zero_fix);
    //Mat AtA = Vec2Mat_f(A_vec, 6,6);
    //cout << AtA << endl;

    //cout << "===========DIVUU===================== " << A_vec[0].value << endl;
    int rows = 6;
    int cols = 6;
    A_vec2[0] = A_vec[0];
    A_vec2[1] = A_vec[1];
    A_vec2[2] = A_vec[2];
    A_vec2[3] = A_vec[3];
    A_vec2[4] = A_vec[4];
    A_vec2[5] = A_vec[5];
    A_vec2[6] = A_vec[1]/A_vec[0];
    A_vec2[12] = A_vec[2]/A_vec[0];
    A_vec2[18] = A_vec[3]/A_vec[0];
    A_vec2[24] = A_vec[4]/A_vec[0];
    A_vec2[30] = A_vec[5]/A_vec[0];
    for(int k = 0; k < rows; k++)
    {
        for(int m = 0; m < k; m++)
        {   
            if(m==0)
            {
                A_vec2[k*cols + k] = A_vec[k*cols + k] - (A_vec2[m*cols + k] * A_vec2[k*cols + m]);
                //cout << "A " << A_vec2[m*cols + k].value_floating << endl;
                //cout << "A " << A_vec2[m*cols + k].value << endl;
                //A_vec2[m*cols + k].print_big_value();
                //cout << "A " << A_vec2[k*cols + m].value_floating << endl;
                //cout << "A " << A_vec2[k*cols + m].value << endl;
                //A_vec2[k*cols + m].print_big_value();
                //cout << "tt " << (k*cols + m) << endl;
 
            }
            else
                A_vec2[k*cols + k] = A_vec2[k*cols + k] - (A_vec2[m*cols + k] * A_vec2[k*cols + m]);
        }
        
        for(int i = k+1; i < cols; i++)
        {
            for(int m = 0; m < k; m++)
            {
                if(m==0)
                    A_vec2[k*cols + i] = A_vec[k*cols + i] - (A_vec2[m*cols + i] * A_vec2[k*cols + m]);
                else
                    A_vec2[k*cols + i] = A_vec2[k*cols + i] - (A_vec2[m*cols + i] * A_vec2[k*cols + m]);
            }
            if(A_vec2[k*cols + k].value==0)
                cout << "===========DIV 0===================== " << k << A_vec2[k*cols + k].value << A_vec[k*cols + k].value << endl;
          
            A_vec2[i*cols + k] = A_vec2[k*cols + i] / A_vec2[k*cols + k] ;
        }

    }
    //Mat AtA = Vec2Mat_f(A_vec, 6,6);
    //cout << AtA << endl;
    //Mat AtA2 = Vec2Mat_f(A_vec2, 6,6);
    //cout << AtA2 << endl;
    //exit(1);

    B_vec2[0] = B_vec[0];
    for(int i = 0; i < rows; i++)
    {
        for(int k = 0; k < i; k++)
        {
            if(k==0)
                B_vec2[i] = B_vec[i] - (A_vec2[i*cols + k]*B_vec2[k]) ;
            else
                B_vec2[i] = B_vec2[i] - (A_vec2[i*cols + k]*B_vec2[k]) ;
        }
    }

    for(int i = rows-1; i >= 0; i--)
    {
        if(A_vec2[i*cols + i].value==0)
            cout << "===========DIV 1===================== " << endl;
        B_vec2[i] = B_vec2[i] / A_vec2[i*cols + i];
        for(int k = i+1; k < rows; k++)
        {
            B_vec2[i] = B_vec2[i] - (A_vec2[k*cols + i]*B_vec2[k]) ;
        }
    }


    x = Vec2Mat_f(B_vec2, 6, 1);
    if(cvIsNaN(x.at<float>(0,0)))
    {
        cout << "x " << x << endl;
        cout << "B " << B_vec2[0].value << endl;
        cout << "B " << B_vec2[0].value_floating << endl;
    }
    //cout << x << endl;
    //cout << "B " << B_vec2[0].value_floating << endl;
    //cout << "B " << B_vec2[1].value_floating << endl;
    //cout << "B " << B_vec2[2].value_floating << endl;
    //cout << "B " << B_vec2[3].value_floating << endl;
    //cout << "B " << B_vec2[4].value_floating << endl;
    //cout << "B " << B_vec2[5].value_floating << endl;
    //cout << "B " << B_vec2[0].value << endl;
    //cout << "B " << B_vec2[1].value << endl;
    //cout << "B " << B_vec2[2].value << endl;
    //cout << "B " << B_vec2[3].value << endl;
    //cout << "B " << B_vec2[4].value << endl;
    //cout << "B " << B_vec2[5].value << endl;
    //B_vec2[0].print_big_value();
    //B_vec2[1].print_big_value();
    //B_vec2[2].print_big_value();
    //B_vec2[3].print_big_value();
    //B_vec2[4].print_big_value();
    //B_vec2[5].print_big_value();
    //exit(1);

}

/*
static
//bool solveSystem(const Mat& AtA, const Mat& AtB, double detThreshold, Mat& x)
void solveSystem(vector<FixedPointScalar>& A_vec, vector<FixedPointScalar>& B_vec, double detThreshold, Mat& x)
{
    Mat AtA = Vec2Mat_f(A_vec, 6, 6);
    Mat AtB = Vec2Mat_f(B_vec, 6, 1);
    //double det = determinant(AtA);
    //cout << AtA << endl;
    //cout << AtB << endl;
    //exit(1);
    //if(fabs (det) < detThreshold || cvIsNaN(det) || cvIsInf(det))
    //    return false;

    solve(AtA, AtB, x, DECOMP_CHOLESKY);
    //cout << x << endl;
    //exit(1);
    //return true;
}
*/
static
bool testDeltaTransformation(const Mat& deltaRt, double maxTranslation, double maxRotation)
{
    double translation = norm(deltaRt(Rect(3, 0, 1, 3)));

    Mat rvec;
    Rodrigues(deltaRt(Rect(0,0,3,3)), rvec);

    double rotation = norm(rvec) * 180. / CV_PI;

    return translation <= maxTranslation && rotation <= maxRotation;
}

static
void computeProjectiveMatrix(const Mat& ksi, Mat& Rt)
{
    //CV_Assert(ksi.size() == Size(1,6) && ksi.type() == CV_64FC1);
    //cout << ksi << endl;
    //cout << ksi.type() << endl;
    CV_Assert(ksi.size() == Size(1,6) && ksi.type() == CV_32FC1);

#ifdef HAVE_EIGEN3_HERE
    const double* ksi_ptr = ksi.ptr<const double>();
    Eigen::Matrix<double,4,4> twist, g;
    twist << 0.,          -ksi_ptr[2], ksi_ptr[1],  ksi_ptr[3],
             ksi_ptr[2],  0.,          -ksi_ptr[0], ksi_ptr[4],
             -ksi_ptr[1], ksi_ptr[0],  0,           ksi_ptr[5],
             0.,          0.,          0.,          0.;
    g = twist.exp();

    eigen2cv(g, Rt);
#else
    // TODO: check computeProjectiveMatrix when there is not eigen library,
    //       because it gives less accurate pose of the camera
    //Rt = Mat::eye(4, 4, CV_64FC1);
    Rt = Mat::eye(4, 4, CV_32FC1);
    Mat ksi_32;
    ksi.convertTo(ksi_32, CV_32F);

    Mat R = Rt(Rect(0,0,3,3));
    //Mat rvec = ksi.rowRange(0,3);
    Mat rvec = ksi_32.rowRange(0,3);

    Rodrigues(rvec, R);

    //Rt.at<double>(0,3) = ksi.at<double>(3);
    //Rt.at<double>(1,3) = ksi.at<double>(4);
    //Rt.at<double>(2,3) = ksi.at<double>(5);
    Rt.at<float>(0,3) = ksi_32.at<float>(3);
    Rt.at<float>(1,3) = ksi_32.at<float>(4);
    Rt.at<float>(2,3) = ksi_32.at<float>(5);
#endif
}

static inline
void calcRgbdEquationCoeffs(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz  = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] =  p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
    C[3] = v0;
    C[4] = v1;
    C[5] = v2;
}

static inline
//void calcICPEquationCoeffs(double* C, const Point3f& p0, const Vec3f& n1)
void calcICPEquationCoeffs(float* C, const Point3f& p0, const Vec3f& n1)
{
    C[0] = -p0.z * n1[1] + p0.y * n1[2];
    C[1] =  p0.z * n1[0] - p0.x * n1[2];
    C[2] = -p0.y * n1[0] + p0.x * n1[1];
    C[3] = n1[0];
    C[4] = n1[1];
    C[5] = n1[2];
}

bool Odometry::compute(Ptr<OdometryFrame>& srcFrame, Ptr<OdometryFrame>& dstFrame, Mat& Rt, const Mat& initRt) const
{
    Size srcSize = prepareFrameCache(srcFrame, OdometryFrame::CACHE_SRC);
    Size dstSize = prepareFrameCache(dstFrame, OdometryFrame::CACHE_DST);

    if(srcSize != dstSize)
        CV_Error(Error::StsBadSize, "srcFrame and dstFrame have to have the same size (resolution).");

    int transformDim = 6;
    CalcRgbdEquationCoeffsPtr rgbdEquationFuncPtr = calcRgbdEquationCoeffs;
    CalcICPEquationCoeffsPtr icpEquationFuncPtr = calcICPEquationCoeffs;

    std::vector<int> iterCounts_vec = iterCounts;

    const int minOverdetermScale = 20;
    const int minCorrespsCount = minOverdetermScale * transformDim;
    //const float icpWeight = 10.0;

    std::vector<Mat> pyramidCameraMatrix;
    buildPyramidCameraMatrix(cameraMatrix, (int)iterCounts_vec.size(), pyramidCameraMatrix);

    //Mat resultRt = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();
    Mat resultRt = initRt.empty() ? Mat::eye(4,4,CV_32FC1) : initRt.clone();
    Mat currRt, ksi;

    bool isOk = false;
    for(int level = (int)iterCounts_vec.size() - 1; level >= 0; level--)
    {
        const Mat& levelCameraMatrix = pyramidCameraMatrix[level];
        const Mat& levelCameraMatrix_inv = levelCameraMatrix.inv(DECOMP_SVD);
        const Mat& srcLevelDepth = srcFrame->pyramidDepth[level];
        const Mat& dstLevelDepth = dstFrame->pyramidDepth[level];
        //const Vector<FixedPointScalar>& srcLevelDepth = srcFrame->pyramidDepth[level].to_vector();
        //const Vector<FixedPointScalar>& dstLevelDepth = dstFrame->pyramidDepth[level].to_vector();

        const double fx = levelCameraMatrix.at<double>(0,0);
        const double fy = levelCameraMatrix.at<double>(1,1);
        const double determinantThreshold = 1e-6;

        Mat AtA_rgbd, AtB_rgbd, AtA_icp, AtB_icp;
        Mat corresps_rgbd, corresps_icp;

        // Run transformation search on current level iteratively.
        for(int iter = 0; iter < iterCounts_vec[level]; iter ++)
        {
            Mat resultRt_inv = resultRt.inv(DECOMP_SVD);

            //computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
            //                srcLevelDepth, srcFrame->pyramidMask[level], dstLevelDepth, dstFrame->pyramidTexturedMask[level],
            //                maxDepthDiff, corresps_rgbd);
            
            computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                            srcLevelDepth, srcFrame->pyramidMask[level], dstLevelDepth, dstFrame->pyramidNormalsMask[level],
                            maxDepthDiff, corresps_icp);
            //cout << corresps_icp << endl;
            cout << "corresps_icp " << corresps_icp.rows << endl;
            //if(iter == 1)
            //  exit(1);
            //Mat AtA(transformDim, transformDim, CV_64FC1, Scalar(0)), AtB(transformDim, 1, CV_64FC1, Scalar(0));
            Mat AtA(transformDim, transformDim, CV_32FC1, Scalar(0)), AtB(transformDim, 1, CV_32FC1, Scalar(0));
            FixedPointScalar zero_fix((int64_t)0, fpconfig);
            vector<FixedPointScalar> A_vec(transformDim*transformDim, zero_fix);
            vector<FixedPointScalar> B_vec(transformDim, zero_fix);
            //if(corresps_rgbd.rows >= minCorrespsCount)
            //{
            //    calcRgbdLsmMatrices(srcFrame->pyramidImage[level], srcFrame->pyramidCloud[level], resultRt,
            //                        dstFrame->pyramidImage[level], dstFrame->pyramid_dI_dx[level], dstFrame->pyramid_dI_dy[level],
            //                        corresps_rgbd, fx, fy, sobelScale,
            //                        AtA_rgbd, AtB_rgbd, rgbdEquationFuncPtr, transformDim);

            //    AtA += AtA_rgbd;
            //    AtB += AtB_rgbd;
            //}
            if(corresps_icp.rows >= minCorrespsCount)
            {
                calcICPLsmMatrices(srcFrame->pyramidCloud[level], resultRt,
                                   dstFrame->pyramidCloud[level], dstFrame->pyramidNormals[level],
                                   corresps_icp, A_vec, B_vec, icpEquationFuncPtr, transformDim);
                //                   corresps_icp, AtA_icp, AtB_icp, icpEquationFuncPtr, transformDim);
                //AtA += AtA_icp;
                //AtB += AtB_icp;
                solveSystem(A_vec, B_vec, determinantThreshold, ksi);
            }
            else
            {
                cout << "===================no calcICPlsm===============" << corresps_icp.rows << endl;
                //break;
            }
            //bool solutionExist = solveSystem(AtA, AtB, determinantThreshold, ksi);
            //if(!solutionExist)
            //    break;
            computeProjectiveMatrix(ksi, currRt);
            resultRt = currRt * resultRt;
            cout << AtA << endl;
            cout << AtB << endl;
            cout << "currRt " << currRt << endl;
            cout << "resultRt " << resultRt << endl;
            if(iter == 1)
              exit(1);
            if(cvIsNaN(ksi.at<float>(0,0)))
            {
                cout << "ksi " << ksi << endl;
                Mat AtA = Vec2Mat_f(A_vec, transformDim, transformDim); //float
                Mat AtB = Vec2Mat_f(B_vec, transformDim, 1); //float
                cout << AtA << endl;
                cout << AtB << endl;
                cout << "currRt " << currRt << endl;
                cout << "resultRt " << resultRt << endl;
                exit(1);
            }
            isOk = true;
        }
        //exit(1);
    }

    Rt = resultRt;

    if(isOk)
    {
        Mat deltaRt;
        if(initRt.empty())
            deltaRt = resultRt;
        else
            deltaRt = resultRt * initRt.inv(DECOMP_SVD);

        isOk = testDeltaTransformation(deltaRt, maxTranslation, maxRotation);
        if(!isOk)
        {
            cout << "deltaRt " << deltaRt << endl;
            cout << "resultRt " << resultRt << endl;
            cout << "initRt " << initRt << endl;
        }
    }

    return isOk;
}


