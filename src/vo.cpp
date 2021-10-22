#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <gmpxx.h>
#include <gmp.h>

#include "vo.hpp"
//#include "fixed_point_util.hpp"

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
            if(!(norm.value==0))
            {
                FixedPointScalar n_x_final = n_x / norm;
                FixedPointScalar n_y_final = n_y / norm;
                FixedPointScalar n_z_final = n_z / norm;
                FixedPointVector normal(n_x_final, n_y_final, n_z_final);   
                normals[y*cols + x] = normal;
            }
        }
    }
  }
  return normals;
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

    pyramidNormals_test.clear();
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
vector<FixedPointVector> buildDnVec(vector<FixedPointVector>& dat, int rows, int cols)
{
    FixedPointScalar zero_fix((FIXP_SCALAR_TYPE)0, fpconfig);
    FixedPointVector zero_vec(zero_fix, zero_fix, zero_fix);
    vector<FixedPointVector> zero_mtx((rows+4)*(cols+4), zero_vec);
    vector<FixedPointVector> dat_filter(rows*cols, zero_vec);
    vector<FixedPointVector> datDn((rows/2)*(cols/2), zero_vec);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            zero_mtx[(r+2)*cols + (c+2)].x = dat[r*cols + c].x;
            zero_mtx[(r+2)*cols + (c+2)].y = dat[r*cols + c].y;
            zero_mtx[(r+2)*cols + (c+2)].z = dat[r*cols + c].z;
        }
    }
    for (int r = 0; r < rows; r++) {
        zero_mtx[(r+2)*cols + 0].x = dat[r*cols + 2].x;
        zero_mtx[(r+2)*cols + 0].y = dat[r*cols + 2].y;
        zero_mtx[(r+2)*cols + 0].z = dat[r*cols + 2].z;
        zero_mtx[(r+2)*cols + 1].x = dat[r*cols + 1].x;
        zero_mtx[(r+2)*cols + 1].y = dat[r*cols + 1].y;
        zero_mtx[(r+2)*cols + 1].z = dat[r*cols + 1].z;
        zero_mtx[(r+2)*cols + (cols+2)].x = dat[r*cols + (cols-2)].x;
        zero_mtx[(r+2)*cols + (cols+2)].y = dat[r*cols + (cols-2)].y;
        zero_mtx[(r+2)*cols + (cols+2)].z = dat[r*cols + (cols-2)].z;
        zero_mtx[(r+2)*cols + (cols+3)].x = dat[r*cols + (cols-3)].x;
        zero_mtx[(r+2)*cols + (cols+3)].y = dat[r*cols + (cols-3)].y;
        zero_mtx[(r+2)*cols + (cols+3)].z = dat[r*cols + (cols-3)].z;
    }
    for (int c = 0; c < cols; c++) {
        zero_mtx[0*cols + c].x = zero_mtx[4*cols + c].x;
        zero_mtx[0*cols + c].y = zero_mtx[4*cols + c].y;
        zero_mtx[0*cols + c].z = zero_mtx[4*cols + c].z;
        zero_mtx[1*cols + c].x = zero_mtx[3*cols + c].x;
        zero_mtx[1*cols + c].y = zero_mtx[3*cols + c].y;
        zero_mtx[1*cols + c].z = zero_mtx[3*cols + c].z;
        zero_mtx[(rows+2)*cols + c].x = zero_mtx[rows*cols + c].x;
        zero_mtx[(rows+2)*cols + c].y = zero_mtx[rows*cols + c].y;
        zero_mtx[(rows+2)*cols + c].z = zero_mtx[rows*cols + c].z;
        zero_mtx[(rows+3)*cols + c].x = zero_mtx[(rows-1)*cols + c].x;
        zero_mtx[(rows+3)*cols + c].y = zero_mtx[(rows-1)*cols + c].y;
        zero_mtx[(rows+3)*cols + c].z = zero_mtx[(rows-1)*cols + c].z;
    }
    
    FixedPointScalar c00((FIXP_SCALAR_TYPE)1, fpconfig);
    FixedPointScalar c01((FIXP_SCALAR_TYPE)4, fpconfig);
    FixedPointScalar c02((FIXP_SCALAR_TYPE)6, fpconfig);
    FixedPointScalar c03((FIXP_SCALAR_TYPE)4, fpconfig);
    FixedPointScalar c04((FIXP_SCALAR_TYPE)1, fpconfig);
    FixedPointScalar c10((FIXP_SCALAR_TYPE)4, fpconfig);
    FixedPointScalar c11((FIXP_SCALAR_TYPE)16, fpconfig);
    FixedPointScalar c12((FIXP_SCALAR_TYPE)24, fpconfig);
    FixedPointScalar c13((FIXP_SCALAR_TYPE)16, fpconfig);
    FixedPointScalar c14((FIXP_SCALAR_TYPE)4, fpconfig);
    FixedPointScalar c20((FIXP_SCALAR_TYPE)6, fpconfig);
    FixedPointScalar c21((FIXP_SCALAR_TYPE)24, fpconfig);
    FixedPointScalar c22((FIXP_SCALAR_TYPE)36, fpconfig);
    FixedPointScalar c23((FIXP_SCALAR_TYPE)24, fpconfig);
    FixedPointScalar c24((FIXP_SCALAR_TYPE)6, fpconfig);
    FixedPointScalar d((FIXP_SCALAR_TYPE)256, fpconfig);
    
    for (int r = 2; r < rows+2; r++) {
        for (int c = 2; c < cols+2; c++) {
            FixedPointScalar c0x = zero_mtx[(r-2)*cols + (c-2)].x*c00 + zero_mtx[(r-2)*cols + (c-1)].x*c01 + zero_mtx[(r-2)*cols + c].x*c02 + zero_mtx[(r-2)*cols + (c+1)].x*c03 + zero_mtx[(r-2)*cols + (c+2)].x*c04 ;
            FixedPointScalar c0y = zero_mtx[(r-2)*cols + (c-2)].y*c00 + zero_mtx[(r-2)*cols + (c-1)].y*c01 + zero_mtx[(r-2)*cols + c].y*c02 + zero_mtx[(r-2)*cols + (c+1)].y*c03 + zero_mtx[(r-2)*cols + (c+2)].y*c04 ;
            FixedPointScalar c0z = zero_mtx[(r-2)*cols + (c-2)].z*c00 + zero_mtx[(r-2)*cols + (c-1)].z*c01 + zero_mtx[(r-2)*cols + c].z*c02 + zero_mtx[(r-2)*cols + (c+1)].z*c03 + zero_mtx[(r-2)*cols + (c+2)].z*c04 ;
            FixedPointScalar c1x = zero_mtx[(r-1)*cols + (c-2)].x*c10 + zero_mtx[(r-1)*cols + (c-1)].x*c11 + zero_mtx[(r-1)*cols + c].x*c12 + zero_mtx[(r-1)*cols + (c+1)].x*c13 + zero_mtx[(r-1)*cols + (c+2)].x*c14 ;
            FixedPointScalar c1y = zero_mtx[(r-1)*cols + (c-2)].y*c10 + zero_mtx[(r-1)*cols + (c-1)].y*c11 + zero_mtx[(r-1)*cols + c].y*c12 + zero_mtx[(r-1)*cols + (c+1)].y*c13 + zero_mtx[(r-1)*cols + (c+2)].y*c14 ;
            FixedPointScalar c1z = zero_mtx[(r-1)*cols + (c-2)].z*c10 + zero_mtx[(r-1)*cols + (c-1)].z*c11 + zero_mtx[(r-1)*cols + c].z*c12 + zero_mtx[(r-1)*cols + (c+1)].z*c13 + zero_mtx[(r-1)*cols + (c+2)].z*c14 ;
            FixedPointScalar c2x = zero_mtx[r*cols + (c-2)].x*c20 + zero_mtx[r*cols + (c-1)].x*c21 + zero_mtx[r*cols + c].x*c22 + zero_mtx[r*cols + (c+1)].x*c23 + zero_mtx[r*cols + (c+2)].x*c24 ;
            FixedPointScalar c2y = zero_mtx[r*cols + (c-2)].y*c20 + zero_mtx[r*cols + (c-1)].y*c21 + zero_mtx[r*cols + c].y*c22 + zero_mtx[r*cols + (c+1)].y*c23 + zero_mtx[r*cols + (c+2)].y*c24 ;
            FixedPointScalar c2z = zero_mtx[r*cols + (c-2)].z*c20 + zero_mtx[r*cols + (c-1)].z*c21 + zero_mtx[r*cols + c].z*c22 + zero_mtx[r*cols + (c+1)].z*c23 + zero_mtx[r*cols + (c+2)].z*c24 ;
            FixedPointScalar c3x = zero_mtx[(r+1)*cols + (c-2)].x*c10 + zero_mtx[(r+1)*cols + (c-1)].x*c11 + zero_mtx[(r+1)*cols + c].x*c12 + zero_mtx[(r+1)*cols + (c+1)].x*c13 + zero_mtx[(r+1)*cols + (c+2)].x*c14 ;
            FixedPointScalar c3y = zero_mtx[(r+1)*cols + (c-2)].y*c10 + zero_mtx[(r+1)*cols + (c-1)].y*c11 + zero_mtx[(r+1)*cols + c].y*c12 + zero_mtx[(r+1)*cols + (c+1)].y*c13 + zero_mtx[(r+1)*cols + (c+2)].y*c14 ;
            FixedPointScalar c3z = zero_mtx[(r+1)*cols + (c-2)].z*c10 + zero_mtx[(r+1)*cols + (c-1)].z*c11 + zero_mtx[(r+1)*cols + c].z*c12 + zero_mtx[(r+1)*cols + (c+1)].z*c13 + zero_mtx[(r+1)*cols + (c+2)].z*c14 ;
            FixedPointScalar c4x = zero_mtx[(r+2)*cols + (c-2)].x*c00 + zero_mtx[(r+2)*cols + (c-1)].x*c01 + zero_mtx[(r+2)*cols + c].x*c02 + zero_mtx[(r+2)*cols + (c+1)].x*c03 + zero_mtx[(r+2)*cols + (c+2)].x*c04 ;
            FixedPointScalar c4y = zero_mtx[(r+2)*cols + (c-2)].y*c00 + zero_mtx[(r+2)*cols + (c-1)].y*c01 + zero_mtx[(r+2)*cols + c].y*c02 + zero_mtx[(r+2)*cols + (c+1)].y*c03 + zero_mtx[(r+2)*cols + (c+2)].y*c04 ;
            FixedPointScalar c4z = zero_mtx[(r+2)*cols + (c-2)].z*c00 + zero_mtx[(r+2)*cols + (c-1)].z*c01 + zero_mtx[(r+2)*cols + c].z*c02 + zero_mtx[(r+2)*cols + (c+1)].z*c03 + zero_mtx[(r+2)*cols + (c+2)].z*c04 ;
            FixedPointScalar c_totalx = c0x + c1x + c2x + c3x + c4x;
            FixedPointScalar c_totaly = c0y + c1y + c2y + c3y + c4y;
            FixedPointScalar c_totalz = c0z + c1z + c2z + c3z + c4z;
            FixedPointScalar c_divx = c_totalx / d;
            FixedPointScalar c_divy = c_totaly / d;
            FixedPointScalar c_divz = c_totalz / d;

            FixedPointScalar n2_x = c_divx*c_divx;
            FixedPointScalar n2_y = c_divy*c_divy;
            FixedPointScalar n2_z = c_divz*c_divz;
            FixedPointScalar norm_pre = n2_x + n2_y + n2_z;
            FixedPointScalar norm = (norm_pre).sqrt();
            if(!(norm.value==0))
            {
                FixedPointScalar n_x_final = c_divx / norm;
                FixedPointScalar n_y_final = c_divy / norm;
                FixedPointScalar n_z_final = c_divz / norm;
                dat_filter[(r-2)*cols + (c-2)].x = n_x_final;
                dat_filter[(r-2)*cols + (c-2)].y = n_y_final;
                dat_filter[(r-2)*cols + (c-2)].z = n_z_final;
            }
            else
            {
                dat_filter[(r-2)*cols + (c-2)].x = zero_fix;
                dat_filter[(r-2)*cols + (c-2)].y = zero_fix;
                dat_filter[(r-2)*cols + (c-2)].z = zero_fix;
            }
        }
    }

    for (int r = 0; r < rows/2; r++) {
        for (int c = 0; c < cols/2; c++) {
            datDn[r*(cols/2) + c] = dat_filter[r*2*cols + c*2];
        }
    }
    //cout << PVec2Mat_f(zero_mtx, rows+4, cols+4);
    //cout << PVec2Mat_f(dat_filter, rows, cols);
    //exit(1);
    return datDn;
}

static
//FixedPointMatrix buildDn(const FixedPointMatrix& dat)
FixedPointMatrix buildDn(FixedPointMatrix& dat)
{
    int rows = dat.value_floating.rows();
    int cols = dat.value_floating.cols();
    FixedPointScalar zero_fix((FIXP_SCALAR_TYPE)0, fpconfig);
    vector<FixedPointScalar> zero_vec((rows+4)*(cols+4), zero_fix);
    vector<FixedPointScalar> zero_vec2(rows*cols, zero_fix);
    vector<FixedPointScalar> zero_vec3((rows/2)*(cols/2), zero_fix);
    FixedPointMatrix zero_mtx (zero_vec, rows+4, cols+4);
    FixedPointMatrix dat_filter (zero_vec2, rows, cols);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            zero_mtx.assign(dat(r, c), r+2, c+2);
        }
    }
    for (int r = 0; r < rows; r++) {
        zero_mtx.assign(dat(r, 2), r+2, 0);
        zero_mtx.assign(dat(r, 1), r+2, 1);
        zero_mtx.assign(dat(r, cols-2), r+2, cols+2);
        zero_mtx.assign(dat(r, cols-3), r+2, cols+3);
    }
    for (int c = 0; c < cols; c++) {
        zero_mtx.assign(zero_mtx(4, c), 0, c);
        zero_mtx.assign(zero_mtx(3, c), 1, c);
        zero_mtx.assign(zero_mtx(rows, c), rows+2, c);
        zero_mtx.assign(zero_mtx(rows-1, c), rows+3, c);
    }

    FixedPointScalar c00((FIXP_SCALAR_TYPE)1, fpconfig);
    FixedPointScalar c01((FIXP_SCALAR_TYPE)4, fpconfig);
    FixedPointScalar c02((FIXP_SCALAR_TYPE)6, fpconfig);
    FixedPointScalar c03((FIXP_SCALAR_TYPE)4, fpconfig);
    FixedPointScalar c04((FIXP_SCALAR_TYPE)1, fpconfig);
    FixedPointScalar c10((FIXP_SCALAR_TYPE)4, fpconfig);
    FixedPointScalar c11((FIXP_SCALAR_TYPE)16, fpconfig);
    FixedPointScalar c12((FIXP_SCALAR_TYPE)24, fpconfig);
    FixedPointScalar c13((FIXP_SCALAR_TYPE)16, fpconfig);
    FixedPointScalar c14((FIXP_SCALAR_TYPE)4, fpconfig);
    FixedPointScalar c20((FIXP_SCALAR_TYPE)6, fpconfig);
    FixedPointScalar c21((FIXP_SCALAR_TYPE)24, fpconfig);
    FixedPointScalar c22((FIXP_SCALAR_TYPE)36, fpconfig);
    FixedPointScalar c23((FIXP_SCALAR_TYPE)24, fpconfig);
    FixedPointScalar c24((FIXP_SCALAR_TYPE)6, fpconfig);
    FixedPointScalar d((FIXP_SCALAR_TYPE)256, fpconfig);
    
    for (int r = 2; r < rows+2; r++) {
        for (int c = 2; c < cols+2; c++) {
            FixedPointScalar c0 = zero_mtx(r-2, c-2)*c00 + zero_mtx(r-2, c-1)*c01 + zero_mtx(r-2, c)*c02 + zero_mtx(r-2, c+1)*c03 + zero_mtx(r-2, c+2)*c04 ;
            FixedPointScalar c1 = zero_mtx(r-1, c-2)*c10 + zero_mtx(r-1, c-1)*c11 + zero_mtx(r-1, c)*c12 + zero_mtx(r-1, c+1)*c13 + zero_mtx(r-1, c+2)*c14 ;
            FixedPointScalar c2 = zero_mtx(r, c-2)*c20 + zero_mtx(r, c-1)*c21 + zero_mtx(r, c)*c22 + zero_mtx(r, c+1)*c23 + zero_mtx(r, c+2)*c24 ;
            FixedPointScalar c3 = zero_mtx(r+1, c-2)*c10 + zero_mtx(r+1, c-1)*c11 + zero_mtx(r+1, c)*c12 + zero_mtx(r+1, c+1)*c13 + zero_mtx(r+1, c+2)*c14 ;
            FixedPointScalar c4 = zero_mtx(r+2, c-2)*c00 + zero_mtx(r+2, c-1)*c01 + zero_mtx(r+2, c)*c02 + zero_mtx(r+2, c+1)*c03 + zero_mtx(r+2, c+2)*c04 ;
            FixedPointScalar c_total = c0 + c1 + c2 + c3 + c4;
            FixedPointScalar c_div = c_total / d;

            dat_filter.assign(c_div, r-2, c-2);
        }
    }
    
    //Mat test = Vec2Mat_f(zero_mtx.to_vector(), zero_mtx.value_floating.rows(), zero_mtx.value_floating.cols());
    //Mat test2 = Vec2Mat_f(dat.to_vector(), dat.value_floating.rows(), dat.value_floating.cols());
    //cout << test << endl;
    //exit(1);
    FixedPointMatrix datDn (zero_vec3, rows/2, cols/2);
    for (int r = 0; r < rows/2; r++) {
        for (int c = 0; c < cols/2; c++) {
            datDn.assign(dat_filter(r*2, c*2), r, c);
        }
    }
    return datDn;
}

static
void buildpy(const Mat& depth, std::vector<FixedPointMatrix>& pyramidDepth, size_t levelCount)
{
    if(pyramidDepth.empty())
    {
        //Mat tmp;
        //for( int i = 0; i < levelCount; i++ )
        //{   
        //    Mat dwn;
        //    vector<FixedPointScalar> depth_fixp;
        //    if(i != 0)
        //    {
        //        pyrDown(tmp, dwn);
        //        depth_fixp = f_Mat2Vec(dwn, fpconfig);
        //        FixedPointMatrix depth_mtx(depth_fixp, dwn.rows, dwn.cols);
        //        pyramidDepth.push_back(depth_mtx);
        //        tmp = dwn;
        //    }
        //    else
        //    {
        //        depth_fixp = f_Mat2Vec(depth, fpconfig);
        //        FixedPointMatrix depth_mtx(depth_fixp, depth.rows, depth.cols);
        //        pyramidDepth.push_back(depth_mtx);
        //        tmp = depth;
        //    }

        //}
        vector<FixedPointScalar> depth_fixp;
        depth_fixp = f_Mat2Vec(depth, fpconfig);
        FixedPointMatrix depth_mtx(depth_fixp, depth.rows, depth.cols);
        for( int i = 0; i < levelCount; i++ )
        {   
            if(i != 0)
            {
                FixedPointMatrix depthDn = buildDn(depth_mtx);
                pyramidDepth.push_back(depthDn);
                depth_mtx = depthDn;
            }
            else
            {
                pyramidDepth.push_back(depth_mtx);
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
         FixedPointScalar p_x((FIXP_SCALAR_TYPE)x, fpconfig);
         
         p_x = (p_x - cx);
         p_x = p_x * depth_vec[y*depth.value_floating.cols() + x];
         p_x = p_x / fx;
         
         FixedPointScalar p_y((FIXP_SCALAR_TYPE)y, fpconfig);

         p_y = (p_y - cy) / fy;
         p_y = p_y * depth_vec[y*depth.value_floating.cols() + x];

         FixedPointScalar p_z = depth_vec[y*depth.value_floating.cols() + x];

         FixedPointVector p3d(p_x, p_y, p_z);
         p3d_vec.push_back(p3d);
    }
  }
  return p3d_vec;
}

static
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
void preparePyramidNormals(const vector<FixedPointVector>& normals, const vector<FixedPointMatrix>& pyramidDepth, vector<vector<FixedPointVector>>& pyramidNormals)
{
    if(!pyramidNormals.empty())
    {
        if(pyramidNormals.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormals.");

    }
    else
    {
        for( int i = 0; i < pyramidDepth.size(); i++ )
        {   
            int rows = pyramidDepth[i].value_floating.rows();
            int cols = pyramidDepth[i].value_floating.cols();
            FixedPointScalar zero_fix((FIXP_SCALAR_TYPE)0, fpconfig);
            FixedPointVector zero_vec(zero_fix, zero_fix, zero_fix);
            vector<FixedPointVector> zero_mtx(rows*cols, zero_vec);
            pyramidNormals.push_back(zero_mtx);
        }
        for( int i = 0; i < pyramidDepth.size(); i++ )
        {   
            if(i != 0)
            {
                int rows = pyramidDepth[i-1].value_floating.rows();
                int cols = pyramidDepth[i-1].value_floating.cols();
                vector<FixedPointVector> normalDn = buildDnVec(pyramidNormals[i-1], rows, cols);
                pyramidNormals[i] = normalDn;
            }
            else
            {
                pyramidNormals[i] = normals;
            }
        }
    }
}

/*
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
*/

static
void preparePyramidMask(const Mat& mask, const vector<FixedPointMatrix>& pyramidDepth, float minDepth, float maxDepth,
                        const vector<vector<FixedPointVector>>& pyramidNormal,
                        std::vector<Mat>& pyramidMask)
{
    minDepth = std::max(0.f, minDepth);

    if(!pyramidMask.empty())
    {
        if(pyramidMask.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Levels count of pyramidMask has to be equal to size of pyramidDepth.");

        //for(size_t i = 0; i < pyramidMask.size(); i++)
        //{
        //    CV_Assert(pyramidMask[i].size() == pyramidDepth[i].size());
        //    CV_Assert(pyramidMask[i].type() == CV_8UC1);
        //}
    }
    else
    {
        Mat validMask;
        if(mask.empty())
            validMask = Mat(pyramidDepth[0].value_floating.rows(), pyramidDepth[0].value_floating.cols(), CV_8UC1, Scalar(255));
        else
            validMask = mask.clone();

        FIXP_INT_SCALAR_TYPE minDepth_fix = static_cast<FIXP_INT_SCALAR_TYPE>(minDepth * (1LL << shift));
        FIXP_INT_SCALAR_TYPE maxDepth_fix = static_cast<FIXP_INT_SCALAR_TYPE>(maxDepth * (1LL << shift));
        //cout << "liyang test" << validMask << endl;
        //exit(1);
        buildPyramid(validMask, pyramidMask, (int)pyramidDepth.size() - 1);

        for(size_t i = 0; i < pyramidMask.size(); i++)
        {
            //Mat levelDepth = Vec2Mat_f(pyramidDepth[i].to_vector(), pyramidDepth[i].value_floating.rows(), pyramidDepth[i].value_floating.cols());
            //patchNaNs(levelDepth, 0);

            Mat& levelMask = pyramidMask[i];
            //levelMask &= (levelDepth > minDepth) & (levelDepth < maxDepth);
            for (int r = 0; r < pyramidDepth[i].value_floating.rows(); r++) {
                for (int c = 0; c < pyramidDepth[i].value_floating.cols(); c++) {
                    if((pyramidDepth[i].value(r, c) > minDepth_fix) && (pyramidDepth[i].value(r, c) < maxDepth_fix))
                    //if((pyramidDepth[i].value_floating(r, c) > minDepth) && (pyramidDepth[i].value_floating(r, c) < maxDepth))
                        levelMask.at<uchar>(r, c) = 255;
                    else
                        levelMask.at<uchar>(r, c) = 0;
                }
            }
            //cout << "liyang test" << levelMask << endl;
            //cout << "liyang test" << pyramidDepth[i].value_floating.rows() << endl;
            //cout << "liyang test" << pyramidDepth[i].value_floating.cols() << endl;
            //cout << "liyang test" << maxDepth_fix << endl;
            //cout << "liyang test" << minDepth_fix << endl;
            //cout << "liyang test" << maxDepth << endl;
            //cout << "liyang test" << pyramidDepth[i].value_floating(479, 639) << endl;
            //cout << "liyang test" << pyramidDepth[i].value(479, 639) << endl;
            //exit(1);

            //if(!pyramidNormal.empty())
            //{
            //    //CV_Assert(pyramidNormal[i].type() == CV_32FC3);
            //    //CV_Assert(pyramidNormal[i].size() == pyramidDepth[i].size());
            //    Mat levelNormal = PVec2Mat_f(pyramidNormal[i], pyramidDepth[i].value_floating.rows(), pyramidDepth[i].value_floating.cols());

            //    Mat validNormalMask = levelNormal == levelNormal; // otherwise it's Nan
            //    //cout << "liyang test" << validNormalMask << endl;
            //    //exit(1);
            //    CV_Assert(validNormalMask.type() == CV_8UC3);

            //    std::vector<Mat> channelMasks;
            //    split(validNormalMask, channelMasks);
            //    validNormalMask = channelMasks[0] & channelMasks[1] & channelMasks[2];

            //    levelMask &= validNormalMask;
            //}
        }
    }
}
/*
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
*/
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
            //CV_Assert(pyramidNormalsMask[i].size() == pyramidMask[i].size());
            //CV_Assert(pyramidNormalsMask[i].type() == pyramidMask[i].type());
        }
    }
    else
    {
        pyramidNormalsMask.resize(pyramidMask.size());

        for(size_t i = 0; i < pyramidNormalsMask.size(); i++)
        {
            pyramidNormalsMask[i] = pyramidMask[i].clone();
            Mat& normalsMask = pyramidNormalsMask[i];
            //for(int y = 0; y < normalsMask.rows; y++)
            //{
            //    const Vec3f *normals_row = pyramidNormals[i].ptr<Vec3f>(y);
            //    uchar *normalsMask_row = pyramidNormalsMask[i].ptr<uchar>(y);
            //    for(int x = 0; x < normalsMask.cols; x++)
            //    {
            //        Vec3f n = normals_row[x];
            //        if(cvIsNaN(n[0]))
            //        {
            //            CV_DbgAssert(cvIsNaN(n[1]) && cvIsNaN(n[2]));
            //            normalsMask_row[x] = 0;
            //        }
            //    }
            //}
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
        vector<FixedPointVector> normals_test; 
        if(frame->normals.empty())
        {
            //if(!frame->pyramidNormals.empty())
                //frame->normals = frame->pyramidNormals[0];
            if(!frame->pyramidNormals_test.empty())
                normals_test = frame->pyramidNormals_test[0];
            else
            {
                //normalsComputer(frame->pyramidCloud[0], frame->depth.rows, frame->depth.cols, frame->normals);
                normals_test = normalsComputer(pyramidCloud_test[0], frame->depth.rows, frame->depth.cols);
                //frame->normals = PVec2Mat_f(normals_test, frame->depth.rows, frame->depth.cols);
            }
        }
        //cout << "normal done" << endl;
        //checkNormals(frame->normals, frame->depth.size());
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

        //preparePyramidNormals(frame->normals, frame->pyramidDepth, frame->pyramidNormals);
        //vector<vector<FixedPointVector>> pyramidNormals_test; 
        //preparePyramidNormals(normals_test, pyramidDepth_test, pyramidNormals_test);
        preparePyramidNormals(normals_test, pyramidDepth_test, frame->pyramidNormals_test);
        if(frame->pyramidNormals.empty())
        {
            for( int i = 0; i < iterCounts.total(); i++ )
            {
                //frame->pyramidNormals.push_back(PVec2Mat_f(pyramidNormals_test[i], pyramidDepth_test[i].value_floating.rows(), pyramidDepth_test[i].value_floating.cols()));
                frame->pyramidNormals.push_back(PVec2Mat_f(frame->pyramidNormals_test[i], pyramidDepth_test[i].value_floating.rows(), pyramidDepth_test[i].value_floating.cols()));
            }  
        }

        //preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
        //                   frame->pyramidNormals, frame->pyramidMask);
        preparePyramidMask(frame->mask, pyramidDepth_test, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals_test, frame->pyramidMask);

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
        //preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
        //                   frame->pyramidNormals, frame->pyramidMask);
        preparePyramidMask(frame->mask, pyramidDepth_test, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals_test, frame->pyramidMask);

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
  FixedPointScalar cx_inv (K_inv.at<float>(0,2), fpconfig);//float
  FixedPointScalar cy_inv (K_inv.at<float>(1,2), fpconfig);//float

  vector<FixedPointScalar> Rt_vec;
  Rt_vec = f_Mat2Vec(Rt, fpconfig);
  vector<FixedPointScalar> d0_vec;
  vector<FixedPointScalar> d1_vec;
  d0_vec = f_Mat2Vec(depth0, fpconfig);//float
  d1_vec = f_Mat2Vec(depth1, fpconfig);//float

  FixedPointScalar RK_inv_00 = Rt_vec[0]*fx_inv;
  FixedPointScalar RK_inv_01 = Rt_vec[1]*fy_inv;
  FixedPointScalar RK_inv_02 = Rt_vec[0]*cx_inv + Rt_vec[1]*cy_inv + Rt_vec[2];
  FixedPointScalar RK_inv_10 = Rt_vec[4]*fx_inv;
  FixedPointScalar RK_inv_11 = Rt_vec[5]*fy_inv;
  FixedPointScalar RK_inv_12 = Rt_vec[4]*cx_inv + Rt_vec[5]*cy_inv + Rt_vec[6];
  FixedPointScalar RK_inv_20 = Rt_vec[8]*fx_inv;
  FixedPointScalar RK_inv_21 = Rt_vec[9]*fy_inv;
  FixedPointScalar RK_inv_22 = Rt_vec[8]*cx_inv + Rt_vec[9]*cy_inv + Rt_vec[10];
  
  FixedPointScalar KRK_inv_00 = fx*RK_inv_00 + cx*RK_inv_20;
  FixedPointScalar KRK_inv_01 = fx*RK_inv_01 + cx*RK_inv_21;
  FixedPointScalar KRK_inv_02 = fx*RK_inv_02 + cx*RK_inv_22;
  FixedPointScalar KRK_inv_10 = fy*RK_inv_10 + cy*RK_inv_20;
  FixedPointScalar KRK_inv_11 = fy*RK_inv_11 + cy*RK_inv_21;
  FixedPointScalar KRK_inv_12 = fy*RK_inv_12 + cy*RK_inv_22;
  FixedPointScalar KRK_inv_20 = RK_inv_20;
  FixedPointScalar KRK_inv_21 = RK_inv_21;
  FixedPointScalar KRK_inv_22 = RK_inv_22;
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
             FixedPointScalar u1_shift ((FIXP_SCALAR_TYPE)u1, fpconfig);
             FixedPointScalar v1_shift ((FIXP_SCALAR_TYPE)v1, fpconfig);
             FixedPointScalar transformed_d1_shift = KRK_inv_20*u1_shift + KRK_inv_21*v1_shift + KRK_inv_22;
             transformed_d1_shift = (d1*transformed_d1_shift) + Kt_2;
             if(transformed_d1_shift.value > 0)
             {
                 FixedPointScalar u0_shift = KRK_inv_00*u1_shift + KRK_inv_01*v1_shift + KRK_inv_02;
                 FixedPointScalar ttt = KRK_inv_00*u1_shift + KRK_inv_01*v1_shift;
                 FixedPointScalar test_shift = ttt + KRK_inv_02;
                 FixedPointScalar v0_shift = KRK_inv_10*u1_shift + KRK_inv_11*v1_shift + KRK_inv_12;
                 u0_shift = (d1*u0_shift) + Kt_0;
                 v0_shift = (d1*v0_shift) + Kt_1;
                 u0_shift = u0_shift / transformed_d1_shift;
                 v0_shift = v0_shift / transformed_d1_shift;
                 int u0 = (int)round(u0_shift.value_floating);
                 int v0 = (int)round(v0_shift.value_floating); 
                 if(r.contains(Point(u0,v0)))
                 {
                     FixedPointScalar d0 = d0_vec[v0*cols + u0];
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
    
    FixedPointScalar correspsCount((FIXP_SCALAR_TYPE)corresps.rows, fpconfig);

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    vector<FixedPointScalar> Rt_vec;
    Rt_vec = f_Mat2Vec(Rt, fpconfig); //float

    vector<FixedPointScalar> diffs_ptr;
    vector<FixedPointVector> tps0_ptr;
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

    FixedPointScalar sigma_final = (sigma/correspsCount).sqrt();

    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u1 = c[2], v1 = c[3];
        
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
        FixedPointScalar c0 = neg_one * tp0.z * n1y + tp0.y * n1z;
        FixedPointScalar c1 = tp0.z * n1x - tp0.x * n1z;
        FixedPointScalar c2 = neg_one * tp0.y * n1x + tp0.x * n1y;
        C_vec[0] = c0;
        C_vec[1] = c1;
        C_vec[2] = c2;
        C_vec[3] = n1x;
        C_vec[4] = n1y;
        C_vec[5] = n1z;

        for(int y = 0; y < transformDim; y++)
        {
            for(int x = y; x < transformDim; x++)
            {
                FixedPointScalar  test = C_vec[y] * C_vec[x];
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
}

static
void solveSystem(vector<FixedPointScalar>& A_vec, vector<FixedPointScalar>& B_vec, double detThreshold, Mat& x)
{
    FixedPointScalar zero_fix((FIXP_SCALAR_TYPE)0, fpconfig);
    vector<FixedPointScalar> A_vec2(6*6, zero_fix);
    vector<FixedPointScalar> B_vec2(6, zero_fix);

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
}

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
            //cout << "corresps_icp " << corresps_icp.rows << endl;
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
            //cout << AtA << endl;
            //cout << AtB << endl;
            //cout << "currRt " << currRt << endl;
            //cout << "resultRt " << resultRt << endl;
            //if(iter == 1)
            //  exit(1);
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


