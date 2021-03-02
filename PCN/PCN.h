#ifndef __PCN__
#define __PCN__

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include "caffe/util/db.hpp"

namespace db = caffe::db;

#include "opencv2/opencv.hpp"
#include "caffe/caffe.hpp"

#define M_PI  3.14159265358979323846
#define CLAMP(x, l, u)  ((x) < (l) ? (l) : ((x) > (u) ? (u) : (x)))
#define EPS  1e-5

#define CYAN CV_RGB(0, 255, 255)
#define BLUE CV_RGB(0, 0, 255)
#define GREEN CV_RGB(0, 255, 0)
#define RED CV_RGB(255, 0, 0)
#define PURPLE CV_RGB(139, 0, 255)

//#define kFeaturePoints 14
#define kDescriptorLen 128

#define NET_STAGE1_WIN_SIZE 24
#define NET_STAGE2_WIN_SIZE 24
#define NET_STAGE3_WIN_SIZE 48
#define NET_TRACK_WIN_SIZE 96
#define NET_EMBED_WIN_SIZE 160

enum{
    FEAT_CHIN_0 = 0,
    FEAT_CHIN_1,
    FEAT_CHIN_2,
    FEAT_CHIN_3,
    FEAT_CHIN_4,
    FEAT_CHIN_5,
    FEAT_CHIN_6,
    FEAT_CHIN_7,
    FEAT_CHIN_8,
    FEAT_NOSE,
    FEAT_EYE_LEFT,
    FEAT_EYE_RIGHT,
    FEAT_MOUTH_LEFT,
    FEAT_MOUTH_RIGHT,
    kFeaturePoints
};

struct Window
{
    int x, y, width,height;
    float angle;
    float yaw; 
    float scale;
    float conf;
    long id;
    cv::Point points14[kFeaturePoints];
    float descriptor[kDescriptorLen];

    Window(int x_, int y_, int w_, int h_, float a_, float s_, float c_, long id_, 
		    cv::Point p14_[kFeaturePoints], float desc_[kDescriptorLen])
        : x(x_), y(y_), width(w_),height(h_), angle(a_), scale(s_), conf(c_), id(id_)
    {
	    set_points(p14_);
	    set_desc(desc_);
    }
    
    //New window without points and ID
    Window(int x_, int y_, int w_, int h_, float a_, float s_, float c_)
        : x(x_), y(y_), width(w_),height(h_), angle(a_), scale(s_), conf(c_), id(-1)
    {}

    void set_desc(float desc_[]) {
	    memcpy(descriptor,&(desc_[0]),kDescriptorLen*sizeof(float));
	    
    }
	
    void calculate_yaw_angle()
    {
	cv::Point vec1 = points14[FEAT_EYE_LEFT] - points14[FEAT_MOUTH_RIGHT];
	cv::Point vec2 = points14[FEAT_EYE_RIGHT] - points14[FEAT_MOUTH_LEFT];
	double ab = vec1.dot(vec2);
	double aa = vec1.dot(vec1);
	double bb = vec2.dot(vec2);
	yaw =  ab / sqrt(aa*bb);
    }

    void set_points(cv::Point p14_[]) {
	    memcpy(points14,&(p14_[0]),kFeaturePoints*sizeof(cv::Point));
	    calculate_yaw_angle();
    }
};

enum {
	NET_STAGE1=0,
	NET_STAGE2,
	NET_STAGE3,
	NET_TRACK,
	NET_EMBED
};

class PCN
{
public:
    PCN(std::string modelDetect, std::string net1, std::string net2, std::string net3,
	    std::string modelTrack, std::string netTrack,
	    std::string modelEmbed, std::string netEmbed);
    /// Get/Set
    void SetMinFaceSize(int minFace);
    void SetDetectionThresh(float thresh1, float thresh2, float thresh3);
    void SetImagePyramidScaleFactor(float factor);
    void SetEmbedding(int doEmbed);
    void SetTrackingPeriod(int period);
    int GetTrackingPeriod();
    void SetTrackingThresh(float thresh);
    std::vector<Window> GenerateThirdLayerInput(cv::Mat img, db::DB *db, float image_label);
    float ProcessSingleImageStage3(std::vector<cv::Mat> image);
    /// detection
    std::vector<Window> Detect(cv::Mat img);
    /// tracking
    std::vector<Window> DetectTrack(cv::Mat img);
    /// embedding
    static cv::Mat CropFace(cv::Mat img, Window face, int cropSize);
    static void DrawPoints(cv::Mat img, Window face); 
    static void DrawFace(cv::Mat img, Window face);
    static void DrawLine(cv::Mat img, std::vector<cv::Point> pointList);
    static cv::Point RotatePoint(float x, float y, float centerX, float centerY, float angle);
private:
    void LoadModel_(std::string modelDetect, std::string net1, std::string net2, std::string net3,
		    std::string modelTrack, std::string netTrack,
		    std::string modelEmbed, std::string netEmbed);
    void SetDeafultValues_();
    cv::Mat ResizeImg_(cv::Mat img, float scale);
    static bool CompareWin_(const Window &w1, const Window &w2);
    bool Legal_(int x, int y, cv::Mat img);
    bool Inside_(int x, int y, Window rect);
    int SmoothAngle_(int a, int b);
    std::vector<Window> SmoothWindowWithId_(std::vector<Window> winList);
    float IoU_(Window &w1, Window &w2);
    std::vector<Window> NMS_(std::vector<Window> &winList, bool local, float threshold);
    std::vector<Window> DeleteFP_(std::vector<Window> &winList);
    cv::Mat PreWhiten_(cv::Mat img);
    cv::Mat PreProcessImg_(cv::Mat img);
    cv::Mat PreProcessImg_(cv::Mat img,  int dim);
    void SetInput_(cv::Mat input, caffe::shared_ptr<caffe::Net<float> > &net);
    void SetInput_(std::vector<cv::Mat> &input, caffe::shared_ptr<caffe::Net<float> > &net);
    cv::Mat PadImg_(cv::Mat img);
    std::vector<Window> TransWindow_(cv::Mat img, cv::Mat imgPad, std::vector<Window> &winList);
    std::vector<Window> Stage1_(cv::Mat img, cv::Mat imgPad, caffe::shared_ptr<caffe::Net<float> > &net, float thres);
    std::vector<Window> Stage2_(cv::Mat img, cv::Mat img180,
                                caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window> &winList);
    std::vector<Window> Stage3_(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90,
                                caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window> &winList);
    std::vector<Window> Embed_(cv::Mat img, caffe::shared_ptr<caffe::Net<float> > &net,std::vector<Window> &winList,int dim);
    std::vector<Window> Detect_(cv::Mat img, cv::Mat imgPad);
    std::vector<Window> Track_(cv::Mat img, caffe::shared_ptr<caffe::Net<float> > &net,
                               float thres, int dim, std::vector<Window> &winList);
    void DumpImage_(std::vector<cv::Mat> &input, db::DB *db, float image_label);
    void PartialStage3_(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90, caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window> &winList, db::DB *db, float image_label);
    float OnlyStage3_(std::vector<cv::Mat> dataList, caffe::shared_ptr<caffe::Net<float> > &net);

    //private data
    caffe::shared_ptr<caffe::Net<float> > net_[5];
    int minFace_;
    float scale_;
    int stride_;
    float classThreshold_[3];
    float nmsThreshold_[3];
    float angleRange_;
    int period_;
    float trackThreshold_;
    float augScale_;
    cv::Scalar mean_;
    std::vector<Window> preList_;
    int trackPeriod_;
    long global_id_; //Global ID incrementor
    //Flags
    int doEmbed_;
};

#endif
