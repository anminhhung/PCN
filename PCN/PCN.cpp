#include "PCN.h"
#include "PCN_API.h"
#include "lmdb.h"

PCN::PCN(std::string modelDetect, std::string net1, std::string net2, std::string net3,
         std::string modelTrack, std::string netTrack,
         std::string modelEmbed, std::string netEmbed) 
{
    LoadModel_(modelDetect, net1, net2, net3, 
		    modelTrack, netTrack,modelEmbed,netEmbed);
    SetDeafultValues_();
}

void PCN::SetDeafultValues_(){ 
    global_id_ = 0;
    minFace_ = 40*1.4;
    classThreshold_[NET_STAGE1] = 0.5;
    classThreshold_[NET_STAGE2] = 0.5;
    classThreshold_[NET_STAGE3] = 0.98;
    nmsThreshold_[NET_STAGE1] = 0.8;
    nmsThreshold_[NET_STAGE2] = 0.8;
    nmsThreshold_[NET_STAGE3] = 0.3;
    stride_ = 8;
    angleRange_ = 45;
    augScale_ = 0.15;
    mean_ = cv::Scalar(104, 117, 123);
    scale_ = 1.45;
    period_ = 30;
    trackPeriod_ = 0;
    trackThreshold_ = 0.9; 
    doEmbed_ = 0;
}

void PCN::SetMinFaceSize(int minFace)
{
    minFace_ = minFace * 1.4;
}

void PCN::SetDetectionThresh(float thresh1, float thresh2, float thresh3)
{
    classThreshold_[NET_STAGE1] = thresh1;
    classThreshold_[NET_STAGE2] = thresh2;
    classThreshold_[NET_STAGE3] = thresh3;
}

void PCN::SetImagePyramidScaleFactor(float factor)
{
    scale_ = factor;
}

void PCN::SetTrackingPeriod(int period)
{
    period_ = period;
    trackPeriod_ = 0;
}

void PCN::SetTrackingThresh(float thres)
{
    trackThreshold_ = thres;
}

void PCN::SetEmbedding(int doEmbed){
	doEmbed_ = doEmbed;
}

std::vector<Window> PCN::Detect(cv::Mat img)
{
    cv::Mat imgPad = PadImg_(img);
    std::vector<Window> winList = Detect_(img, imgPad);
    std::vector<Window> pointsList = Track_(imgPad, net_[NET_TRACK], -1, NET_TRACK_WIN_SIZE, winList);
    for (int i = 0; i < winList.size(); i++)
	winList[i].set_points(pointsList[i].points14);
    winList = TransWindow_(img, imgPad, winList);
    return winList;
}

std::vector<Window> PCN::DetectTrack(cv::Mat img)
{
    cv::Mat imgPad = PadImg_(img);
    std::vector<Window> winList = preList_;
    if (trackPeriod_ == 0)
    {
        std::vector<Window> tmpList = Detect_(img, imgPad);

        for (int i = 0; i < tmpList.size(); i++)
        {
            winList.push_back(tmpList[i]);
        }
    }
    winList = NMS_(winList, false, nmsThreshold_[NET_STAGE3]);
    winList = Track_(imgPad, net_[NET_TRACK], trackThreshold_, NET_TRACK_WIN_SIZE, winList);
    winList = NMS_(winList, false, nmsThreshold_[NET_STAGE3]);
    winList = DeleteFP_(winList);
    winList = SmoothWindowWithId_(winList);
    preList_ = winList;
    if (trackPeriod_-- <= 0)
        trackPeriod_ = period_;
    return TransWindow_(img, imgPad, winList);
}

int PCN::GetTrackingPeriod()
{
    return trackPeriod_;
}

// Static functions
cv::Mat PCN::CropFace(cv::Mat img, Window face, int cropSize)
{
    float x1 = face.x;
    float y1 = face.y;
    float x2 = face.width + face.x - 1;
    float y2 = face.width + face.y - 1;
    float centerX = (x1 + x2) / 2;
    float centerY = (y1 + y2) / 2;
    cv::Point2f srcTriangle[3];
    cv::Point2f dstTriangle[3];
    srcTriangle[0] = PCN::RotatePoint(x1, y1, centerX, centerY, face.angle);
    srcTriangle[1] = PCN::RotatePoint(x1, y2, centerX, centerY, face.angle);
    srcTriangle[2] = PCN::RotatePoint(x2, y2, centerX, centerY, face.angle);
    dstTriangle[0] = cv::Point(0, 0);
    dstTriangle[1] = cv::Point(0, cropSize - 1);
    dstTriangle[2] = cv::Point(cropSize - 1, cropSize - 1);
    cv::Mat rotMat = cv::getAffineTransform(srcTriangle, dstTriangle);
    cv::Mat ret;
    cv::warpAffine(img, ret, rotMat, cv::Size(cropSize, cropSize));
    return ret;
}

void PCN::DrawPoints(cv::Mat img, Window face)
{
    int width = 2;
    for (int i = 1; i <= 8; i++)
        cv::line(img, face.points14[i - 1], face.points14[i], BLUE, width);

    for (int i = 0; i < kFeaturePoints; i++)
    {
        if (i <= 8)
            cv::circle(img, face.points14[i], width, CYAN, -1);
        else if (i <= 9)
            cv::circle(img, face.points14[i], width, GREEN, -1);
        else if (i <= 11)
            cv::circle(img, face.points14[i], width, PURPLE, -1);
        else
            cv::circle(img, face.points14[i], width, RED, -1);
    }
    
}

cv::Point PCN::RotatePoint(float x, float y, float centerX, float centerY, float angle)
{
    x -= centerX;
    y -= centerY;
    float theta = -angle * M_PI / 180;
    float rx = centerX + x * std::cos(theta) - y * std::sin(theta);
    float ry = centerY + x * std::sin(theta) + y * std::cos(theta);
    return cv::Point(rx, ry);
}

void PCN::DrawLine(cv::Mat img, std::vector<cv::Point> pointList)
{
    int width = 2;
    cv::line(img, pointList[0], pointList[1], CYAN, width);
    cv::line(img, pointList[1], pointList[2], CYAN, width);
    cv::line(img, pointList[2], pointList[3], CYAN, width);
    cv::line(img, pointList[3], pointList[0], BLUE, width);
}

void PCN::DrawFace(cv::Mat img, Window face)
{
    float x1 = face.x;
    float y1 = face.y;
    float x2 = face.width + face.x - 1;
    float y2 = face.width + face.y - 1;
    float centerX = (x1 + x2) / 2;
    float centerY = (y1 + y2) / 2;
    std::vector<cv::Point> pointList;
    pointList.push_back(PCN::RotatePoint(x1, y1, centerX, centerY, face.angle));
    pointList.push_back(PCN::RotatePoint(x1, y2, centerX, centerY, face.angle));
    pointList.push_back(PCN::RotatePoint(x2, y2, centerX, centerY, face.angle));
    pointList.push_back(PCN::RotatePoint(x2, y1, centerX, centerY, face.angle));
    PCN::DrawLine(img, pointList);
    cv::putText(img, std::string("id:") + std::to_string(face.id),
		    cv::Point(x1, y1), 2, 1, cv::Scalar(255, 0, 0));
}



void PCN::LoadModel_(std::string modelDetect, std::string net1, std::string net2, std::string net3,
                     std::string modelTrack, std::string netTrack,
		     std::string modelEmbed, std::string netEmbed)
{
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    google::InitGoogleLogging("VR");
    FLAGS_logtostderr = 0;

    net_[NET_STAGE1].reset(new caffe::Net<float>(net1.c_str(), caffe::TEST));
    net_[NET_STAGE1]->CopyTrainedLayersFrom(modelDetect.c_str());
    net_[NET_STAGE2].reset(new caffe::Net<float>(net2.c_str(), caffe::TEST));
    net_[NET_STAGE2]->CopyTrainedLayersFrom(modelDetect.c_str());
    net_[NET_STAGE3].reset(new caffe::Net<float>(net3.c_str(), caffe::TEST));
    net_[NET_STAGE3]->CopyTrainedLayersFrom(modelDetect.c_str());
    net_[NET_TRACK].reset(new caffe::Net<float>(netTrack.c_str(), caffe::TEST));
    net_[NET_TRACK]->CopyTrainedLayersFrom(modelTrack.c_str());
    net_[NET_EMBED].reset(new caffe::Net<float>(netEmbed.c_str(), caffe::TEST));
    net_[NET_EMBED]->CopyTrainedLayersFrom(modelEmbed.c_str());

    google::ShutdownGoogleLogging();
}

cv::Mat PCN::PreWhiten_(cv::Mat img){
    cv::Scalar mean3,stddev3; 
    cv::Mat imgF,img_rgb;
    cv::cvtColor(img,img_rgb, cv::COLOR_BGR2RGB);//must be rgb for embeding
    img_rgb.convertTo(imgF, CV_32FC3);//must be float
    cv::meanStdDev(imgF,mean3,stddev3);
    float m1 = (mean3[0] + mean3[1] + mean3[2])/3;
    float v2 = (stddev3[0]*stddev3[0]+ stddev3[1]*stddev3[1] +stddev3[2]*stddev3[2]+
		    mean3[0]*mean3[0] + mean3[1]*mean3[1] + mean3[2]*mean3[2])/3;
    float one_std = 1.0/(sqrt(v2 - m1*m1) + 1e-4);
    cv::Scalar mean3_corr(m1,m1,m1);
    cv::Mat prewhitenedIImg = (imgF - mean3_corr)*one_std;
    return prewhitenedIImg;
}

cv::Mat PCN::PreProcessImg_(cv::Mat img)
{
    cv::Mat mean(img.size(), CV_32FC3, mean_);
    cv::Mat imgF;
    img.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}

cv::Mat PCN::PreProcessImg_(cv::Mat img, int dim)
{
    cv::Mat imgNew;
    cv::resize(img, imgNew, cv::Size(dim, dim));
    cv::Mat mean(imgNew.size(), CV_32FC3, mean_);
    cv::Mat imgF;
    imgNew.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}

void PCN::SetInput_(cv::Mat input, caffe::shared_ptr<caffe::Net<float> > &net)
{
    int rows = input.rows, cols = input.cols;
    int length = rows * cols;
    caffe::Blob<float>* inputBlobs = net->input_blobs()[0];
    inputBlobs->Reshape(1, 3, rows, cols);
    net->Reshape();
    std::vector<cv::Mat> tmp;
    cv::split(input, tmp);
    float *p = inputBlobs->mutable_cpu_data();
    for (int i = 0; i < tmp.size(); i++)
    {
        memcpy(p, tmp[i].data, sizeof(float) * length);
        p += length;
    }
}

void PCN::DumpImage_(std::vector<cv::Mat> &input, db::DB *db, float image_label)
{
    std::vector<cv::Mat> tmp;
    int rows = input[0].rows; 
    int cols = input[0].cols;
    int length = rows * cols;
    float * p = NULL;
    float * copy_pointer = NULL;
    int number_of_channels = 3;
    int total_input_size = length * number_of_channels * input.size() * sizeof(*p);

    for (int i = 0; i < input.size(); i++)
    {
        LMDB__add_to_database(db, input[i], image_label);
    }
}

void PCN::SetInput_(std::vector<cv::Mat> &input, caffe::shared_ptr<caffe::Net<float> > &net)
{
    int rows = input[0].rows, cols = input[0].cols;
    int length = rows * cols;
    caffe::Blob<float>* inputBlobs = net->input_blobs()[0];
    inputBlobs->Reshape(input.size(), 3, rows, cols);
    net->Reshape();
    float *p = inputBlobs->mutable_cpu_data();
    std::vector<cv::Mat> tmp;
    for (int i = 0; i < input.size(); i++)
    {
        cv::split(input[i], tmp);
        for (int j = 0; j < tmp.size(); j++)
        {
            memcpy(p, tmp[j].data, sizeof(float) * length);
            p += length;
        }
    }
}

cv::Mat PCN::ResizeImg_(cv::Mat img, float scale)
{
    cv::Mat ret;
    cv::resize(img, ret, cv::Size(int(img.cols / scale), int(img.rows / scale)));
    return ret;
}

bool PCN::CompareWin_(const Window &w1, const Window &w2)
{
    return w1.conf > w2.conf;
}

bool PCN::Legal_(int x, int y, cv::Mat img)
{
    if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
        return true;
    else
        return false;
}

bool PCN::Inside_(int x, int y, Window rect)
{
    if (x >= rect.x && y >= rect.y && x < rect.x + rect.width && y < rect.y + rect.height)
        return true;
    else
        return false;
}

int PCN::SmoothAngle_(int a, int b)
{
    if (a > b)
        std::swap(a, b);
    int diff = (b - a) % 360;
    if (diff < 180)
        return a + diff / 2;
    else
        return b + (360 - diff) / 2;
}

float PCN::IoU_(Window &w1, Window &w2)
{
    float xOverlap = std::max(0, std::min(w1.x + w1.width - 1, w2.x + w2.width - 1) - std::max(w1.x, w2.x) + 1);
    float yOverlap = std::max(0, std::min(w1.y + w1.height - 1, w2.y + w2.height - 1) - std::max(w1.y, w2.y) + 1);
    float intersection = xOverlap * yOverlap;
    float unio = w1.width * w1.height + w2.width * w2.height - intersection;
    return float(intersection) / unio;
}

std::vector<Window> PCN::NMS_(std::vector<Window> &winList, bool local, float threshold)
{
    if (winList.size() == 0)
        return winList;
    std::sort(winList.begin(), winList.end(), CompareWin_);
    bool flag[winList.size()];
    memset(flag, 0, winList.size());
    for (int i = 0; i < winList.size(); i++)
    {
        if (flag[i])
            continue;
        for (int j = i + 1; j < winList.size(); j++)
        {
            if (local && abs(winList[i].scale - winList[j].scale) > EPS)
                continue;
            if (IoU_(winList[i], winList[j]) > threshold)
                flag[j] = 1;
        }
    }
    std::vector<Window> ret;
    for (int i = 0; i < winList.size(); i++)
    {
        if (!flag[i]) ret.push_back(winList[i]);
    }
    return ret;
}

/// to delete some false positives
std::vector<Window> PCN::DeleteFP_(std::vector<Window> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::sort(winList.begin(), winList.end(), CompareWin_);
    bool flag[winList.size()];
    memset(flag, 0, winList.size());
    for (int i = 0; i < winList.size(); i++)
    {
        if (flag[i])
            continue;
        for (int j = i + 1; j < winList.size(); j++)
        {
            if (Inside_(winList[j].x, winList[j].y, winList[i]) && Inside_(winList[j].x + winList[j].width - 1, winList[j].y + winList[j].height - 1, winList[i]))
                flag[j] = 1;
        }
    }
    std::vector<Window> ret;
    for (int i = 0; i < winList.size(); i++)
    {
        if (!flag[i]) ret.push_back(winList[i]);
    }
    return ret;
}

/// to detect faces on the boundary
cv::Mat PCN::PadImg_(cv::Mat img)
{
    int row = std::min(int(img.rows * 0.2), 100);
    int col = std::min(int(img.cols * 0.2), 100);
    cv::Mat ret;
    cv::copyMakeBorder(img, ret, row, row, col, col, cv::BORDER_CONSTANT, mean_);
    return ret;
}

std::vector<Window> PCN::Stage1_(cv::Mat img, cv::Mat imgPad, caffe::shared_ptr<caffe::Net<float> > &net, float thres)
{
    int row = (imgPad.rows - img.rows) / 2;
    int col = (imgPad.cols - img.cols) / 2;
    std::vector<Window> winList;
    int netSize = NET_STAGE1_WIN_SIZE;
    float curScale;
    curScale = minFace_ / float(netSize);
    cv::Mat imgResized = ResizeImg_(img, curScale);
    while (std::min(imgResized.rows, imgResized.cols) >= netSize)
    {
        SetInput_(PreProcessImg_(imgResized), net);
        net->Forward();
        caffe::Blob<float>* reg = net->output_blobs()[0];
        caffe::Blob<float>* prob = net->output_blobs()[1];
        caffe::Blob<float>* rotateProb = net->output_blobs()[2];
        float w = netSize * curScale;
        for (int i = 0; i < prob->height(); i++)
        {
            for (int j = 0; j < prob->width(); j++)
            {
                if (prob->data_at(0, 1, i, j) > thres)
                {
                    float sn = reg->data_at(0, 0, i, j);
                    float xn = reg->data_at(0, 1, i, j);
                    float yn = reg->data_at(0, 2, i, j);
                    int rx = j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w + col;
                    int ry = i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w + row;
                    int rw = w * sn;
                    if (Legal_(rx, ry, imgPad) && Legal_(rx + rw - 1, ry + rw - 1, imgPad))
                    {
                        if (rotateProb->data_at(0, 1, i, j) > 0.5)
                            winList.push_back(Window(rx, ry, rw, rw, 0, curScale, prob->data_at(0, 1, i, j)));
                        else
                            winList.push_back(Window(rx, ry, rw, rw, 180, curScale, prob->data_at(0, 1, i, j)));
                    }
                }
            }
        }
        imgResized = ResizeImg_(imgResized, scale_);
        curScale = float(img.rows) / imgResized.rows;
    }
    return winList;
}

std::vector<Window> PCN::Stage2_(cv::Mat img, cv::Mat img180, caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::vector<cv::Mat> dataList;
    int height = img.rows;
    for (int i = 0; i < winList.size(); i++)
    {
        if (abs(winList[i].angle) < EPS)
            dataList.push_back(PreProcessImg_(img(cv::Rect(winList[i].x, winList[i].y, winList[i].width, winList[i].height)), dim));
        else
        {
            int y2 = winList[i].y + winList[i].height - 1;
            dataList.push_back(PreProcessImg_(img180(cv::Rect(winList[i].x, height - 1 - y2, winList[i].width, winList[i].height)), dim));
        }
    }
    SetInput_(dataList, net);
    net->Forward();
    caffe::Blob<float>* reg = net->output_blobs()[0];
    caffe::Blob<float>* prob = net->output_blobs()[1];
    caffe::Blob<float>* rotateProb = net->output_blobs()[2];
    std::vector<Window> ret;
    for (int i = 0; i < winList.size(); i++)
    {
        if (prob->data_at(i, 1, 0, 0) > thres)
        {
            float sn = reg->data_at(i, 0, 0, 0);
            float xn = reg->data_at(i, 1, 0, 0);
            float yn = reg->data_at(i, 2, 0, 0);
            int cropX = winList[i].x;
            int cropY = winList[i].y;
            int cropW = winList[i].width;
            if (abs(winList[i].angle)  > EPS)
                cropY = height - 1 - (cropY + cropW - 1);
            int w = sn * cropW;
            int x = cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW;
            int y = cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW;
            float maxRotateScore = 0;
            int maxRotateIndex = 0;
            for (int j = 0; j < 3; j++)
            {
                if (rotateProb->data_at(i, j, 0, 0) > maxRotateScore)
                {
                    maxRotateScore = rotateProb->data_at(i, j, 0, 0);
                    maxRotateIndex = j;
                }
            }
            if (Legal_(x, y, img) && Legal_(x + w - 1, y + w - 1, img))
            {
                float angle = 0;
                if (abs(winList[i].angle)  < EPS)
                {
                    if (maxRotateIndex == 0)
                        angle = 90;
                    else if (maxRotateIndex == 1)
                        angle = 0;
                    else
                        angle = -90;
                    ret.push_back(Window(x, y, w, w, angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
                }
                else
                {
                    if (maxRotateIndex == 0)
                        angle = 90;
                    else if (maxRotateIndex == 1)
                        angle = 180;
                    else
                        angle = -90;
                    ret.push_back(Window(x, height - 1 -  (y + w - 1), w, w, angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
                }
            }
        }
    }
    return ret;
}

float PCN::OnlyStage3_(std::vector<cv::Mat> dataList, caffe::shared_ptr<caffe::Net<float> > &net)
{
    SetInput_(dataList, net);
    net->Forward();
    caffe::Blob<float>* prob = net->output_blobs()[1];

    return prob->data_at(0, 1, 0, 0);
}

void PCN::PartialStage3_(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90, caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window> &winList, db::DB *db, float image_label)
{
    if (winList.size() == 0)
        return;
    std::vector<cv::Mat> dataList;
    int height = img.rows;
    int width = img.cols;

    winList = NMS_(winList, false, nmsThreshold_[NET_STAGE3]);
    winList = DeleteFP_(winList);

    for (int i = 0; i < winList.size(); i++)
    {
        if (abs(winList[i].angle) < EPS)
            dataList.push_back(PreProcessImg_(img(cv::Rect(winList[i].x, winList[i].y, winList[i].width, winList[i].height)), dim));
        else if (abs(winList[i].angle - 90) < EPS)
        {
            dataList.push_back(PreProcessImg_(img90(cv::Rect(winList[i].y, winList[i].x, winList[i].height, winList[i].width)), dim));
        }
        else if (abs(winList[i].angle + 90) < EPS)
        {
            int x = winList[i].y;
            int y = width - 1 - (winList[i].x + winList[i].width - 1);
            dataList.push_back(PreProcessImg_(imgNeg90(cv::Rect(x, y, winList[i].width, winList[i].height)), dim));
        }
        else
        {
            int y2 = winList[i].y + winList[i].height - 1;
            dataList.push_back(PreProcessImg_(img180(cv::Rect(winList[i].x, height - 1 - y2, winList[i].width, winList[i].height)), dim));
        }
    }

    DumpImage_(dataList, db, image_label);
}

std::vector<Window> PCN::Stage3_(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90, caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::vector<cv::Mat> dataList;
    int height = img.rows;
    int width = img.cols;
    for (int i = 0; i < winList.size(); i++)
    {
        if (abs(winList[i].angle) < EPS)
            dataList.push_back(PreProcessImg_(img(cv::Rect(winList[i].x, winList[i].y, winList[i].width, winList[i].height)), dim));
        else if (abs(winList[i].angle - 90) < EPS)
        {
            dataList.push_back(PreProcessImg_(img90(cv::Rect(winList[i].y, winList[i].x, winList[i].height, winList[i].width)), dim));
        }
        else if (abs(winList[i].angle + 90) < EPS)
        {
            int x = winList[i].y;
            int y = width - 1 - (winList[i].x + winList[i].width - 1);
            dataList.push_back(PreProcessImg_(imgNeg90(cv::Rect(x, y, winList[i].width, winList[i].height)), dim));
        }
        else
        {
            int y2 = winList[i].y + winList[i].height - 1;
            dataList.push_back(PreProcessImg_(img180(cv::Rect(winList[i].x, height - 1 - y2, winList[i].width, winList[i].height)), dim));
        }
    }
    SetInput_(dataList, net);
    net->Forward();
    caffe::Blob<float>* reg = net->output_blobs()[0];
    caffe::Blob<float>* prob = net->output_blobs()[1];
    caffe::Blob<float>* rotateProb = net->output_blobs()[2];
    std::vector<Window> ret;
    for (int i = 0; i < winList.size(); i++)
    {
        if (prob->data_at(i, 1, 0, 0) > thres)
        {
            float sn = reg->data_at(i, 0, 0, 0);
            float xn = reg->data_at(i, 1, 0, 0);
            float yn = reg->data_at(i, 2, 0, 0);
            int cropX = winList[i].x;
            int cropY = winList[i].y;
            int cropW = winList[i].width;
            cv::Mat imgTmp = img;
            if (abs(winList[i].angle - 180)  < EPS)
            {
                cropY = height - 1 - (cropY + cropW - 1);
                imgTmp = img180;
            }
            else if (abs(winList[i].angle - 90)  < EPS)
            {
                std::swap(cropX, cropY);
                imgTmp = img90;
            }
            else if (abs(winList[i].angle + 90)  < EPS)
            {
                cropX = winList[i].y;
                cropY = width - 1 - (winList[i].x + winList[i].width - 1);
                imgTmp = imgNeg90;
            }

            int w = sn * cropW;
            int x = cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW;
            int y = cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW;
            float angle = angleRange_ * rotateProb->data_at(i, 0, 0, 0);
	    //At stage 3 we add the global ID
            if (Legal_(x, y, imgTmp) && Legal_(x + w - 1, y + w - 1, imgTmp))
            {
                if (abs(winList[i].angle)  < EPS)
                    ret.push_back(Window(x, y, w, w, angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
                else if (abs(winList[i].angle - 180)  < EPS)
                {
                    ret.push_back(Window(x, height - 1 -  (y + w - 1), w, w, 180 - angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
                }
                else if (abs(winList[i].angle - 90)  < EPS)
                {
                    ret.push_back(Window(y, x, w, w, 90 - angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
                }
                else
                {
                    ret.push_back(Window(width - y - w, x, w, w, -90 + angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
                }
            }
        }
    }
    return ret;
}

std::vector<Window> PCN::TransWindow_(cv::Mat img, cv::Mat imgPad, std::vector<Window> &winList)
{
    int row = (imgPad.rows - img.rows) / 2;
    int col = (imgPad.cols - img.cols) / 2;

    for(int i = 0; i < winList.size(); i++)
    {
        if (winList[i].width > 0 && winList[i].height > 0)
        {
            for (int j = 0; j < kFeaturePoints; j++)
            {
                winList[i].points14[j].x -= col;
                winList[i].points14[j].y -= row;
            }
	    winList[i].x -= col;
	    winList[i].y -= row;
        }
    }
    return winList;
}

#define kMinIoU_Tracking 0.1
std::vector<Window> PCN::SmoothWindowWithId_(std::vector<Window> winList)
{
    //static std::vector<Window> preList_;
    for (int i = 0; i < winList.size(); i++)
    {
        int jmax = -1;//Hold max IOU index window
        float max_iou = 0;
	
        //Find max IOU	
        for (int j = 0; j < preList_.size(); j++)
        {
            float iou = IoU_(winList[i], preList_[j]);
            if (iou > max_iou){
                jmax = j;
                max_iou = iou;
            }
        }

        if (max_iou > kMinIoU_Tracking) 
        {
            winList[i].conf = (winList[i].conf + preList_[jmax].conf) / 2;
            winList[i].x = (max_iou*winList[i].x + (1-max_iou)*preList_[jmax].x);
            winList[i].y = (max_iou*winList[i].y + (1-max_iou)*preList_[jmax].y);
            winList[i].width = (max_iou*winList[i].width + (1-max_iou)*preList_[jmax].width);
            winList[i].height = (max_iou*winList[i].height + (1-max_iou)*preList_[jmax].height);
            winList[i].angle = SmoothAngle_(winList[i].angle, preList_[jmax].angle);

            if (winList[i].id < 0) // in case this window just detected
                winList[i].set_points(preList_[jmax].points14);
            else
                for (int k = 0; k < kFeaturePoints; k++) 
                {
                    winList[i].points14[k].x = (max_iou * winList[i].points14[k].x + (1-max_iou) * preList_[jmax].points14[k].x);
                    winList[i].points14[k].y = (max_iou * winList[i].points14[k].y + (1-max_iou) * preList_[jmax].points14[k].y);
                }

            winList[i].id = preList_[jmax].id; 

        }
        else
        {
            //trackPeriod_ = 0;
            winList[i].id = global_id_++; 
        }
    }

    //if ((preList_.size() > winList.size()) | 
    //    	    (winList.size()==0))
    //    trackPeriod_ = 0;

    preList_ = winList;
    return winList;
}

std::vector<Window> PCN::Embed_(cv::Mat img,caffe::shared_ptr<caffe::Net<float> > &net, std::vector<Window> &winList,int dim)
{
    std::vector<cv::Mat> dataList;
    if (winList.size()==0)
	    return winList;
    for (int i = 0; i < winList.size(); i++)
        dataList.push_back(PreWhiten_(PCN::CropFace(img, winList[i], dim)));

    SetInput_(dataList, net);
    net->Forward();
    caffe::Blob<float>* desc = net->output_blobs()[0];
    float *p = desc->mutable_cpu_data();
    std::vector<Window> tmpWinList;
    for (int i = 0; i < winList.size(); i++)
    {
        Window win(winList[i].x, winList[i].y,
                   winList[i].width, winList[i].height, 
		   winList[i].angle, winList[i].scale, 
		   winList[i].conf, winList[i].id, 
		   winList[i].points14, winList[i].descriptor);
        tmpWinList.push_back(win);
    }

    for (int i = 0; i < tmpWinList.size(); i++){
	memcpy(tmpWinList[i].descriptor,p,sizeof(float)*kDescriptorLen);
        p += kDescriptorLen;
    }
    return tmpWinList;
}

std::vector<Window> PCN::Detect_(cv::Mat img, cv::Mat imgPad)
{
    cv::Mat img180, img90, imgNeg90;
    cv::flip(imgPad, img180, 0);
    cv::transpose(imgPad, img90);
    cv::flip(img90, imgNeg90, 0);

    std::vector<Window> winList = Stage1_(img, imgPad, net_[NET_STAGE1], 
		    classThreshold_[NET_STAGE1]);
    winList = NMS_(winList, true, nmsThreshold_[NET_STAGE1]);

    winList = Stage2_(imgPad, img180, net_[NET_STAGE2], 
		    classThreshold_[NET_STAGE2], NET_STAGE2_WIN_SIZE, winList);
    winList = NMS_(winList, true, nmsThreshold_[NET_STAGE2]);

    winList = Stage3_(imgPad, img180, img90, imgNeg90, net_[NET_STAGE3], 
		    classThreshold_[NET_STAGE3], NET_STAGE3_WIN_SIZE, winList);
    winList = NMS_(winList, false, nmsThreshold_[NET_STAGE3]);
    winList = DeleteFP_(winList);
    if (doEmbed_)
	    winList = Embed_(imgPad,net_[NET_EMBED],winList,NET_EMBED_WIN_SIZE);
    return winList;
}

float PCN::ProcessSingleImageStage3(std::vector<cv::Mat> image)
{
    return OnlyStage3_(image, net_[NET_STAGE3]);
}

std::vector<Window> PCN::GenerateThirdLayerInput(cv::Mat img, db::DB *db, float image_label)
{
    cv::Mat img180, img90, imgNeg90;
    cv::Mat imgPad = PadImg_(img);

    cv::flip(imgPad, img180, 0);
    cv::transpose(imgPad, img90);
    cv::flip(img90, imgNeg90, 0);

    std::vector<Window> winList = Stage1_(img, imgPad, net_[NET_STAGE1], 
		    classThreshold_[NET_STAGE1]);
    winList = NMS_(winList, true, nmsThreshold_[NET_STAGE1]);

    winList = Stage2_(imgPad, img180, net_[NET_STAGE2], 
		    classThreshold_[NET_STAGE2], NET_STAGE2_WIN_SIZE, winList);
    winList = NMS_(winList, true, nmsThreshold_[NET_STAGE2]);

    PartialStage3_(imgPad, img180, img90, imgNeg90, net_[NET_STAGE3], 
		    classThreshold_[NET_STAGE3], NET_STAGE3_WIN_SIZE, winList, db, image_label);


    return winList;
}

std::vector<Window> PCN::Track_(cv::Mat img, caffe::shared_ptr<caffe::Net<float> > &net,
                                 float thres, int dim, std::vector<Window> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::vector<Window> tmpWinList;
    for (int i = 0; i < winList.size(); i++)
    {
        Window win(winList[i].x - augScale_ * winList[i].width,
                   winList[i].y - augScale_ * winList[i].width,
                   winList[i].width + 2 * augScale_ * winList[i].width, 
                   winList[i].height + 2 * augScale_ * winList[i].height, 
		   winList[i].angle, winList[i].scale, 
		   winList[i].conf, winList[i].id, winList[i].points14,
		   winList[i].descriptor);
        tmpWinList.push_back(win);
    }
    std::vector<cv::Mat> dataList;
    for (int i = 0; i < tmpWinList.size(); i++)
    {
        dataList.push_back(PreProcessImg_(PCN::CropFace(img, tmpWinList[i], dim), dim));
    }
    
    SetInput_(dataList, net);
    net->Forward();
    caffe::Blob<float>* reg = net->output_blobs()[0];
    caffe::Blob<float>* prob = net->output_blobs()[1];
    caffe::Blob<float>* pointsReg = net->output_blobs()[2];
    caffe::Blob<float>* rotateProb = net->output_blobs()[3];
    std::vector<Window> ret;
    for (int i = 0; i < tmpWinList.size(); i++)
    {
        if (prob->data_at(i, 1, 0, 0) > thres)
        {
            float cropX = tmpWinList[i].x;
            float cropY = tmpWinList[i].y;
            float cropW = tmpWinList[i].width;
            float centerX = (2 * tmpWinList[i].x + tmpWinList[i].width - 1) / 2;
            float centerY = (2 * tmpWinList[i].y + tmpWinList[i].width - 1) / 2;
            std::vector<cv::Point> points14;
            for (int j = 0; j < (pointsReg->shape(1)) >> 1; j++)
            {
                points14.push_back(PCN::RotatePoint((pointsReg->data_at(i, 2 * j, 0, 0) + 0.5) * (cropW - 1) + cropX,
                                               (pointsReg->data_at(i, 2 * j + 1, 0, 0) + 0.5) * (cropW - 1) + cropY,
                                               centerX, centerY, tmpWinList[i].angle));
            }
            float sn = reg->data_at(i, 0, 0, 0);
            float xn = reg->data_at(i, 1, 0, 0);
            float yn = reg->data_at(i, 2, 0, 0);
            float theta = -tmpWinList[i].angle * M_PI / 180;
            int w = sn * cropW;
            int x = cropX  - 0.5 * sn * cropW +
                    cropW * sn * xn * std::cos(theta) - cropW * sn * yn * std::sin(theta) + 0.5 * cropW;
            int y = cropY  - 0.5 * sn * cropW +
                    cropW * sn * xn * std::sin(theta) + cropW * sn * yn * std::cos(theta) + 0.5 * cropW;
            float angle = angleRange_ * rotateProb->data_at(i, 0, 0, 0);
            if (thres > 0)
            {
                if (Legal_(x, y, img) && Legal_(x + w - 1, y + w - 1, img))
                {
                    int tmpW = w / (1 + 2 * augScale_);
                    if (tmpW >= 20)
                    {
                        ret.push_back(Window(x + augScale_ * tmpW,
                                              y + augScale_ * tmpW,
                                              tmpW, tmpW, winList[i].angle + angle, winList[i].scale, prob->data_at(i, 1, 0, 0),winList[i].id,&points14[0], winList[i].descriptor));
                    }
                }
            }
            else
            {
                int tmpW = w / (1 + 2 * augScale_);
                ret.push_back(Window(x + augScale_ * tmpW,
                                      y + augScale_ * tmpW,
                                      tmpW, tmpW, winList[i].angle + angle, winList[i].scale, prob->data_at(i, 1, 0, 0),winList[i].id,&points14[0], winList[i].descriptor));
            }
        }
    }
    return ret;
}
