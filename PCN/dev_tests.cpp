#include "PCN.h"

int main()
{
    PCN detector("model/PCN.caffemodel",
                 "model/PCN-1.prototxt", "model/PCN-2.prototxt", "model/PCN-3.prototxt",
                 "model/PCN-Tracking.caffemodel", "model/PCN-Tracking.prototxt",
		 "model/resnetInception-128.caffemodel","model/resnetInception-128.prototxt");
    
    /// detection
    detector.SetMinFaceSize(15);
    detector.SetImagePyramidScaleFactor(1.45);
    detector.SetDetectionThresh(0.5, 0.5, 0.98);
    /// tracking
    detector.SetTrackingPeriod(30);
    detector.SetTrackingThresh(0.98);

    /// embedding
    detector.SetEmbedding(1);

    cv::Mat img =
    cv::imread("../../Recognition/dlib_face_recognition/lfw/Emil_Winebrand/Emil_Winebrand_0001.jpg");
    cv::TickMeter tm;
    tm.reset();
    tm.start();
    std::vector<Window> faces = detector.Detect(img);

    printf("Embedding[0] %f\n",faces[0].descriptor[0]);

    tm.stop();
    std::cout << "Time Cost: "<<
    	  tm.getTimeMilli() << " ms" << std::endl;
    for (int j = 0; j < faces.size(); j++)
    {
        PCN::DrawFace(img, faces[j]);
        PCN::DrawPoints(img, faces[j]);
    }
    cv::imshow("PCN", img);
    //cv::imshow("wpf", wpf);
    cv::waitKey();
    

    cv::destroyAllWindows();

    return 0;
}
