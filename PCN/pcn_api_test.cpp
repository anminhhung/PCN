#include "PCN_API.h"
#define CROPPED_FACE 150
int main(int argc, char **argv)
{
	const char *detection_model_path = "./model/PCN.caffemodel";
	const char *pcn1_proto = "./model/PCN-1.prototxt";
	const char *pcn2_proto = "./model/PCN-2.prototxt";
	const char *pcn3_proto = "./model/PCN-3.prototxt";
	const char *tracking_model_path = "./model/PCN-Tracking.caffemodel";
	const char *tracking_proto = "./model/PCN-Tracking.prototxt";
	const char *embed_model = "./model/resnetInception-128.caffemodel";
	const char *embed_proto = "model/resnetInception-128.prototxt";
	PCN* detector = (PCN*) init_detector(detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
			tracking_model_path,tracking_proto,embed_model, embed_proto,
			40,1.45,0.5,0.5,0.98,30,0.9,1);
	
	cv::VideoCapture capture;
	if (argc >1)
		capture.open(argv[1]);
	else
		capture.open(0);

	if (!capture.isOpened())
		return 0;

	cv::Mat img;
	cv::Mat crpImg(CROPPED_FACE, CROPPED_FACE, CV_8UC3, cv::Scalar(0,0, 0));
	cv::TickMeter tm;
	while (1)
	{
		bool ret = capture.read(img);
		if (!ret)
			break;
		tm.reset();
		tm.start();

		Window* wins = NULL;
		int lwin = -1;
		wins = detect_and_track_faces(detector,img.data,img.rows, img.cols,&lwin);
		tm.stop();
		int fps = 1000.0 / tm.getTimeMilli();
		std::stringstream ss;
		ss << std::setw(4) << fps;
		cv::putText(img, std::string("PCN:") + ss.str() + "FPS",
				cv::Point(20, 45), 4, 1, cv::Scalar(0, 0, 125));

		if (lwin>0){
			get_aligned_face(img.data,img.rows,img.cols,
					&wins[0], crpImg.data,CROPPED_FACE);
			cv::imshow("Crop", crpImg);
		}

		for (int i = 0; i < lwin; i++){
			cv::putText(img, std::string("id:") + std::to_string(wins[i].id),
					cv::Point(wins[i].x,wins[i].y ), 1, 1, cv::Scalar(255, 0, 0));
			for (int p=0; p < kFeaturePoints; p++){
				cv::Point pt(wins[i].points14[p].x,wins[i].points14[p].y);
				cv::circle(img, pt, 2, RED, -1);
			}
		}

		cv::imshow("PCN", img);
		if (cv::waitKey(1) == 'q')
			break;
	}

	capture.release();
	cv::destroyAllWindows();
	free_detector(detector);
	return 0;
}
