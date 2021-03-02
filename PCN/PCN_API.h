#ifndef __PCN_API__
#define __PCN_API__
#include "PCN.h"

#define kFeaturePoints 14
//TODO: prioritize faces between stages if more than X faces
#define kMaxFaces 64

struct API_PCN{
	PCN* detector;
	Window wins[kMaxFaces];			
};
typedef struct API_PCN API_PCN;

extern "C"
{
	void *init_detector(const char *detection_model_path, 
			const char *pcn1_proto, const char *pcn2_proto, const char *pcn3_proto, 
			const char *tracking_model_path, const char *tracking_proto,
			const char *embed_model_path, const char *embed_proto,
			int min_face_size, float pyramid_scale_factor, float detection_thresh_stage1,
			float detection_thresh_stage2, float detection_thresh_stage3, 
			int tracking_period, float tracking_thresh, int do_embedding)
	{
		API_PCN* api_pcn = (API_PCN*)malloc(sizeof(API_PCN));

		api_pcn->detector = new PCN(detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
				tracking_model_path,tracking_proto,
				embed_model_path,embed_proto);

		/// detection
		api_pcn->detector->SetMinFaceSize(min_face_size);
		api_pcn->detector->SetImagePyramidScaleFactor(pyramid_scale_factor);
		api_pcn->detector->SetDetectionThresh(
				detection_thresh_stage1,
				detection_thresh_stage2,
				detection_thresh_stage3);
		/// tracking
		api_pcn->detector->SetTrackingPeriod(tracking_period);
		api_pcn->detector->SetTrackingThresh(tracking_thresh);

		/// embedding
		api_pcn->detector->SetEmbedding(do_embedding);	

		return static_cast<void*> (api_pcn);
	}
	
	int get_track_period(void* pcn){
		API_PCN* api_pcn = (API_PCN*) pcn;
		return  api_pcn->detector->GetTrackingPeriod();
	}

	float process_single_image(void *pcn, std::vector<cv::Mat> dataList)
	{
		API_PCN *api_pcn = (API_PCN *) pcn;
		return api_pcn->detector->ProcessSingleImageStage3(dataList);
	}

	void generate_third_detect_layer_input(void* pcn, unsigned char* raw_img, size_t rows, size_t cols, int *lwin, db::DB *db, float label)
	{
		API_PCN* api_pcn = (API_PCN*) pcn;
		cv::Mat img(rows, cols, CV_8UC3, (void *)raw_img);
		std::vector<Window> faces = api_pcn->detector->GenerateThirdLayerInput(img, db, label);

		*lwin = faces.size();
		if (*lwin > kMaxFaces)
			*lwin = kMaxFaces;
		memcpy(api_pcn->wins,&faces[0],*lwin * sizeof(Window));
	}

	Window* detect_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, int *lwin)
	{
		API_PCN* api_pcn = (API_PCN*) pcn;
		cv::Mat img(rows,cols, CV_8UC3, (void*)raw_img);
		std::vector<Window> faces = api_pcn->detector->Detect(img);

		*lwin = faces.size();
		if (*lwin > kMaxFaces)
			*lwin = kMaxFaces;
		memcpy(api_pcn->wins,&faces[0],*lwin * sizeof(Window));
		return api_pcn->wins;
	}

	Window* detect_and_track_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, int *lwin)
	{
		API_PCN* api_pcn = (API_PCN*) pcn;
		cv::Mat img(rows,cols, CV_8UC3, (void*)raw_img);
		std::vector<Window> faces = api_pcn->detector->DetectTrack(img);

		*lwin = faces.size();
		if (*lwin > kMaxFaces)
			*lwin = kMaxFaces;

		memcpy(api_pcn->wins,&faces[0],*lwin * sizeof(Window));
		return api_pcn->wins;
	}

	void get_aligned_face(unsigned char* input_image, size_t rows, size_t cols, 
			Window* face, unsigned char* output_image, size_t cropSize)
	{
		cv::Mat in_img(rows,cols, CV_8UC3, (void*)input_image);
		cv::Mat out_img(cropSize,cropSize, CV_8UC3, (void*)output_image);
		cv::Mat temp_img = PCN::CropFace(in_img, *face, cropSize);
		temp_img.copyTo(out_img);
	}


	void free_detector(void *pcn)
	{
		API_PCN* api_pcn = (API_PCN*) pcn;
		delete api_pcn->detector;
		free(api_pcn);
	}
}



#endif

