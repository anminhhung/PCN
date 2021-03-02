#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "PCN_API.h"
#include "lmdb.h"
#include <dirent.h>
#include <stdlib.h>
#include <regex>
#include <sys/stat.h>
#include "caffe/util/db.hpp"

namespace db = caffe::db;

using namespace cv;

#define CROPPED_FACE (150)
#define IMAGE_PATH_TEMPLATE "[0-9]*-([0-9]*).jpg"
#define NORMALIZE_FACTOR (90)
#define YAW_LABEL_INIT (-5)
#define detection_model_path "./model/PCN.caffemodel"
#define pcn1_proto "./model/PCN-1.prototxt"
#define pcn2_proto "./model/PCN-2.prototxt"
#define pcn3_proto "./model/PCN-3.prototxt"
#define tracking_model_path "./model/PCN-Tracking.caffemodel"
#define tracking_proto "./model/PCN-Tracking.prototxt"
#define embed_model "./model/resnetInception-128.caffemodel"
#define embed_proto "model/resnetInception-128.prototxt"

typedef enum {
    ARGUMENT__PROGRAM_NAME = 0,
    ARGUMENT__IMAGES_DIRECTORY,
    ARGUMENT__TRAIN_DIRECTORY,
    ARGUMENT__TEST_DIRECTORY,
    ARGUMENT__RATIO_TRAIN_TEST,
    ARGUMENT__NUMBER_OF_ARGUMENTS,
} arguments_t;

typedef enum {
    FILE_TYPE__DIRECTORY = 0,
    FILE_TYPE__FILE,
    FILE_TYPE__NUMBER_OF_ARGUMENTS,
} file_type_t;

std::map<std::string, int> index_to_label_g = {
    { "01", 1 },
    { "02", 2 },
    { "03", 3 },
    { "04", 4 },
    { "05", 5 },
    { "06", 0 },
    { "07", 7 },
    { "08", 8 },
    { "09", 9 },
    { "10", 10 },
    { "11", 0 },
    { "12", 0 },
    { "13", 0 },
    { "14", 0 },
};

std::map<std::string, float> index_to_degrees_g = {
    { "01", -90 },
    { "02", -72 },
    { "03", -54 },
    { "04", -36 },
    { "05", -18 },
    { "06", 0 },
    { "07", 22.5 },
    { "08", 45 },
    { "09", 67.5 },
    { "10", 90 },
    { "11", 0 },
    { "12", 0 },
    { "13", 0 },
    { "14", 0 },
};

float NormalizeDegrees(float degrees)
{
    return degrees / NORMALIZE_FACTOR;
}

std::string SplitFilename (std::string& str)
{
       std::size_t found = str.find_last_of("/\\");
       return str.substr(found+1);
}

std::string GetRawFaceOrientationText(std::string image_path)
{
    std::regex re(IMAGE_PATH_TEMPLATE);
    std::smatch match;
    std::string basename = SplitFilename(image_path);
    bool rc = false;

    rc = std::regex_search(basename, match, re);
    if (rc && match.size() > 1)
    {
        return match.str(1);
    }
    else 
    {
        return std::string("");
    }
}

int GetIntLabelFromImagePath(std::string image_path)
{
    std::string orientation_text = GetRawFaceOrientationText(image_path);
    return index_to_label_g[orientation_text];
}

float GetLabelFromImagePath(std::string image_path)
{
    std::string orientation_text = GetRawFaceOrientationText(image_path);
    float degrees = index_to_degrees_g[orientation_text];
    return NormalizeDegrees(degrees);

}

/* Returns a list of files in a directory (except the ones that begin with a dot) */
void GetFilesInDirectory(std::vector<std::string> &out, const std::string &directory, file_type_t file_type)
{
    DIR *dir;
    class dirent *ent;
    class stat st;

    dir = opendir(directory.c_str());
    while ((ent = readdir(dir)) != NULL) {
        const std::string file_name = ent->d_name;
        const std::string full_file_name = directory + "/" + file_name;

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory && FILE_TYPE__DIRECTORY == file_type)
            out.push_back(full_file_name.c_str());
        if (!is_directory && FILE_TYPE__FILE == file_type)
            out.push_back(full_file_name.c_str());
    }
    closedir(dir);
}

cv::Mat DatumToCVMat(Datum *datum)
{
    int datum_channels = datum->channels();
    int datum_height = datum->height();
    int datum_width = datum->width();

    string strData = datum->data();
    cv::Mat cv_img;

    if (strData.size() != 0)
    {
        cv_img.create(datum_height, datum_width, CV_8UC(datum_channels));
        const string& data = datum->data();
        std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());

        for (int h = 0; h < datum_height; ++h) {
            uchar* ptr = cv_img.ptr<uchar>(h);
            int img_index = 0;
            for (int w = 0; w < datum_width; ++w) {
                for (int c = 0; c < datum_channels; ++c) {
                    int datum_index = (c * datum_height + h) * datum_width + w;
                    ptr[img_index++] = static_cast<uchar>(vec_data[datum_index]);
                }
            }
        }
    }

    else
    {
        cv_img.create(datum_height, datum_width, CV_32FC(datum_channels));
        for (int h = 0; h < datum_height; ++h) {
            float* ptr = cv_img.ptr<float>(h);
            int img_index = 0;
            for (int w = 0; w < datum_width; ++w) {
                for (int c = 0; c < datum_channels; ++c) {
                    int datum_index = (c * datum_height + h) * datum_width + w;
                    ptr[img_index++] = static_cast<float>(datum->float_data(datum_index));
                }
            }
        }
    }

    return cv_img;
}

void test_images(PCN *detector, std::string train_db_directory)
{
    db::DB *db(db::GetDB("lmdb"));
    db->Open(train_db_directory + "lmdb", db::READ);
    db::Cursor *cursor(db->NewCursor());
    int train_index = 0;
    caffe::Datum datum;
    cv::Mat image;
    std::vector<cv::Mat> dataList(1);
    float result = -1;

    dataList.clear();

    while(cursor->valid())
    {
		Window* wins = NULL;
        cv::Mat image;
		int lwin = -1;

        datum.ParseFromString(cursor->value());
        image = DatumToCVMat(&datum);
        dataList.push_back(image);

        /* Call Stage3 */
        result = process_single_image(detector, dataList);
        if (result < 0.8)
        {
            printf("%d: %f\n", train_index, result);
            cv::imshow("", image);
            cv::waitKey(0);
        }

        dataList.clear();

        train_index++;
        cursor->Next();
    }

l_Cleanup:
    db->Close();
}

int main(int argc, char **argv)
{
    char * images_directory = NULL;
    char * train_directory = NULL;
    char * test_directory = NULL;
    float test_train_ratio = 0;
    int number_of_train_images = 0;
    int number_of_test_images = 0;
    int train_index = 0;
	std::vector<std::string> directories;
	std::vector<std::string> out;
    db::DB *db;
    db::DB *train_db;
    db::DB *test_db;

    if (ARGUMENT__NUMBER_OF_ARGUMENTS != argc) {
        /* not enough parameters */
		printf("not enough parameters\n");
		printf("Usage: %s <images_db_path> <train_dir_path> <test_dir_path> <train_test_ratio>\n", argv[ARGUMENT__PROGRAM_NAME]);
        return 1;
    }

    images_directory = argv[ARGUMENT__IMAGES_DIRECTORY];
    train_directory = argv[ARGUMENT__TRAIN_DIRECTORY];
    test_directory = argv[ARGUMENT__TEST_DIRECTORY];
    test_train_ratio = atof(argv[ARGUMENT__RATIO_TRAIN_TEST]);

	PCN* detector = (PCN*) init_detector(detection_model_path, pcn1_proto, pcn2_proto, pcn3_proto,
			tracking_model_path, tracking_proto, embed_model, embed_proto,
			40, 1.45, 0.5, 0.5, 0.98, 30, 0.9, 1);

    if (0 == test_train_ratio)
    {
        // test_images(detector, train_directory);
        test_images(detector, train_directory);
        goto l_Exit;
    }
	
	GetFilesInDirectory(directories, images_directory, FILE_TYPE__DIRECTORY);
    for (std::vector<std::string>::iterator i = directories.begin();
            i != directories.end(); i++) {
        GetFilesInDirectory(out, i->c_str(), FILE_TYPE__FILE);
    }

    number_of_train_images = (int)(out.size() * test_train_ratio);
    number_of_test_images = out.size() - number_of_train_images;


    train_db = LMDB__init_db(train_directory, "lmdb");
    test_db = LMDB__init_db(test_directory, "lmdb");

    db = train_db;
    for (std::vector<std::string>::iterator i = out.begin(); i != out.end(); i++)
	{
		Window* wins = NULL;
        cv::Mat image;
		int lwin = -1;
        const char * image_path = i->c_str();
        float yaw_label = YAW_LABEL_INIT;
        int yaw_label_int = YAW_LABEL_INIT;

        yaw_label = GetLabelFromImagePath(*i);
        //yaw_label_int = GetIntLabelFromImagePath(*i);
        
        image = imread(image_path);
        if (image.empty())
        {
            std::cout << "!!! Failed imread(): image not found" << std::endl;
            goto l_Cleanup;
        }

        if (train_index == number_of_train_images)
        {
            db = test_db;
        }
        
		generate_third_detect_layer_input(detector, image.data, image.rows, image.cols, &lwin, db, yaw_label);

        train_index++;
	}

l_Cleanup:
    LMDB__fini_db(test_db);
    LMDB__fini_db(train_db);
    /*
    */
l_Exit:
	cv::destroyAllWindows();
	free_detector(detector);
	return 0;
}
