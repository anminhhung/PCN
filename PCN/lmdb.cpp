#include <opencv2/opencv.hpp>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/format.hpp"

using std::string;
namespace db = caffe::db;

static int db_index = 0;

db::DB *LMDB__init_db(string path, string db_type)
{
	db::DB *db(db::GetDB(db_type));
	db->Open(path + db_type, db::NEW);

    return db;
}

void LMDB__fini_db(db::DB *db)
{
    db->Close();
}

void LMDB__add_to_database(db::DB *db, cv::Mat cv_img, float label)
{
    caffe::Datum datum;
	db::Transaction *txn(db->NewTransaction());

    datum.set_channels(cv_img.channels());
    datum.set_height(cv_img.rows);
    datum.set_width(cv_img.cols);

    int datum_channels = datum.channels();
    int datum_height = datum.height();
    int datum_width = datum.width();
    int datum_size = datum_channels * datum_height * datum_width;
    float p[datum_size] = {0, };

    //std::string buffer(datum_size, ' ');

    for (int h = 0; h < datum_height; ++h) {
        const float * ptr = cv_img.ptr<float>(h);
        int img_index = 0;
        for (int w = 0; w < datum_width; ++w) {
            for (int c = 0; c < datum_channels; ++c) {
                int datum_index = (c * datum_height + h) * datum_width + w;
                // datum.add_float_data(static_cast<float>(ptr[img_index]));
                p[datum_index] = static_cast<float>(ptr[img_index]);
                img_index++;
            }
        }
    }

    for (int i = 0; i < datum_size; i++) {
        datum.add_float_data(p[i]);
    }

    datum.set_label(label);
    //datum.set_data(buffer, datum_size);

    string out;
    datum.SerializeToString(&out);

    txn->Put(caffe::format_int(db_index, 5), out);
    db_index++;

	txn->Commit();
}
