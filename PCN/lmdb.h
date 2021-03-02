#ifndef __LMDB_H__
#define __LMDB_H__

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using caffe::Datum;
using std::string;
namespace db = caffe::db;

/* Functions */
db::DB *LMDB__init_db(string path, string db_type);

void LMDB__fini_db(db::DB *db);

void LMDB__add_to_database(db::DB *db, cv::Mat cv_img, float label);

#endif /* __LMDB_H__*/
