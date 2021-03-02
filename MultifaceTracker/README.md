### PCN Face tracker
This face tracker includes:
0. Supports multiple faces in video
1. Face detection (based on PCN)
2. Face tracking (based on PCN with ID)
3. Smoothing based on IoU
4. Face embedding based on FaceNet with 128 features
5. Feature matching classifier based on scikit MLPClassifier
6. History handling of new / old faces and matching accordingly

### Download and install PCN
(https://github.com/EmilWine/FaceKit.git)
```
cd FaceKit/PCN
make;sudo make install
./setup.py build
./setup.py install --user
```
### Create a dataset of vector embeddings
```
./create_face_embeddings_dict.py
```
### Evaluate classifiers
```
./eval_all_classifiers.py
```
### Dataset
Download: (http://vis-www.cs.umass.edu/lfw/lfw.tgz)
extract to: `./lfw`

### Run tracker
```
./multiface_tracker.py
```

