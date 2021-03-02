#!/usr/bin/python3
from ctypes import *
import cv2
import numpy as np
import sys
import os
from enum import IntEnum
import scipy.spatial.qhull ##whitchcraft

class Point(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int)]

FEAT_POINTS = 14
DESCRIPTORS = 128
class Window(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("width", c_int),
                ("height", c_int),
                ("angle", c_float),
                ("yaw", c_float),
                ("scale", c_float),
                ("conf", c_float),
                ("id", c_long),
                ("points",Point*FEAT_POINTS),
                ("descriptor",c_float*DESCRIPTORS)]

class FeatEnam(IntEnum):
    CHIN_0 = 0
    CHIN_1 = 1
    CHIN_2 = 2
    CHIN_3 = 3
    CHIN_4 = 4
    CHIN_5 = 5
    CHIN_6 = 6
    CHIN_7 = 7
    CHIN_8 = 8
    NOSE = 9
    EYE_LEFT = 10
    EYE_RIGHT = 11
    MOUTH_LEFT = 12
    MOUTH_RIGHT = 13
    FEAT_POINTS = 14

lib = CDLL("libPCN.so", RTLD_GLOBAL)

#	void *init_detector(const char *detection_model_path, 
#			const char *pcn1_proto, const char *pcn2_proto, const char *pcn3_proto, 
#			const char *tracking_model_path, const char *tracking_proto,
#			const char *embed_model_path, const char *embed_proto,
#			int min_face_size, float pyramid_scale_factor, float detection_thresh_stage1,
#			float detection_thresh_stage2, float detection_thresh_stage3, 
#			int tracking_period, float tracking_thresh, int do_embedding)
init_detector = lib.init_detector
init_detector.argtypes = [
        c_char_p, c_char_p, c_char_p, 
        c_char_p, c_char_p, c_char_p,
        c_char_p, c_char_p,
        c_int,c_float,c_float,c_float,
        c_float,c_int,c_float,c_int]
init_detector.restype = c_void_p

#int get_track_period(void* pcn)
get_track_period = lib.get_track_period
get_track_period.argtypes = [c_void_p]
get_track_period.restype = c_int

#Window* detect_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, int *lwin)
detect_faces = lib.detect_faces
detect_faces.argtypes = [c_void_p, POINTER(c_ubyte),c_size_t,c_size_t,POINTER(c_int)]
detect_faces.restype = POINTER(Window)

#Window* detect_and_track_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, int *lwin)
detect_and_track_faces = lib.detect_and_track_faces
detect_and_track_faces.argtypes = [c_void_p, POINTER(c_ubyte),c_size_t,c_size_t,POINTER(c_int)]
detect_and_track_faces.restype = POINTER(Window)

#void get_aligned_face(unsigned char* input_image, size_t rows, size_t cols, 
#                Window* face, unsigned char* output_image, size_t cropSize)
get_aligned_face = lib.get_aligned_face
get_aligned_face.argtypes = [POINTER(c_ubyte),c_size_t,c_size_t,
        POINTER(Window),POINTER(c_ubyte),c_size_t]

# void free_detector(void *pcn)
free_detector = lib.free_detector
free_detector.argtypes= [c_void_p]

class PCN():
    def __init__(self,detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
            tracking_model_path,tracking_proto,embed_model_path,embed_proto,
            min_face=40,pyramid_scale=1.45,th1=0.5,th2=0.5,th3=0.5, 
            track_period = 30,track_th=0.9,do_embed=1):

        self.track_period = track_period
        self.pcn_detector = init_detector(
                PCN.c_str(detection_model_path),PCN.c_str(pcn1_proto),
                PCN.c_str(pcn2_proto),PCN.c_str(pcn3_proto),
                PCN.c_str(tracking_model_path),PCN.c_str(tracking_proto),
                PCN.c_str(embed_model_path),PCN.c_str( embed_proto),
                min_face,pyramid_scale,th1,th2,th3,track_period,track_th,do_embed)
    
    def __del__(self):
        free_detector(self.pcn_detector)

    @staticmethod
    def c_str(str_in):
        return c_char_p(str_in.encode('utf-8'))

    def Detect(self,img):
        face_count = c_int(0)
        raw_data = img.ctypes.data_as(POINTER(c_ubyte))
        pcn_dets = detect_faces(self.pcn_detector, raw_data, 
                img.shape[0], img.shape[1],
                pointer(face_count))
        pcn_faces = [pcn_dets[k] for k in range(face_count.value)]
        return pcn_faces

    def DetectAndTrack(self,img):
        face_count = c_int(0)
        raw_data = img.ctypes.data_as(POINTER(c_ubyte))
        pcn_dets = detect_and_track_faces(self.pcn_detector, raw_data, 
                img.shape[0], img.shape[1],
                pointer(face_count))
        pcn_faces = [pcn_dets[k] for k in range(face_count.value)]
        return pcn_faces

    def CheckTrackingPeriod(self):
        return get_track_period(self.pcn_detector)

    @staticmethod
    def DrawFace(win,img,face_id=None):
        x1 = win.x;
        y1 = win.y;
        x2 = win.width + win.x - 1;
        y2 = win.width + win.y - 1;
        centerX = (x1 + x2) / 2;
        centerY = (y1 + y2) / 2;
        angle = win.angle
        R = cv2.getRotationMatrix2D((centerX,centerY),angle,1)
        pts = np.array([[x1,y1,1],[x1,y2,1],[x2,y2,1],[x2,y1,1]], np.int32)
        pts = (pts @ R.T).astype(int) #Rotate points
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,0,255))
        if face_id is None:
            face_id = str(win.id)
        cv2.putText(img,"{0}:{1}".format(face_id,win.id),(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)

    #@staticmethod
    #def CalculateFaceYaw(win):
    #    vec1 =np.array((
    #        win.points[FeatEnam.EYE_LEFT].x-win.points[FeatEnam.MOUTH_RIGHT].x,
    #        win.points[FeatEnam.EYE_LEFT].y-win.points[FeatEnam.MOUTH_RIGHT].y))
    #    vec2 =np.array(( 
    #        win.points[FeatEnam.EYE_RIGHT].x-win.points[FeatEnam.MOUTH_LEFT].x,
    #        win.points[FeatEnam.EYE_RIGHT].y-win.points[FeatEnam.MOUTH_LEFT].y))
    #    cos_angle = np.dot(vec1,vec2)/np.linalg.norm(vec1)/np.linalg.norm(vec2)
    #    return cos_angle

    @staticmethod
    def DrawPoints(win,img):
        f = FeatEnam.NOSE
        cv2.circle(img,(win.points[f].x,win.points[f].y),2,(255, 153, 255))
        f = FeatEnam.EYE_LEFT
        cv2.circle(img,(win.points[f].x,win.points[f].y),2,(0,255,255))
        f = FeatEnam.EYE_RIGHT
        cv2.circle(img,(win.points[f].x,win.points[f].y),2,(0,255,255))
        f = FeatEnam.MOUTH_LEFT
        cv2.circle(img,(win.points[f].x,win.points[f].y),2,(0,255,0))
        f = FeatEnam.MOUTH_RIGHT
        cv2.circle(img,(win.points[f].x,win.points[f].y),2,(0,255,0))

def SetThreadCount(threads):
    os.environ['OMP_NUM_THREADS'] = str(threads)


if __name__=="__main__":

    SetThreadCount(1)
    if len(sys.argv)==2:
        cap = cv2.VideoCapture(sys.argv[1])
    else:
        cap = cv2.VideoCapture(0)

    detection_model_path = "../model/PCN.caffemodel"
    pcn1_proto = "../model/PCN-1.prototxt"
    pcn2_proto = "../model/PCN-2.prototxt"
    pcn3_proto = "../model/PCN-3.prototxt"
    tracking_model_path = "../model/PCN-Tracking.caffemodel"
    tracking_proto = "../model/PCN-Tracking.prototxt"
    embed_model_path = "../model/resnetInception-128.caffemodel"
    embed_proto = "../model/resnetInception-128.prototxt"

    detector = PCN(detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
			tracking_model_path,tracking_proto, 
                        embed_model_path, embed_proto,
			15,1.45,0.5,0.5,0.98,30,0.9,0)

    while cap.isOpened():
        ret, frame = cap.read()
        if frame.shape[0] == 0:
            break
        windows = detector.DetectAndTrack(frame)
        #if face_count.value > 0:
        #    crpSize = 150
        #    cropped_face = np.zeros((crpSize,crpSize,3),dtype=np.uint8)
        #    raw_crp = cropped_face.ctypes.data_as(POINTER(c_ubyte))
        #    get_aligned_face(raw_data, int(height), int(width),pointer(windows[0]), raw_crp,int(crpSize))
        #    cv2.imshow('crop', cropped_face)

        for win in windows:
            PCN.DrawFace(win,frame)
            PCN.DrawPoints(win,frame)

        cv2.imshow('window', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


