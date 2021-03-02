from multiprocessing import Manager, Pool
import pickle
from PyPCN import *
import os
import glob

def normalize_desc(desc):
    desc = np.array(desc)
    desc /= np.linalg.norm(desc) #performane improves with normalization
    return desc.tolist() #save lists to allow json serialization

def assign_files_to_persons(root_path):
    assign_dict = {}
    for p in os.listdir(root_path):
        assign_dict[p] = []
        for fl in glob.glob(root_path + p+'/*.jpg'):
            assign_dict[p].append(fl)
    return assign_dict

def embed(elem):
    idp,(person,pix) = elem
    print(idp,person)
    shared_dict[person] = manager.list() #must be shared list
    for person_img in pix:
        try:
            img = cv2.imread(person_img)
        except:
            continue

        if img is None:
            continue
        faces = detector.Detect(img)
        if len(faces) != 1:
            continue
        shared_dict[person].append(normalize_desc(faces[0].descriptor))

if __name__=="__main__":
    detection_model_path = "./model/PCN.caffemodel"
    pcn1_proto = "./model/PCN-1.prototxt"
    pcn2_proto = "./model/PCN-2.prototxt"
    pcn3_proto = "./model/PCN-3.prototxt"
    tracking_model_path = "./model/PCN-Tracking.caffemodel"
    tracking_proto = "./model/PCN-Tracking.prototxt"
    embed_model_path = "./model/resnetInception-128.caffemodel"
    embed_proto = "./model/resnetInception-128.prototxt"

    detector = PCN(detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
			tracking_model_path,tracking_proto, 
                        embed_model_path, embed_proto,
			15,1.45,0.5,0.5,0.98,30,0.9,1)

    manager = Manager()
    shared_dict = manager.dict()


    ### serial version
    #persons_dict= assign_files_to_persons("./lfw/")  
    #for elem in enumerate(persons_dict.items()):
    #    embed(elem)
   

    #parallel version
    pool = Pool (processes=7)
    persons_dict= assign_files_to_persons("./EFI/")  
    pool.map(embed, enumerate(persons_dict.items()))

    persons_dict= assign_files_to_persons("./lfw/")  
    pool.map(embed, enumerate(persons_dict.items()))

    pool.close()

    persons_dict = {k:list(v) for k,v in shared_dict.items()}

    with open('persons_dict.pb', 'wb') as fd:
        pickle.dump(dict(persons_dict),fd)
