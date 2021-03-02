from sklearn import svm
import pickle
from random import choice, sample,choices
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Correlation():
    def __init__(self):
        self.th = 0.5
        pass
    def predict_proba(self,x):
        x = np.array(x)
        vec_len = x.shape[1]
        left,right = np.array_split(x,2,axis=1)
        proba = []
        for l,r in zip(left,right):
            proba.append(np.corrcoef(l,r)[0,1])
        proba = np.array(proba)
        proba = np.vstack((1-proba,proba))
        return proba.T

    def predict(self,x):
        proba = self.predict_proba(x)[:,1]
        y = np.zeros(len(proba))
        y[proba > self.th] = 1
        return y
    def fit(self,x,y):
        pass
    def score(self,x,y):
        pass


classifiers = {
        "MLPClassifier":MLPClassifier(alpha=1e-3,tol=1e-3, max_iter=10000,hidden_layer_sizes=(128,128)),
        #"SVC_RBF":SVC(gamma=2, C=1,probability=True),
        #"Correlation":Correlation(),
        #"SVC_RBF_Test":SVC(gamma=3, C=1,probability=True),
        #"KNeighborsClassifier":KNeighborsClassifier(3),
        #"SVC_Linear":SVC(kernel="linear", C=0.025,probability=True),
        #"GaussianProcessClassifier":GaussianProcessClassifier(1.0 * RBF(1.0)),
        #"DecisionTreeClassifier":DecisionTreeClassifier(max_depth=5),
        #"RandomForestClassifier":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #"AdaBoostClassifier":AdaBoostClassifier(),
        #"GaussianNB":GaussianNB(),
        "QuadraticDiscriminantAnalysis":QuadraticDiscriminantAnalysis(),
        }

def generate_set(persons_dict,keys,Nsim,Ndiff):
    sims_dict_train = {k:persons_dict[k] for k in keys if len(persons_dict[k])>1}
    diff_dict_train = {k:persons_dict[k] for k in keys if len(persons_dict[k])>0}

    #Different
    diff = []
    sims = []
    for _ in range(Ndiff):
        keys = sample(list(diff_dict_train.keys()),2)
        diff.append(np.concatenate([choice(diff_dict_train[keys[0]]),choice(diff_dict_train[keys[1]])]))

    from ipdb import set_trace as dbg
    for _ in range(Nsim):
        key = sample(list(sims_dict_train.keys()),1)[0]
        descs = sample(sims_dict_train[key],2)
        sims.append(np.concatenate(descs))
    X = sims+diff
    y = [1]*Nsim + [0]*Ndiff
    return X,y 

with open("persons_dict.pb", 'rb') as f:
    persons_dict =  pickle.load(f)

train_test_ratio = 0.8
train_cut_off = int(len(persons_dict)*train_test_ratio)
train_keys,test_keys = np.split(np.random.permutation(list(persons_dict.keys())),[train_cut_off])

Ntrain = 15000
Ntest = 10000

X_train, y_train = generate_set(persons_dict,train_keys,Ntrain,Ntrain)
X_test, y_test = generate_set(persons_dict,test_keys,Ntest,Ntest)

# impose symmetry
X_train += [np.fft.fftshift(desc) for desc in X_train]
y_train += y_train

X_test2 = [np.fft.fftshift(desc) for desc in X_test]

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    probs = clf.predict_proba(X_test)[:,1]
    preds = clf.predict(X_test)
    plt.figure()
    bins = np.arange(-0.02,1.02,0.01)
    plt.subplot(311)
    h0,_,_ = plt.hist(probs[np.array(y_test)==0],bins, alpha=0.5, label='0')
    h1,_,_ = plt.hist(probs[np.array(y_test)==1],bins, alpha=0.5, label='1')
    plt.title(name)
    plt.subplot(312)
    cum_err = np.sum(h0)-np.cumsum(h0) + np.cumsum(h1)
    min_th = np.argmin(cum_err)
    plt.semilogy(bins[1:],cum_err)
    
    plt.subplot(313)
    probs2 = clf.predict_proba(X_test2)[:,1]
    plt.hist(probs2-probs,bins=100)#,cumulative=True,density=True, histtype='step')

    print("{0},score={1:0.4f},threshold={2:0.3f},optimal_rate={3:0.4f}".format(name,score,bins[min_th+1],1-cum_err[min_th]/len(y_test)))
    with open("./model/trained_{0}_model.clf".format(name), 'wb') as f:
        pickle.dump(clf, f)
plt.show()
