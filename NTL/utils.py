

from pathlib import Path
import json
import yaml
import pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def read_config_file(dict_or_filelike):
    if isinstance(dict_or_filelike, dict):
        return dict_or_filelike

    path = Path(dict_or_filelike)
    if path.suffix == ".json":
        return json.load(open(path, "r"))
    elif path.suffix in [".yaml", ".yml"]:
        return yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    elif path.suffix in [".pkl", ".pickle"]:
        return pickle.load(open(path, "rb"))

    raise ValueError("Only JSON, YaML and pickle files supported.")


class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()


def compute_pre_recall_f1(target, score):
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary')

    return f1

class EarlyStopper:

    def stop(self, epoch, val_loss, val_auc=None,  test_loss=None, test_auc=None, test_ap=None,test_f1=None, test_score=None,train_loss=None):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        return  self.train_loss, self.val_loss,self.val_auc,self.test_loss,self.test_auc,self.test_ap,self.test_f1, self.test_score,self.best_epoch

class Patience(EarlyStopper):

    '''
    Implement common "patience" technique
    '''

    def __init__(self, patience=10, use_train_loss=True):
        self.local_val_optimum = float("inf")
        self.use_train_loss = use_train_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.train_loss= None
        self.val_loss, self.val_auc, = None, None
        self.test_loss, self.test_auc,self.test_ap,self.test_f1,self.test_score = None, None,None, None,None

    def stop(self, epoch, val_loss, val_auc=None, test_loss=None, test_auc=None, test_ap=None,test_f1=None,test_score=None,train_loss=None):
        if self.use_train_loss:
            if train_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = train_loss
                self.best_epoch = epoch
                self.train_loss= train_loss
                self.val_loss, self.val_auc= val_loss, val_auc
                self.test_loss, self.test_auc, self.test_ap,self.test_f1,self.test_score\
                    = test_loss, test_auc, test_ap,test_f1, test_score
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.train_loss= train_loss
                self.val_loss, self.val_auc = val_loss, val_auc
                self.test_loss, self.test_auc, self.test_ap,self.test_f1,self.test_score\
                    = test_loss, test_auc, test_ap,test_f1, test_score
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
