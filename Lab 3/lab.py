"""
Created on Sep 26 2018
Last edited on July 03 2023
@author: M.Sc. Dennis Wittich,  M.Sc. Hubert Kanyamahanga
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
import torchvision
import pandas as pd
import torch
import time
from torch.utils.data import Dataset
from torchvision.io import read_image
from os.path import join as pjoin
from functools import partial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================= DATA GENERATION ============================================

def generate_gaussian_clusters(clusters, limits=(0, 255, 0, 255)):
    """Draws samples in 2D space according to the given normal distributions and limits.

        Parameters
        ----------
        clusters : list of tuples
            Each cluster is defined by 7 parameters:
            [(class, c_x, c_y, varx, vary, angle (degree), num_samples), ...]
        limits : tuple (optional)
            Contains min/max values for sample values:
            (min_x, max_x, min_x, max_x)

        Returns
        -------
        out : numpy.ndarray of type float32
            N x 3 matrix containing N samples (coordinates and label):
            [[x1, y1, c1], ..., [xN, yN, cN]]

        Notes
        -----
        Samples are drawn randomly (no fixed seed) and clipped to the given limits.
        Classes in result are also floats. Angles given in degree.
    """

    samples = []
    for (cx, cy, varx, vary, angle, c, num) in clusters:
        angle = angle / 180 * np.pi
        ca = np.cos(angle)
        sa = np.sin(angle)
        xr = np.random.randn((num)) * varx
        yr = np.random.randn((num)) * vary
        xs = ca * xr - sa * yr + cx
        ys = sa * xr + ca * yr + cy
        xs = np.minimum(np.maximum(xs, limits[0]), limits[1])
        ys = np.minimum(np.maximum(ys, limits[2]), limits[3])
        for x, y in zip(xs, ys):
            samples.append((x, y, c))

    return np.array(samples, dtype=float)


def get_feature_vectors_by_classes(X, y):
    """Groups samples by their classes.

        Parameters
        ----------
        X : Features as 2D array [num_samples x num_features]
        y : Labels as 1D array [num_samples]

        Returns
        -------
        out : list
            Element i of the list contains the samples
            of class i as 2D array [num_samples_i x num_features]

        Notes
        -----
        Samples are drawn randomly (no fixed seed) and clipped to the given limits.
        Classes in result are also floats. Angles given in rad.
    """

    num_classes = int(np.max(y) + 1)
    num_feature_vectors = len(X)
    feature_vectors_by_classes = []
    for c in range(num_classes):
        feature_vectors_by_classes.append(list())
    for i in range(num_feature_vectors):
        feature_vectors_by_classes[int(y[i])].append(X[i])
    for c in range(num_classes):
        feature_vectors_by_classes[c] = np.array(feature_vectors_by_classes[c])
    return feature_vectors_by_classes

# ========================= MODELS =============================================
def print_summary(Ls, TBAs, VAs, CM):
    print('\nFinal validation accuracy: {:.1%}'.format(VAs[-1]))
    print('\nTEST SET ACCURACY: {:.1%}\n'.format(np.trace(CM)/np.sum(CM)))

    plt.subplot(1,2,1)
    plt.plot(Ls,label='Cross Entropy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot([x*100 for x in TBAs], label='Training (Batch) Accuracy')
    plt.plot([x*100 for x in VAs], label='Validation Accuracy')
    plt.ylim(0, 105)
    plt.legend()
    plt.show()


class MnistGenerator():

    def __init__(self, train_batch_size=32, num_train=59000, num_valid=1000, num_test=10000):
        self.trainset = torchvision.datasets.MNIST(root='./../mnist', train=True, download=True)
        self.testset = torchvision.datasets.MNIST(root='./../mnist', train=False, download=True)
        # print('training samples:', len(self.trainset))
        # print('test samples:', len(self.testset))

        self.num_train = num_train
        self.num_valid = num_valid
        self.num_test = num_test

        assert num_train + num_valid <= len(self.trainset), "Not enough samples for training + validation"
        assert num_test <= len(self.testset), "Not enough samples for testing"

        self.Ti = 0
        self.TBS = train_batch_size

    def get_train_batch(self):
        Xs = np.zeros((self.TBS, 1, 28, 28), dtype=np.float32)
        Ys = np.zeros(self.TBS, dtype=np.int32)
        for i in range(self.TBS):
            Xi, Yi = self.trainset[self.Ti]
            Xs[i, 0] = np.array(Xi)
            Ys[i] = int(Yi)
            self.Ti += 1
            if self.Ti > self.num_train:
                self.Ti = 0
        return Xs, Ys

    def get_validation_batch(self):
        Xs = np.zeros((self.num_valid, 1, 28, 28), dtype=np.float32)
        Ys = np.zeros(self.num_valid, dtype=np.int32)
        for i in range(self.num_valid):
            Xi, Yi = self.trainset[self.num_train + i]
            Xs[i, 0] = np.array(Xi)
            Ys[i] = int(Yi)
            self.Ti += 1
            if self.Ti > self.num_train:
                self.Ti = 0
        return Xs, Ys

    def get_test_batch(self):
        Xs = np.zeros((self.num_test, 1, 28, 28), dtype=np.float32)
        Ys = np.zeros(self.num_test, dtype=np.int32)
        for i in range(self.num_test):
            Xi, Yi = self.testset[i]
            Xs[i, 0] = np.array(Xi)
            Ys[i] = int(Yi)
            self.Ti += 1
            if self.Ti > self.num_train:
                self.Ti = 0
        return Xs, Ys

# ========================= UAV DATA LOADER ====================================

class UaVidDataset(Dataset):
    """
        Implementation by inheritance from ``torch.util.data.Dataset`` 
        - init: Which data should be loaded? Setup attributes
        - len: How many items are in the dataset (defines length of loop)
        - getitem: Returns the item at index ``idx`` (may include preprocessing / augmentation steps)
    """
    def __init__(self, subset, tf = None):
        assert subset in ['train', 'val', 'test'], "Invalid subset"
        self.df = pd.read_pickle(f'./uavid_pkl/{subset}_df.pkl')
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 3]
        ref_path = self.df.iloc[idx, 2]
        image = (read_image(img_path) / 255.0)
        idmap = read_image(ref_path).long()
        sample = {'image':image, 'idmap':idmap}
        if self.tf:
            sample = self.tf(sample)
        return sample

class UaVidDataset_PL(Dataset):
    """
        Implementation by inheritance from ``torch.util.data.Dataset`` 
        - init: Which data should be loaded? Setup attributes
        - len: How many items are in the dataset (defines length of loop)
        - getitem: Returns the item at index ``idx`` (may include preprocessing / augmentation steps)
        - will preload 50 images and it's 3 times faster than UaVidDataset class.
    """
    def __init__(self, subset, num_smpl = 50):
        assert subset in ['train', 'val', 'test'], "Invalid subset"
        self.df = pd.read_pickle(f'./uavid_pkl/{subset}_df.pkl')
        self.num_smpl = num_smpl
        
        self.images = []
        self.refs = []
        print(f'preloading {num_smpl} images')
        
        for idx in range(num_smpl):
            img_path = self.df.iloc[idx, 3]
            ref_path = self.df.iloc[idx, 2]
            self.images.append((read_image(img_path)/255.0).type(torch.float16))
            self.refs.append(read_image(ref_path).type(torch.int8))
            
    def __len__(self):
        return self.num_smpl

    def __getitem__(self, idx):
        image = self.images[idx].float()
        idmap = self.refs[idx].long()
        return {'image': image, 'idmap': idmap}

def idmap2labelmap(idmap):
    """
    Function converts ID-maps to coloured label maps 
    """
    h,w = idmap.shape[:2]
    labelmap = colours[idmap.reshape(-1)].reshape((h,w,3))
    return labelmap

# ==================== SAVE and LOAD PARAMS ======================================

def save_net(net, name):
    save_dict = {'state_dict': net.state_dict()}
    torch.save(save_dict, pjoin('./checkpoints', f'{name}.pt'))

def load_net(net, name):
    load = torch.load if torch.cuda.is_available() else partial(torch.load, map_location='cpu')
    checkpoint = load(pjoin('./checkpoints', f'{name}.pt'))
    state_dict_to_load = checkpoint['state_dict']
    net.load_state_dict(state_dict_to_load)

# ========================= NETWORK EVALUATION ====================================

def eval_net(network, dataloader, metric='mf1'):
    '''The next function is used to evaluate on a subset (uses own 
    auxiliary functions to aggregate confusion matrix)'''
    
    num_cls = 7 
    ign_index = 7
    
    print('\nRunning validation .. ', end='')
    conf_matrix = np.zeros((num_cls, num_cls), int) 
    
    for batch in dataloader:
        image = batch['image'].to(device.type)
        idmap = batch['idmap']
        
        idmap[idmap==ign_index]=-1
        
        with torch.no_grad():                       # bit faster, steps for back propagation are skipped
            logits = network(image)
            preds = torch.argmax(logits, dim=1)
        preds_np = preds.cpu().data.numpy().ravel() # use Tensor.cpu() to convert Temsor to Numpy
        idmap_np = idmap.data.numpy().ravel()

        update_confusion_matrix(conf_matrix, preds_np, idmap_np)
        
    if metric == 'cm':
        return conf_matrix
    else:
        metrics = get_confusion_metrics(conf_matrix)
        return metrics[metric]       


def update_confusion_matrix(confusions, predicted_labels, reference_labels):
    # reference labels with label < 0 will not be considered
    reshaped_pr = np.ravel(predicted_labels)
    reshaped_gt = np.ravel(reference_labels)
    for predicted, actual in zip(reshaped_pr, reshaped_gt):
        if actual >= 0 and predicted >= 0:
            confusions[predicted, actual] += 1

def get_confusion_metrics(confusion_matrix):
    """Computes confusion metrics out of a confusion matrix (N classes)
        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]
        Returns
        -------
        metrics : dict
            a dictionary holding all computed metrics
        Notes
        -----
        Metrics are: 'percentages', 'precisions', 'recalls', 'f1s', 'mf1', 'oa'
    """

    tp = np.diag(confusion_matrix)
    tp_fn = np.sum(confusion_matrix, axis=0)
    tp_fp = np.sum(confusion_matrix, axis=1)

    has_no_rp = tp_fn == 0
    has_no_pp = tp_fp == 0

    tp_fn[has_no_rp] = 1
    tp_fp[has_no_pp] = 1

    percentages = tp_fn / np.sum(confusion_matrix)
    precisions = tp / tp_fp
    recalls = tp / tp_fn

    p_zero = precisions == 0
    precisions[p_zero] = 1

    f1s = 2 * (precisions * recalls) / (precisions + recalls)
    ious = tp / (tp_fn + tp_fp - tp)

    precisions[has_no_pp] *= 0.0
    precisions[p_zero] *= 0.0
    recalls[has_no_rp] *= 0.0

    f1s[p_zero] *= 0.0
    f1s[percentages == 0.0] = np.nan
    ious[percentages == 0.0] = np.nan

    mf1 = np.nanmean(f1s)
    miou = np.nanmean(ious)
    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    metrics = {'percentages': percentages,
               'precisions': precisions,
               'recalls': recalls,
               'f1s': f1s,
               'mf1': mf1,
               'ious': ious,
               'miou': miou,
               'oa': oa}

    return metrics

def print_metrics(confusions):
    metrics = get_confusion_metrics(confusions)

    print('\nclass | pct of data | precision |   recall  |    f1     |    iou',
          '\n-----------------------------------------------------------------')

    percentages = metrics["percentages"]
    precisions = metrics["precisions"]
    recall = metrics["recalls"]
    f1 = metrics["f1s"]
    ious = metrics["ious"]
    mf1 = metrics["mf1"]
    miou = metrics["miou"]
    oa = metrics["oa"]

    for i in range(len(percentages)):
        pct = '{:.3%}'.format(percentages[i]).rjust(9)
        p = '{:.3%}'.format(precisions[i]).rjust(7)
        r = '{:.3%}'.format(recall[i]).rjust(7)
        f = '{:.3%}'.format(f1[i]).rjust(7)
        u = '{:.3%}'.format(ious[i]).rjust(7)
        print('   {:2d} |  {}  |  {}  |  {}  |  {}  |  {}\n'.format(i, pct, p, r, f, u))

    print('mean f1-score: {:.3%}'.format(mf1))
    print('mean iou: {:.3%}'.format(miou))
    print('Overall accuracy: {:.3%}'.format(oa))
    print('Samples: {}'.format(np.sum(confusions)))