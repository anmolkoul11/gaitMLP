import numpy as np
from scipy import stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import welch
from scipy.stats import entropy
import bisect
from torch.optim.lr_scheduler import CyclicLR

# This function loads the signal measusrementa and labels, and splits it into time and values.
def loadTrial_Train(dataFolder,id):
    xt = np.genfromtxt('{}trial{:02d}.x.t.csv'.format(dataFolder,id),delimiter=',')
    xv = np.genfromtxt('{}trial{:02d}.x.v.csv'.format(dataFolder,id),delimiter=',')
    yt = np.genfromtxt('{}trial{:02d}.y.t.csv'.format(dataFolder,id),delimiter=',')
    yv = np.genfromtxt('{}trial{:02d}.y.v.csv'.format(dataFolder,id),delimiter=',')
    yv = yv.astype(int)

    # Returning x measurements and y labels
    return xt, xv, yt, yv

# This function extracts features from the measurements.
def extractFeat(xt,xv,winSz,timeStart,timeEnd,timeStep):
    tList = []
    featList = []

    # Specifying the initial window for extracting features
    t0 = timeStart
    t1 = t0+winSz

    while(t1<=timeEnd):
        # Using the middle time of the window as a reference time
        tList.append((t0+t1)/2)

        # Extracting features
        xWin = xv[(xt>=t0)*(xt<=t1),:]
        f1 = np.mean(xWin,axis=0)
        f2 = np.std(xWin,axis=0)

        mag = np.linalg.norm(xWin, axis=1) #per sample resultant accerelation
        f3  = [np.mean(mag)] #mean resultant acceleration

        fs  = 40
        f, P = welch(mag, fs=fs, nperseg=len(mag))
        P   /= P.sum()                 # normalise
        f4  = [entropy(P)] #spectral entropy

        # Storing the features
        featList.append(np.concatenate((f1,f2,f3,f4)))

        # Updating the window by shifting it by the step size
        t0 = t0+timeStep
        t1 = t0+winSz

    tList = np.array(tList)
    featList = np.array(featList)

    return tList, featList

# ---- NEW: extractSeq (returns tensor sequence + per‑step labels) ----
def extractSeq(xt, xv, yt, yv, winSz, timeStart, timeEnd, timeStep):
    """
    Returns:
        Xseq  (T, 6, 4)   – 10 time‑steps × 6 channels × 4 samples
        yseq  (T,)        – 10 labels (one every 0.1 s)
    """
    stepHz  = 10                     # label rate
    sampHz  = 40                     # sensor rate
    SAMP    = int(winSz * sampHz)    # 40
    STEPS   = int(winSz * stepHz)    # 10
    PATCH   = SAMP // STEPS          # 4

    t0, t1 = timeStart, timeStart + winSz
    Xseqs, yseqs = [], []

    while t1 <= timeEnd:
        # ----- slice raw window -----
        xWin = xv[(xt >= t0) & (xt < t1), :]        # (40, 6)
        xWin = xWin.reshape(STEPS, PATCH, 6)        # (10, 4, 6)
        xWin = xWin.transpose(0, 2, 1)              # (10, 6, 4)

        # ----- align 10 label ticks -----
        centres = np.linspace(t0 + 0.05, t1 - 0.05, STEPS)
        lbl = []
        for ts in centres:
            idx = bisect.bisect_left(yt, ts)
            if idx == 0:
                lbl.append(yv[0])
            elif idx == len(yt):
                lbl.append(yv[-1])
            else:
                lbl.append(yv[idx-1] if ts - yt[idx-1] < yt[idx] - ts else yv[idx])

        Xseqs.append(xWin.astype(np.float32))
        yseqs.append(np.array(lbl, dtype=np.int64))

        t0 += timeStep
        t1 += timeStep

    return np.stack(Xseqs), np.stack(yseqs)


# This function returns the mode over a window of data to make it compatible with the features
# extracted.
def extractLabel(yt,yv,winSz,timeStart,timeEnd,timeStep):
    tList = []
    yList = []

    # Specifying the initial window for extracting features
    t0 = timeStart
    t1 = t0+winSz

    while t1 <= timeEnd:
        t_center = (t0 + t1) / 2          # centre of the window

        # find index of the first label timestamp ≥ centre
        idx = bisect.bisect_left(yt, t_center)

        # choose the closer of idx‑1 or idx
        if idx == 0:
            label = yv[0]
        elif idx == len(yt):
            label = yv[-1]
        else:
            prev_t, next_t = yt[idx-1], yt[idx]
            label = yv[idx-1] if (t_center - prev_t) <= (next_t - t_center) else yv[idx]

        tList.append(t_center)
        yList.append(label)

        t0 += timeStep
        t1 += timeStep

    tList = np.array(tList)
    yList = np.array(yList)

    return tList, yList

# It loads the data and extracts the features
def loadFeatures(dataFolder,winSz,timeStep,idList):
    for k,id in enumerate(idList):
        # Loading the raw data
        xt, xv, yt, yv = loadTrial_Train(dataFolder,id=id)

        # Extracting the time window for which we have values for the measurements and the response
        timeStart = np.max((np.min(xt),np.min(yt)))
        timeEnd = np.min((np.max(xt),np.max(yt)))

        # Extracting the features
        _, feat = extractFeat(xt,xv,winSz,timeStart,timeEnd,timeStep)
        _, lab = extractLabel(yt,yv,winSz,timeStart,timeEnd,timeStep)

        # Storing the features
        if(k==0):
            featList = feat
            labList = lab
        else:
            featList = np.concatenate((featList,feat),axis=0)
            labList = np.concatenate((labList,lab),axis=0)

    return featList, labList

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

# Creating a wrapper so we have the same interface for all the methods. This wrapper takes as an
# input an mlp model so we can reuse it with different model architectures.
class NetWrapper:
    def __init__(self, model, device, epochs, weights):
        self.model = model.to(device)
        self.device, self.epochs = device, epochs

        alpha = torch.tensor(weights, device=device)
        self.loss_fn = FocalLoss(gamma=2, alpha=alpha)

        self.opt = torch.optim.AdamW(self.model.parameters(),
                                     lr=3e-3, weight_decay=1e-4)
        self.sched = CyclicLR(self.opt,
                      base_lr=1e-4, max_lr=3e-3,
                      step_size_up=200, cycle_momentum=False)

    def fit(self, X, y, X_val, y_val):
        ds = TensorDataset(torch.from_numpy(X).float(),
                           torch.from_numpy(y).long())
        
        labels_for_sampling = y if y.ndim == 1 else y[:, 0]

        weight = 1./torch.bincount(torch.tensor(labels_for_sampling))
        sampler = WeightedRandomSampler(weight[labels_for_sampling], num_samples=len(labels_for_sampling), replacement=True)
        loader = DataLoader(ds, batch_size=128, sampler=sampler)

        best_bacc, patience, best_state = 0., 15, None
        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)             # logits (B, 10, 4)
                loss   = self.loss_fn(logits.view(-1, 4),
                      yb.view(-1))
                self.opt.zero_grad(); loss.backward(); self.opt.step()

            # ---- validation ----
            self.model.eval()
            with torch.no_grad():
                y_hat = self.predict(X_val)
            bacc = metrics.balanced_accuracy_score(y_val, y_hat)
            self.sched.step(1-bacc)

            if bacc > best_bacc:
                best_bacc, patience = bacc, 15
                best_state = self.model.state_dict()
            else:
                patience -= 1
                if patience == 0:
                    break

        self.model.load_state_dict(best_state)

    def predict(self, X):
        xb = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model(xb)
            predictions = logits.argmax(dim=-1)
        return predictions.cpu().numpy()

# ---------------------------------------------------------------------
# NEW helper: loads *sequence* tensors instead of flat feature vectors
# ---------------------------------------------------------------------
def loadSeq(dataFolder, winSz, timeStep, idList):
    for k, tid in enumerate(idList):
        xt, xv, yt, yv = loadTrial_Train(dataFolder, id=tid)

        timeStart = max(xt.min(), yt.min())
        timeEnd   = min(xt.max(), yt.max())

        Xseq, yseq = extractSeq(xt, xv, yt, yv,
                                winSz, timeStart, timeEnd, timeStep)

        if k == 0:
            X_all, y_all = Xseq, yseq
        else:
            X_all = np.concatenate((X_all, Xseq), axis=0)
            y_all = np.concatenate((y_all, yseq), axis=0)

    return X_all, y_all        # shapes: (N, 10, 6, 4)  and  (N, 10)


# This function produces a summary of performance metrics including a confusion matrix
def summaryPerf(yTrain,yTrainHat,y,yHat):
    # Plotting confusion matrix for the non-training set:
    cm = metrics.confusion_matrix(y,yHat,normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=
                                  ['Walk Hard','Down Stairs','Up Stairs','Walk Soft'])
    disp.plot()

    # Displaying metrics for training and non-training sets
    print('Training:  Acc = {:4.3f}'.format(metrics.accuracy_score(yTrain,yTrainHat)))
    print('Training:  BalAcc = {:4.3f}'.format(metrics.balanced_accuracy_score(yTrain,yTrainHat)))
    print('Validation: Acc = {:4.3f}'.format(metrics.accuracy_score(y,yHat)))
    print('Validation: BalAcc = {:4.3f}'.format(metrics.balanced_accuracy_score(y,yHat)))