import numpy as np
from sklearn.metrics import balanced_accuracy_score

def test_1():
    yVal = np.genfromtxt('Data/val.y.csv',delimiter=',')
    yPred = np.genfromtxt('Data/2LayerPred.y.csv',delimiter=',')

    assert balanced_accuracy_score(yVal,yPred) > 0.82, 'Your balanced accuracy is not high enough'
    
if __name__ == "__main__":
    test_1()
