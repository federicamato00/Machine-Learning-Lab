

import sklearn.datasets
import numpy
import scipy.optimize


def polynomial_kernel_with_bias(x1, x2,xi,ci):
     d=2
     return ((numpy.dot(x1.T, x2) + ci) ** d) + xi

def rbf_kernel_with_bias(x1, x2,xi, gamma):
     return numpy.exp(-gamma * numpy.sum((x1 - x2) ** 2)) + xi


def compute_kernel_score(alpha, DTR, L, kernel_func, x,xi,ci):
     Z = numpy.zeros(L.shape)
     Z[L == 1] = 1
     Z[L == 0] = -1
     score = 0
     for i in range(alpha.shape[0]):
         if alpha[i] > 0:
             score += alpha[i]*Z[i]* kernel_func(DTR[:, i],x,xi,ci)
     return score

def mcol(v):
    return v.reshape((v.size,1))
def mRow(v):
    return v.reshape((1,v.size))

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def compute_lagrangian_wrapper(H):
    def compute_lagrangian(alpha):
        alpha = alpha.reshape(-1, 1)
        Ld_alpha = 0.5 * alpha.T @ H @ alpha - numpy.sum(alpha)
        gradient = H @ alpha - 1
        return Ld_alpha.item(), gradient.flatten()
    return compute_lagrangian

def accuracy(predicted_labels, original_labels):
    total_samples = len(predicted_labels)
    correct = (predicted_labels == original_labels).sum()
    return (correct / total_samples) * 100

def error_rate(predicted_labels, original_labels):
    return 100 - accuracy(predicted_labels, original_labels)

def compute_H(DTR,LTR,kernel_func,xi,ci):
     n_samples = DTR.shape[1]
     Hc = numpy.zeros((n_samples, n_samples))
     Z = numpy.where(LTR == 0, -1, 1)
     for i in range(n_samples):
         for j in range(n_samples):
             Hc[i, j] = Z[i]*Z[j]* kernel_func(DTR[:, i], DTR[:, j],xi,ci)
     return Hc


if __name__=='__main__':
     
     D, L = load_iris_binary()
     (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
     for kernel_func in [polynomial_kernel_with_bias,rbf_kernel_with_bias]:
        if kernel_func==polynomial_kernel_with_bias:
            print("KERNEL FUNCTION: POLYNOMIAL")
            print("\t")
            value=[0,1]
        else:
            print("KERNEL FUNCTION RBF")
            print("\t")
            value=[1.0,10.0]
        for ci in value:
            for K in [0,1.0]:
                xi=K*K
                Hc = compute_H(DTR, LTR, kernel_func,xi,ci)
                compute_lag=compute_lagrangian_wrapper(Hc)
                bound_list=[(0,1.0)]*LTR.size
                (alpha,f,d)=scipy.optimize.fmin_l_bfgs_b(compute_lag,x0=numpy.zeros(LTR.size),approx_grad=False,factr=1.0,bounds=bound_list)
                score=numpy.array([compute_kernel_score(alpha,DTR,LTR,kernel_func,x,xi,ci) for x in DTE.T])
                predicted_labels = numpy.where(score > 0, 1, 0)
                # Create predicted_labels array based on scores
                predicted_labels = numpy.where(score > 0, 1, 0)
                error = (predicted_labels != LTE).mean()
                print("K:",K)
                if kernel_func==polynomial_kernel_with_bias:
                    print("C: 1.0, polynomial with d: 2 and c: ", ci)
                else:
                    print("C: 1.0, rbf with gamma: ", ci)
                print("dual loss: ", -f)
                print(f'Error rate: {error * 100:.1f}%')
                print("\t")
             

             

     
     