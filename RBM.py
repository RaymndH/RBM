from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
from scipy.special import expit 

def get_en(v):
    return .5*(np.dot(v.T,np.dot(wmat,v)))

def Is_en(v):
    w=2*v-1
    return .5*(np.dot(w,np.dot(wmat,w)))

def RBM_en(v,h):
    return np.dot(v.T,np.dot(W,h))

def gibbs(w):
    p=np.dot(W,w) 
    
    expit(p, out=p)
    return (np.random.uniform(size=w.shape) < p)


# Load your dataset
fname = "wishart_planting_N_16_alpha_1.12/wishart_planting_N_16_alpha_1.12_inst_1.txt"
digits = np.loadtxt(fname)
N=16
wmat=np.zeros((16,16))
for line in digits:
    wmat[int(line[0]), int(line[1])] = line[2]
wmat=wmat+wmat.T 

W = -wmat

dotones1 = np.dot(W,np.ones(N))



alpha = 2

C = -alpha * np.abs(dotones1)
#C = -.5*np.ones(N)

for i in range(N):
    W[i,i]=-C[i]

dotones2=np.dot(W,np.ones(N))

b = -1/2*dotones2


# Create an RBM model

if(1):
    v=np.random.choice([0,1],[16])
    h=np.random.choice([0,1],[16])
    #v=.5+np.array([-1., -1., -1.,  1.,  1.,  1., -1.,  1., -1., -1., -1., -1., -1., -1.,  1.,  1.])/2

else:
    v=np.random.choice([-1,1],[16,1])
    v=np.ones_like(v)
    h=v
#h=vnp

print("initial state of v is", v)
print("initial state of h is", h)
print()
min_found=100
for i in range(30):
   
    #print("energy is", RBM_en(v,h))
    #print("other energy is", get_en(v))
    print("ising energy is", Is_en(v))
    if(Is_en(v)<min_found):
        min_found=Is_en(v)
    #forward gibbs sample
    h=gibbs(v)
    #print(h.T.astype(int))
    v=gibbs(h)
    print(v.T.astype(int))
    #backwards gibbs sample
    print()

print("min",min_found)
