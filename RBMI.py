from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
from scipy.special import expit 

def get_en(v):
    return .5*(np.dot(v.T,np.dot(wmat,v)))

def Is_en(v):
    return .5*(np.dot(v,np.dot(wmat,v)))

def RBM_en(v,h):
    return np.dot(v.T,np.dot(W,h))

def neuron(w):
    p=np.dot(W,w)
    logp=np.sum(np.log(1+np.exp(p)))
    return p, logp

def gibbs(w, max_prob, best_state):
    p,logp=neuron(w)
    
    if(logp>max_prob):
        max_prob=logp
        best_state=w
    #print("logp",logp)
    expit(p, out=p)
    return 2*(np.random.uniform(size=w.shape) < p)-1, best_state, max_prob


# Load your dataset
fnames = [  "wishart_planting_N_16_alpha_0.19/wishart_planting_N_16_alpha_0.19_inst_1.txt",
            "wishart_planting_N_16_alpha_0.50/wishart_planting_N_16_alpha_0.50_inst_1.txt",
            "wishart_planting_N_16_alpha_0.75/wishart_planting_N_16_alpha_0.75_inst_1.txt",
            "wishart_planting_N_16_alpha_0.88/wishart_planting_N_16_alpha_0.88_inst_1.txt",
            "wishart_planting_N_16_alpha_1.12/wishart_planting_N_16_alpha_1.12_inst_1.txt"]
grounds = [ -1.2655691,
            -3.9861914,
            -6.987742,
            -6.2008785,
            -8.9782550]


            

iter = 4

ground = grounds[iter]
fname = fnames[iter]
digits = np.loadtxt(fname)
N=16
wmat=np.zeros((16,16))
for line in digits:
    wmat[int(line[0]), int(line[1])] = line[2]
wmat=wmat+wmat.T 

W = -5*wmat

dotones1 = np.dot(W,np.ones(N))



alpha = 1.1

C = -alpha * np.abs(dotones1)
#C = -.25*np.ones(N)
#C = -.5*np.ones(N)

for i in range(N):
    W[i,i]=-C[i]


# Create an RBM model

plant=np.array([-1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1])
print("plant is \n",plant)
successes=0
N_trials=19
success_states=[]
print("beginning trials:")
for trial in range(N_trials):
    if(1):
        v=np.random.choice([-1,1],[16])
        h=np.random.choice([-1,1],[16])
        #
    else:
        v=np.random.choice([-1,1],[16,1])
        v=np.ones_like(v)
        h=v
    #h=v

    #print("initial state of v is", v)
    #print("initial state of h is", h)
    #print()
    min_found=100
    max_prob=-100
    best_state=np.zeros_like(v)
    bester=np.zeros_like(v)
    for i in range(10000):
    
        #print("energy is", RBM_en(v,h))
        #print("other energy is", get_en(v))
        #print("ising energy is", Is_en(v))
        if(Is_en(v)<min_found):
            min_found=Is_en(v)
            bester=v
            #print(v.T.astype(int))
        #forward gibbs sample
        h,best_state,max_prob=gibbs(v,max_prob, best_state)
        #print(h.T.astype(int))
        v,best_state,max_prob=gibbs(h,max_prob, best_state)
        
        #backwards gibbs sample


    #print("max p state", best_state)
    #print("diff",best_state-plant)
    #print("bestest",bester)
    #print("")
    print("trial",trial,"minimum energy found: ",min_found)
    print("Ising energy of highest probability state : ", Is_en(best_state))
    print()
    if (Is_en(best_state) < ground + 1e-3):
        successes+=1
        success_states.append(best_state)
        
print(successes/N_trials, " success rate")
print()