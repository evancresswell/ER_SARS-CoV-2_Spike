import sys,os
import data_processing as dp	
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
import expectation_reflection as ER
from direct_info import direct_info
from direct_info import sort_di
from joblib import Parallel, delayed
import numpy as np
import pickle
from Bio import Phylo
import networkx, pylab
import pandas as pd
import Bio.SubsMat.FreqTable
from Bio.Align import AlignInfo
from Bio import AlignIO, SeqIO

from scipy.spatial import distance
import random
# Proplerly defined Energy.. 1/24/2023

# Expectation Reflection                                                                                 
#=========================================================================================#
def predict_w(s,i0,i1i2,niter_max,l2):                                                                   
    #print('i0:',i0)                                                                                     
    i1,i2 = i1i2[i0,0],i1i2[i0,1]                                                                        
    x = np.hstack([s[:,:i1],s[:,i2:]])                                                                   
    y = s[:,i1:i2]                                                                                       
    h01,w1 = ER.fit(x,y,niter_max,l2)                                                                    
    return h01,w1                                                                                        



def col_H(i, i1i2, w, b, s):
    for ii, (i1,i2) in enumerate(i1i2):
        if i in range(i1,i2):
            break
    
    H_i = 0
    for j in range(len(s)):
        if j in range(i1,i2):
            continue
        H_i +=  w[i,j] * s[j] 
    H_i += b[i]
    return H_i

def prob_mut1(seq, i1i2, w, b, ncpu=2):
    
    # Caluculate Hi for every column of H, includeing bias.
    resH = Parallel(n_jobs = ncpu)(delayed(col_H)                                                   
        (i0, i1i2, w, b, seq)
        for i0 in range(len(seq))) 
    H_array = resH

    mut_probs = []
    prob_tot = 0.
    for i in range(len(seq)):
        H_i = H_array[i]
        mut_prob =  np.exp( H_i) / (2 * np.cosh(H_i))
        prob_tot += mut_prob
        mut_probs.append(mut_prob)
        
    return prob_tot, np.array(mut_probs)


# Choose single mutation given sequence and w/b.

def w_seq_walk(seq, i1i2, w, b, n_iter=1000,seed=42,ncpu=2):
    seq_walk = [seq]
    n_var = len(i1i2)

    random.seed(seed)
    for itr in range(n_iter):
        
        # Randomly choose seqwuence mutation (weighted with current sequence and w/b)
        prob_gp1, prob_array_gp1 = prob_mut1(seq_walk[-1], i1i2, w, b, ncpu=ncpu)
        mut_pos_aa = np.random.choice(range(len(prob_array_gp1)), p=prob_array_gp1/np.sum(prob_array_gp1))
        # print([(aa,j) for aa,j in enumerate(seq_walk[-1])])
        
        # find position mut_pos_aa is in
        found = False
        for i0 in range(n_var):
            i1,i2 = i1i2[i0][0], i1i2[i0][1]
            for i, ii0 in enumerate(range(i1,i2)):
                if mut_pos_aa == ii0:
                    found = True
                    break
            if found:
                break
        # print('%d (%d, %d) in ' % (mut_pos_aa, i, ii0), i1i2[i0])
        
        # apply mutation
        temp_sequence = np.copy(seq_walk[-1])
        sig_section = np.zeros(i2-i1)
        sig_section[i] = 1.
        temp_sequence[i1:i2] = sig_section
        # print([(aa,j) for aa,j in enumerate(temp_sequence)])
        # print('\n\n')
        seq_walk.append(temp_sequence)
    return seq_walk

#========================================================================================
data_path = '/data/cresswellclayec/covid_data/'
root_dir = '/data/cresswellclayec/covid_data/'




#========================================================================================
data_path = '/data/cresswellclayec/covid_data/spikeprot1201/'
root_dir = '/data/cresswellclayec/covid_data/'

cpus_per_job = 60



c1 = int(sys.argv[1])
c2 = int(sys.argv[2])
cw = int(sys.argv[3])
niter = int(sys.argv[4])
nwalk = int(sys.argv[5])
print('Generating walk with boundary sequence for clusters %d and %d with w/b of cluster %d\n using %d iterations over %d walks' % (c1, c2, cw, niter, nwalk))

print('loading sequences and getting onehot fit and pca')
# sample n sequences from each alignment
n = 10000

variants = ['Omicron', 'Delta']

s01 = np.load('%s_processed_s0.npy' %variants[0])
s_ipdb1 = np.load('%s_processed_ipdb.npy' %variants[0])
s_index1 = np.load('%s_processed_sindex.npy' %variants[0])

s02 = np.load('%s_processed_s0.npy' %variants[1])
s_ipdb2 = np.load('%s_processed_ipdb.npy' %variants[1])
s_index2 = np.load('%s_processed_sindex.npy' %variants[1])

# get all possible samplings from families for consistent grouping 
s01_partial_index = random.sample([indx for indx in range(len(s01))],n)
s02_partial_index = random.sample([indx for indx in range(len(s02))],n)

# combine variant MSAs to encode.
s0_full = np.concatenate((s01[s01_partial_index], s02[s02_partial_index]), axis=0)

# s0_full = s01

onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
print('Omicron s0: ',s0_full.shape,'\n',s0_full)
onehot_encoder.fit(s0_full)
s = onehot_encoder.transform(s0_full)
print('s: ',s.shape,'\n',s)

from sklearn.decomposition import PCA
pca_dim=3

pca = PCA(n_components = pca_dim)
s_pca = pca.fit_transform(s)


# number of positions
n_var = s0_full.shape[1]
n_seq = s0_full.shape[0]

print("Number of residue positions:",n_var)
print("Number of sequences:",n_seq)

# number of aminoacids at each position
mx = np.array([len(np.unique(s0_full[:,i])) for i in range(n_var)])
print("Number of different amino acids at each position",mx)

mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T


# number of variables
mx_sum = mx.sum()
print("Total number of variables",mx_sum)


from joblib import Parallel, delayed                                                                     
import expectation_reflection as ER                                                                      

variant_w = {}
variant_b = {} 

from sklearn.cluster import SpectralClustering
n_cluster = 4 # this is hardcoded into following implementation. to make general will need looping
if os.path.exists("cov_spike_%dcluster_ODvars.pkl" % n_cluster):
    clustering = pickle.load( open("cov_spike_%dcluster_ODvars.pkl" % n_cluster, "rb"))
else:
    clustering = SpectralClustering(n_clusters=n_cluster, random_state=0).fit(s_pca)
    # save 4 clustered omicron and delta.
    pickle.dump(clustering, open("cov_spike_%dcluster_ODvars.pkl" % n_cluster, "wb"))

colors = ['r','b','g','y']
print(np.unique(clustering.labels_, return_counts=True))
print(clustering.labels_)
cluster_colors = [colors[i][0] for i in clustering.labels_]

c1_indx = [indx for i,indx in enumerate(range(len(s_pca))) if clustering.labels_[i] == 0]
c2_indx = [indx for i,indx in enumerate(range(len(s_pca))) if clustering.labels_[i] == 1]
c3_indx = [indx for i,indx in enumerate(range(len(s_pca))) if clustering.labels_[i] == 2]
c4_indx = [indx for i,indx in enumerate(range(len(s_pca))) if clustering.labels_[i] == 3]

# generate w for cluster 1 (red cluster)
clusters = ['c1o4', 'c2o4', 'c3o4', 'c4o4']
cluster_indx = [c1_indx, c2_indx, c3_indx, c4_indx]
cluster_w = {}
cluster_b = {}

for i, cluster in enumerate(clusters):
    print('Generating w and b for cluster %s' % cluster)
    w_file = "%s_w_partial.npy" % cluster        # partial because not using all sequences
    b_file = "%s_b_partial.npy" % cluster        # partial because not using all sequences

    if os.path.exists(w_file):                                                          
        w_ER = np.load(w_file)       
        b = np.load(b_file)                                                                               

    else:  
        s_train = s[cluster_indx[i]]
        # Define wight matrix with variable for each possible amino acid at each sequence position               
        w_ER = np.zeros((mx.sum(),mx.sum()))                                                                     
        h0 = np.zeros(mx.sum())    
        #-------------------------------                                                                     
        # parallel                                                                                           
        start_time = timeit.default_timer()                                                                  
        res = Parallel(n_jobs = 20-2)(delayed(predict_w)                                                   
                (s_train,i0,i1i2,niter_max=10,l2=100.0)                                                          
                for i0 in range(n_var))                                                                      

        run_time = timeit.default_timer() - start_time                                                       
        print('run time:',run_time)                                                                          
        #------------------------------- 
        for i0 in range(n_var):
            i1,i2 = i1i2[i0,0],i1i2[i0,1]                                                                    

            h01 = res[i0][0]                                                                                 
            w1 = res[i0][1]

            h0[i1:i2] = h01                                                                                  
            w_ER[:i1,i1:i2] = w1[:i1,:]                                                                      
            w_ER[i2:,i1:i2] = w1[i1:,:]                                                                      

        # make w symmetric                                                                                   
        w_ER = (w_ER + w_ER.T)/2.                                                                            
        b = h0

        np.save(w_file, w_ER)
        np.save(b_file, b)

    cluster_w[cluster] = w_ER
    cluster_b[cluster] = b

# number of bias term
n_linear = mx_sum - n_var

from sklearn import svm
gp1 = s_pca[cluster_indx[c1-1],:3]
gp2 = s_pca[cluster_indx[c2-1],:3]
gp1_mean = np.mean(gp1, axis=0)
gp2_mean = np.mean(gp2, axis=0)

gp12 = np.append(gp1, gp2, axis=0)
clf = svm.SVC(kernel='linear')
y = np.zeros(len(gp12))
y[:len(gp1)] = 1
clf.fit(gp12,y)

def perp_dist(x1, y1, z1, a, b, c, d):
    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))
    return d/e


c1_indx = cluster_indx[c1-1]
import math
min_dist = 100.
print(len(c1_indx))
for i, pt in enumerate(gp1):
    pt_dist = perp_dist(pt[0],pt[1],pt[2], clf.coef_[0][0],clf.coef_[0][1],clf.coef_[0][2],clf.intercept_[0])
    if pt_dist < min_dist:
        min_dist = pt_dist
        min_id = c1_indx[i]
print(min_dist)
print(min_id)

boundary_seq = s[min_id]

start_time = timeit.default_timer()                                                                  
res = Parallel(n_jobs = 50)(delayed( w_seq_walk)                                                   
        (boundary_seq, i1i2, cluster_w['c%do4' % cw], cluster_b['c%do4'% cw], n_iter = niter,seed=i0, ncpu = 2)                                                          
        for i0 in range(nwalk))                                                                      

run_time = timeit.default_timer() - start_time    
print('run time:',run_time) 
np.save('%dBSW_%di_c%d%d_w%d.npy' % (nwalk,niter,c1,c2,cw), res)


