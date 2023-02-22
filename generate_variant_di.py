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


#========================================================================================
data_path = '/data/cresswellclayec/covid_data/'
root_dir = '/data/cresswellclayec/covid_data/'




#========================================================================================
data_path = '/data/cresswellclayec/covid_data/spikeprot1201/'
root_dir = '/data/cresswellclayec/covid_data/'

cpus_per_job = 40


# Swarm aligned file  
ref_file = root_dir+"spike_ref.fa"

variant_name = sys.argv[1]


# Load appropriate files
input_data_file = 'spike_s0_data.pickle'
with open(input_data_file, 'rb') as f:
	s0_dict = pickle.load(f)

input_data_file = 'spike_index_data.pickle'
with open(input_data_file, 'rb') as f:
	index_dict = pickle.load(f)

input_data_file = 'spike_ipdb_data.pickle'
with open(input_data_file, 'rb') as f:
	ipdb_dict = pickle.load(f)

input_data_file = 'spike_removed_data.pickle'
with open(input_data_file, 'rb') as f:
	removed_dict = pickle.load(f)


s_ipdb = ipdb_dict[variant_name]
s0 = s0_dict[variant_name]


print('\n\n DATA LOADED!!! \n\n')

# lets try it --->
# if  data processing do not compute
#if preprocessing:
#    sys.exit()


# Expectation Reflection -- 5000g mem   2-00:00:00

# 8/10/22 --> Running out of ram with 3000g and 20 cpu so lets trim unecessary data
#nucleotide_letters_full = np.array(['A','C','G','T','N','R','Y','S','W','K','M','B','D','H','V','U','-'])
#s0_letter = gdp.convert_number2letter(s0)
#print('s0 letter shape: ',s0_letter.shape)

print(' pre-processed s0        shape: ',s0.shape)

# get the intersection of removed columns in all variants.
# we want s0 with union of retained columns across variants.

from functools import reduce
removed_cols_intersect = reduce(np.intersect1d, tuple([removed_dict[vn] for vn in removed_dict.keys()]))
print('All variants share %d removed columns' % len(removed_cols_intersect))
print(removed_cols_intersect)
print(type(removed_cols_intersect))

print('deleting intersection of removed columns across all variants')
s0_temp = np.load('%s_processed_s0.npy' % variant_name)
print('before deleting intersection of removed cols: ', s0_temp.shape)
s0_temp = np.delete(s0_temp,removed_cols_intersect, axis =1)
print('after deleting intersection of removed cols: ', s0_temp.shape)

# s0 was defined with faulty intersectin during data processing. being fixed and rerun. in the time being do this here.
s0 = s0_temp



n_var = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

print('\n\n MX CREATED!!! \n\n')


#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
onehot_encoder = OneHotEncoder(sparse=False)
print('%s s0: ' % variant_name,s0.shape,'\n',s0)
#print('s0 letter: ',s0_letter.shape,'\n',s0_letter)
s = onehot_encoder.fit_transform(s0)

print('\n\n S_ONEHOT CREATED!!! \n\n')

mx_sum = mx.sum()
my_sum = mx.sum() #!!!! my_sum = mx_sum

w = np.zeros((mx_sum,my_sum))
h0 = np.zeros(my_sum)


#=========================================================================================
def predict_w(s,i0,i1i2,niter_max,l2):
    
    print('starting parallel sim %d' % i0) 
    #print('i0:',i0)
    i1,i2 = i1i2[i0,0],i1i2[i0,1]

    x = np.hstack([s[:,:i1],s[:,i2:]])
    y = s[:,i1:i2]

    h01,w1 = ER.fit(x,y,niter_max,l2)

    return h01,w1

#-------------------------------
# parallel

print('\n\n STARTING PARALLEL SIMULATION!!! (%d processes) \n' % n_var)
res = Parallel(n_jobs = cpus_per_job-2)(delayed(predict_w)\
        (s,i0,i1i2,niter_max=10,l2=100.0)\
        for i0 in range(n_var))

print('\n FINISHED PARALLEL SIMULATION!!! \n\n')
#-------------------------------
for i0 in range(n_var):
    i1,i2 = i1i2[i0,0],i1i2[i0,1]
       
    h01 = res[i0][0]
    w1 = res[i0][1]

    h0[i1:i2] = h01    
    w[:i1,i1:i2] = w1[:i1,:]
    w[i2:,i1:i2] = w1[i1:,:]


print('\n\n GENERATING DIRECT INFO!!! \n\n')
# make w to be symmetric
w = (w + w.T)/2.
np.save(data_path+'%s_w.npy' % variant_name, w)
np.save(data_path+'%s_b.npy' % variant_name, h0)

di = direct_info(s0,w)

er_gen_DI = sort_di(di)

for site_pair, score in er_gen_DI[:5]:
    print(site_pair, score)

with open(data_path+'%s_spike_DI.pickle' % variant_name, 'wb') as f:
    pickle.dump(er_gen_DI, f)
f.close()


