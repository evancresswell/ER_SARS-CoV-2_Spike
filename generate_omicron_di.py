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

cpus_per_job = int(sys.argv[1])


# Swarm aligned file  
msa_file = data_path+"spikeprot1201.fasta"
msa_file = root_dir+"omicron.fa"
ref_file = root_dir+"spike_ref.fa"

s0 = np.load("omicron_s.npy")
s0 = np.char.decode(s0)

s0 = np.delete(s0, -1, axis=1) # remove * from alignment

# data processing -- requires 36:00:00 time and 2000g mem
preprocessing = False
preprocessing = True
if preprocessing:
	# Preprocess data using ATGC
	s0,cols_removed,s_index,s_ipdb,orig_seq_len = dp.data_processing_experiment(s0, 'omicron', 0,\
    				gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.975)
	print('after data processing' , s0.shape)
else:
        #if os.path.exists(input_data_file):
        if os.path.exists(input_data_file):
            	with open(input_data_file, 'rb') as f:
            		pf_dict = pickle.load(f)
            	f.close()
            
            	s0 = pf_dict['s0']
            	print(s0.shape)
            	s_index = pf_dict['s_index']
            	cols_removed = pf_dict['cols_removed']
            	s_ipdb = 0
        else:
                # Load appropriate files
                print('loading files\n')
                aligned_genome_file = "cov_processed.npy"
                removed_cols_file = "covGEN_removed_cols.npy"
                s_index_file = "covGEN_s_index.npy"
                
                s0 = np.load(aligned_genome_file,'c')
                cols_removed = np.load(removed_cols_file)
                s_index = np.load(s_index_file)
                print('trimmed s loaded, ', s0.shape)

                s_ipdb = 0
 

    
saving_preprocessed = False
if saving_preprocessed:
	# Save processed data
	print('writing fasta file')
	msa_outfile, ref_outfile = gdp.write_FASTA(s0,'COV_GENOME',s_ipdb,path=data_path)	
	pf_dict = {}
	pf_dict['s0'] = s0
	pf_dict['s_index'] = s_index
	pf_dict['s_ipdb'] = s_ipdb
	pf_dict['cols_removed'] = cols_removed

	with open(input_data_file, 'wb') as f:
		pickle.dump(pf_dict, f)
	f.close()

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

print('s0        shape: ',s0.shape)


n_var = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

print('\n\n MX CREATED!!! \n\n')


#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
onehot_encoder = OneHotEncoder(sparse=False)
print('s0: ',s0.shape,'\n',s0)
#print('s0 letter: ',s0_letter.shape,'\n',s0_letter)
s = onehot_encoder.fit_transform(s0)

print('\n\n S_ONEHOT CREATED!!! \n\n')

mx_sum = mx.sum()
my_sum = mx.sum() #!!!! my_sum = mx_sum

w = np.zeros((mx_sum,my_sum))
h0 = np.zeros(my_sum)

saving_onehot = False
if saving_onehot:
	nucleotide_count = {}
	if s0_letter.shape[1] == len(s_index):
		for i in range(s0_letter.shape[1]):
			column = s0_letter[:,i]
			#print(column)
			letter_counts = []
			for letter in nucleotide_letters_full:
			
				letter_counts.append( np.count_nonzero(column == letter) )
			nucleotide_count[s_index[i]] = letter_counts
			print('nucleotide_counts[%d] : '%(s_index[i]),nucleotide_count[s_index[i]])

		print(len(nucleotide_count))	
		if len(sys.argv) > 2:
			with open('cov_genome_nucleotide_count.pickle', 'wb') as f:
				pickle.dump(nucleotide_count, f)
			f.close()
		else:
			with open('cov_genome_clade_%s_nucleotide_count.pickle'%(clade_file[:2]), 'wb') as f:
				pickle.dump(nucleotide_count, f)
			f.close()

	else:
		print('S_index and s0 shape do not match!!')
	for i0 in range(n_var):
		i1,i2 = i1i2[i0,0],i1i2[i0,1]

		x = np.hstack([s[:,:i1],s[:,i2:]])
		y = s[:,i1:i2]
		print('x:\n',x)
		print('y:\n',y)
		#for x_mini in x:
		    #print(len(x_mini)) # 3343
		print('x:\n',len(x)) # 137634 
		print('y:\n',len(y)) # 137634
		sys.exit()


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
res = Parallel(n_jobs = cpus_per_job-10)(delayed(predict_w)\
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
di = direct_info(s0,w)

er_gen_DI = sort_di(di)

for site_pair, score in er_gen_DI[:5]:
    print(site_pair, score)

if len(sys.argv) > 2:
	with open(root_dir+ "cov_genome_clade_%s_DI.pickle"%(clade_file[:2]),'wb') as f:
	    pickle.dump(er_gen_DI, f)
	f.close()
else:
	with open(root_dir+'cov_genome_DI.pickle', 'wb') as f:
	    pickle.dump(er_gen_DI, f)
	f.close()


