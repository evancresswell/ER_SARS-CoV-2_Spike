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

ref_seq = np.array(list('MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'))
n_cols = len(ref_seq)
ref_seq = ref_seq.reshape(1,n_cols)
print('Spike reference sequence shape: ', ref_seq.shape)

#========================================================================================
data_path = '/data/cresswellclayec/covid_data/spikeprot1201/'
root_dir = '/data/cresswellclayec/covid_data/'

cpus_per_job = 40



variants = ['Omicron', 'Delta', 'Zeta', 'Beta', 'Gamma', 'Iota', 'Kappa', 'Alpha', 'Eta']



# data processing -- requires 36:00:00 time and 2000g mem
s0_dict = {}
accession_dict = {}
index_dict = {}
ipdb_dict = {}
removed_dict = {}

for variant_name in variants:
    print('Loading/Generating MSA array of unique sequences for %s' % variant_name)
    if not os.path.exists('%s_unique_s0.npy' % variant_name):
    
        print('Creating unique MSA array for %s' % variant_name)
        s0 = np.load("%s_s.npy" % variant_name)                                                    
        variant_accession = np.load("%s_accession.npy" % variant_name)
        print(s0[:20,:])                                                                                             


        # Decode Unicoded aa characters
        try:            
            s0 = np.char.decode(s0)                                                  
        except(UnicodeDecodeError):
            # go through sequences individual and throw out baddies            
            rows_to_remove = []
            new_s0 = []                       
            for i, arr in enumerate(s0):
                try:
                    arr = np.char.decode(arr)
                    new_s0.append(arr)
                except(UnicodeDecodeError):
                    rows_to_remove.append(i)
                    pass
            s0 = np.array(new_s0)
        except(AttributeError):
            print('s0 does not need decoding')
            print(s0[0])
            print(s0.shape)


	# Filter out duplicate sequences in MSA
        print('unique aa in s0 columns')
        num_unique_aa = []
        for col in range(s0.shape[1]):
            unique_col = np.unique(s0[:,col],return_counts=True)
            num_unique_aa.append(len(unique_col[0]))
        print(num_unique_aa)

        unique_seqs = np.unique(s0,axis=0, return_index=True)
        np.save('%s_unique_index.npy' % variant_name ,unique_seqs[1])
        print('full s0: ', s0.shape)
        s0 = unique_seqs[0] 
        variant_accession = variant_accession[unique_seqs[1])
        print('unique s0:', s0.shape)
        np.save('%s_unique_s0.npy' % variant_name,s0)
        np.save('%s_unique_accession.npy' % variant_name,variant_accession)
        print(unique_seqs[1])
        np.save('%s_unique_index.npy' % variant_name,unique_seqs[1])
        if len(variant_accession) != len(s0):
            print(len(variant_accession), len(s0))
            print('after removing duplicates, variant accession array and MSA are not the same length..\nERROR!!!\nDATA PROCESSING NOT COMPLETED')
            sys.exit()
    
    else:
        print('Loading pre-existing MSA array for %s' % variant_name)
        s0_indx = np.load('%s_unique_index.npy' % variant_name)
        s0 = np.load('%s_unique_s0.npy' % variant_name)
        variant_accession = np.load('%s_unique_accession.npy' % variant_name)

    # filter out empty sequences in MSA    
    empty_seqs = []
    for i, seq_array in enumerate(s0):
        if '' in seq_array:
            empty_seqs.append(i)
    print(len(empty_seqs))
    s0 = np.delete(s0,empty_seqs,axis=0) 
    variant_accession = np.delete(variant_accession,empty_seqs,axis=0) 
    
    print('adding ref_seq to s0 (',s0.shape,')')
    s0 = np.concatenate((ref_seq,s0), axis=0)
    print('new s0 shape: ', s0.shape)
    print('data processing...\n\n')
    s0,cols_removed,s_index,s_ipdb,orig_seq_len,bad_seq_indx = dp.data_processing_experiment(s0, variant_name, 0,\
                                gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols_thresh=0.95,remove_cols=False,n_cpus=cpus_per_job)
    variant_accession = np.delete(variant_accession, bad_seq_indx, axis=0) # remove bad sequences which were removed from s0 during dp
        if len(variant_accession) != len(s0):
            print(len(variant_accession), len(s0))
            print('after data processing, variant accession array and MSA are not the same length..\nERROR!!!\nDATA PROCESSING NOT COMPLETED')
            sys.exit()

    print('\n\ns0 shape after data processing' , s0.shape,'\n\n')

    
    
    # Save processed data by variant in case of crash
    print('writing processed data file')
    np.save('%s_processed_s0.npy' % variant_name, s0)
    np.save('%s_processed_accession.npy' % variant_name, variant_accession)
    np.save('%s_processed_sindex.npy' % variant_name, s_index)
    np.save('%s_processed_ipdb.npy' % variant_name, s_ipdb)
    np.save('%s_processed_cols_removed.npy' % variant_name, cols_removed)
    np.save('%s_processed_seqs_removed.npy' % variant_name,bad_seq_indx )


    s0_dict[variant_name] = s0
    accessions_dict[variant_name] = variant_accession
    index_dict[variant_name] = s_index
    ipdb_dict[variant_name] = s_ipdb
    removed_dict[variant_name] = cols_removed

# verify dimensionality of all variant alignments.
for i, variant_name in enumerate(s0_dict.keys()):
    if i == 0:
        cols = s0_dict[variant_name].shape[1]
    elif cols != s0_dict[variant_name].shape[1] or cols != len(index_dict[variant_name]):
        print('not all variant MSA (s0) have the same number of columns')
        sys.exit(0)


# get intersection of removed columns from all variants. 
from functools import reduce
removed_cols_intersect = reduce(np.intersect1d, tuple([removed_dict[vn] for vn in removed_dict.keys()]))
print('All variants share %d removed columns' % len(removed_cols_intersect))
print(removed_cols_intersect)
print(type(removed_cols_intersect))

for i, variant_name in enumerate(s0_dict.keys()):
    s0_dict[variant_name] = np.delete(s0_dict[variant_name], removed_cols_intersect, axis =1) 
    index_dict[variant_name] = np.delete(index_dict[variant_name], removed_cols_interesct) 

# verify dimensionality of all variant alignments again...
for i, variant_name in enumerate(s0_dict.keys()):
    if i == 0:
        cols = s0_dict[variant_name].shape[1]
    elif cols != s0_dict[variant_name].shape[1] or cols != len(index_dict[variant_name]):
        print('not all variant MSA (s0) have the same number of columns after removing bad columns')
        sys.exit(0)

input_data_file = 'spike_s0_data.pickle'
# Save final dicts    
with open(input_data_file, 'wb') as f:
	pickle.dump(s0_dict, f)
f.close()
input_data_file = 'spike_index_data.pickle'
with open(input_data_file, 'wb') as f:
	pickle.dump(index_dict, f)
f.close()
input_data_file = 'spike_ipdb_data.pickle'
with open(input_data_file, 'wb') as f:
	pickle.dump(ipdb_dict, f)
f.close()
input_data_file = 'spike_removed_data.pickle'
with open(input_data_file, 'wb') as f:
	pickle.dump(removed_dict, f)
f.close()
print('done')
