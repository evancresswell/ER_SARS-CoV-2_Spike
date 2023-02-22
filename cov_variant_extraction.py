import sys,os
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
import expectation_reflection as ER
from joblib import Parallel, delayed
import numpy as np
import pickle
from Bio import Phylo
import networkx, pylab
import pandas as pd
import Bio.SubsMat.FreqTable
from Bio.Align import AlignInfo
from Bio import AlignIO, SeqIO

# Extract all the variant data of the msa_file. 
#  -- Get accessions of invididual sequences from msa and allocate by variant from variant_surveillance data

#========================================================================================
data_path = '/data/cresswellclayec/covid_data/spikeprot1201/fasta_files/'
root_dir = '/data/cresswellclayec/covid_data/'

cpus_per_job = 40


# # Swarm aligned file  
# msa_file = data_path+"spikeprot1201.fasta"
msa_file = data_path+"cov19_spike_full_aligned.fasta" # broken-up, mafft-aligned, then concatenated spikeprot1201.fasta 

# ref_file = root_dir+"EPI_ISL_402124.fasta"
ref_id = 'YP_009724390.1' 

variants = ['Omicron', 'Delta', 'Zeta', 'Beta', 'Gamma', 'Iota', 'Kappa', 'Alpha', 'Eta']

# load sequence records as numpy array of aa characters
# 	- keep track of which sequence is the reference sequence
all_seqs = []
all_seqs_array = []
with open(msa_file, 'r') as f:
    seq_iter = SeqIO.parse(f,'fasta')
#     seq_iter = AlignIO.parse(f,'fasta')
    try:
        for i, record in enumerate(seq_iter):
            if record.id == ref_id:
                print('reference sequences at index %d' % i)
                i_ref = i
            all_seqs.append(record) 
            all_seqs_array.append(np.array(list(record.seq)))
       
    except(UnicodeDecodeError):
        print('error at i = %d'%i)
all_seqs_array = np.array(all_seqs_array)
print(len(all_seqs))


# get the accessions of all the sequences 
# 	- save the accestion as a dict of the index in the MSA
# 	  this way we can go back and get the MSA index given the accession ID
#	  this is important for comparing to variant metadata
accessions = []
accessions_indx = []
for i,seq in enumerate(all_seqs):
    if 'EPI_ISL' in seq.id:
        for id_str in seq.id.split('|'):
            if 'EPI_ISL' in id_str:
                accessions.append(id_str)
                accessions_indx.append(i)
                break
accession_as_dict = dict(zip(accessions,range(0,len(accessions))))
with open('full_accession2msa_dict.pickle', 'wb') as handle:
    pickle.dump(accession_as_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
# Load variant metadata - gives virus metadata 
variant_metadata = pd.read_csv("/data/cresswellclayec/covid_data/variant_surveillance/variant_surveillance.tsv",sep='\t',low_memory=False)
print(variant_metadata.head())
print(len(variant_metadata))
variant_metadata = variant_metadata.dropna(subset=['Accession ID', 'Variant'])
print(len(variant_metadata))
variant_metadata['Collection date'] = pd.to_datetime(variant_metadata['Collection date'])

if not os.path.exists('variant_dfs.pickle'):
    # get dictionary of dataframes for each variant.
    variant_list = variant_metadata.Variant.unique()
    
    variant_dfs = {}
    for variant in variant_list:

        print('full variant title: ', variant)
        print('variant name: ', str(variant).split(' ')[1])
        variant_dfs[variant] = variant_metadata.loc[variant_metadata['Variant'] == variant]
        print('length of df: ', len(variant_dfs[variant]))

    new = {}
    for variant_key in variant_dfs.keys():
        variant_name = str(variant_key).split(' ')[1]
        if variant_name not in variants:
            print(variant_name, ' key is not a known varianat\n VARIANT EXTRACTION NOT COMPLETED!!')
            sys.exit()

        new[variant_name] = variant_dfs[variant_key]
    
    variant_dfs = new.copy()
    del new
    print('Generated dictionary of variant metadata dataframes')
    print(variant_dfs.keys())
    print(variant_dfs[variant_name1].head())
    
    # save variant dataframes
    with open('variant_dfs.pickle', 'wb') as handle:
        pickle.dump(variant_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # load variant dataframes
    with open('variant_dfs.pickle', 'rb') as handle:
        variant_dfs = pickle.load(handle)


# Go back through and create MSA of each variant
# 	- Use the accession_as_dict to quickly find if accession exists in MSA, and what it's index is
# 	- This way we only use sequences for which we have metadata
# 	- Save the resulting MSA (for which we have a link back to metadata) for data-processesing and analysis
for variant in variant_dfs.keys():
    variant_accession = variant_dfs[variant]['Accession ID'].tolist()
    variant_indx = []
    for acc in variant_accession:
        try:
            variant_indx.append(accession_as_dict[acc])
        except(KeyError):
            continue
    
    variant_seqs = [all_seqs[accessions_indx[i]] for i in variant_indx] 
    
    # update accession array to save it in accordance with MSA
    variant_accession = np.array(variant_accession)


    variant_char_mat = []
    for i,record in enumerate(variant_seqs):
        variant_char_mat.append(np.array(list(record.seq)))
    print('variant_char_mat created')
    print(len(variant_char_mat))
    print(variant_char_mat[0])
    variant_char_mat = np.array(variant_char_mat)
    print('%s variant sequences in MSA: '% variant, len(variant_char_mat))
    if variant == 'GH/490R':
        try:
            np.save('GH-490R_s.npy' , variant_char_mat)
            np.save('GH-490R_accession.npy' , variant_accession)
        except:
            continue
    else:
        np.save('%s_s.npy' % variant , variant_char_mat)
        np.save('%s_accession.npy' % variant , variant_accession)




