import sys,os
import timeit
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
import glob


data_path = '/data/cresswellclayec/covid_data/spikeprot1201/fasta_files/'
root_dir = '/data/cresswellclayec/covid_data/spikeprot1201/fasta_files/'


# Load fasta file for all covid proteins
from Bio import SeqIO



# Load list of fasta files
aligned_cov_fasta_files =  glob.glob(root_dir+'cov19_spike_aligned*.fasta')
print('Concatenating %d aligned files:\n' % len(aligned_cov_fasta_files),aligned_cov_fasta_files[:10])

ref_id = 'YP_009724390.1' 

concat_list = []
first_ref = True
for file_indx,aligned_file in enumerate(aligned_cov_fasta_files):

	try:
		# iterate through records in mini-alignment
		for i,record in enumerate(SeqIO.parse(aligned_file, "fasta")):
			# Once at to the end of orginal alignment:
			#	-- start concatenating
			if record.id == ref_id:
				if first_ref: 
					concat_list.append(record)
					first_ref = False
				else:	
					continue
			else:
				concat_list.append(record)
	except(UnicodeDecodeError):
		print('error at i = %d'%i)
		print(record)
        
print('Concatenated records count= %d'%len(concat_list))

with open("cov19_spike_full_aligned.fasta", "w") as output_handle:
	SeqIO.write(concat_list, output_handle, "fasta")
output_handle.close()
