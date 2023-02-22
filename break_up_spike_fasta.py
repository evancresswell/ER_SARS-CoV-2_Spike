import sys, os
import ecc_tools as tools
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


# break up covid fasta file into smaller fasta for alignment with NCBI spike reference sequence.
# Write swarm file to mafft align spike protein sequences 100 at a time
# 	- swarm file is cov_spike_align.swarm
# 	- submit swarm job with following command:
# 		-	./submit_spike_align_swarm.script

# Creates aligned files which then need to be concatenated..
#	- Aligned file sequences will be REDUNDANT in once you've concatenated all the swarm output file!!!!


data_path = '/data/cresswellclayec/covid_data/spikeprot1201/fasta_files'
root_dir = '/data/cresswellclayec/covid_data/spikeprot1201/'
data_out = '/data/cresswellclayec/covid_data/spikeprot1201/fasta_files/'



ref_prot_file = data_path + '/ncbi_spike_ref.fasta'
subject_prot_file = data_path + '/spikeprot1201.fasta'


aligned_prot_file = data_out + '/cov_spike_aligned.fasta'

import re
#subject_prot = re.sub(".fasta","",subject_prot_file)
subject_prot = 'cov19_spike'

nucleotide_letters_full = np.array(['A','C','G','T','N','R','Y','S','W','K','M','B','D','H','V','U','-'])

from Bio import SeqIO

f = open('cov_spike_align.swarm','w')

with open(subject_prot_file,"r") as handle:
	fasta_sequences = SeqIO.parse(handle,'fasta')

	records = []
	try:
		for i,record in enumerate(fasta_sequences):

			records.append(record)

			if i%10000 ==0:
				print('\n\nwriting file with sequences: \n')	
				for record_sub in records:
					print(record_sub.id)
				
				out_file1 = data_out+subject_prot+'_%d.fasta'%i
				with open(out_file1,"w") as output_handle:
					SeqIO.write(records,output_handle,"fasta")
				output_handle.close()

				out_file2 = data_out+subject_prot + '_aligned_%d.fasta' % i
				#f.write("muscle -profile -in1 %s -in2 %s -out %s\n"%(ref_prot_file, out_file1,out_file2))
				# Mafft tips: https://mafft.cbrc.jp/alignment/software/closelyrelatedviralgenomes.html
				f.write("mafft  --thread -1 --keeplength --addfragments %s %s > %s\n"
					% (out_file1,ref_prot_file, out_file2))
				# Empty records list for next batch
				records = []
	except(UnicodeDecodeError):
		print('error at i = %d'%i)
		print(record)
        

handle.close()	
f.close()

