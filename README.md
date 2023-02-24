# Data Information 
This gives the data files and associated description files 


	* METHOD_for_generating_allprot_spikeprot.txt 	-- This file is from GISAID, descripes how the got the sequences
	* FASTA_header_format_for_allprot_spikeprot.txt -- This file is from GISAID, describes FASTA header format of sequences


#  Data Pipeline
** This outlines the process of going from raw data to processed variant MSAs
	* posiiton columns are consistent across the different variant MSAs	
** Each enumerated item is the command to be run 
** Each command is described


1. -- python break_up_spike_fasta.py
	- requires ~500G RAM (maybe more)
	- Break up full GISAID fasta file for swarm alignment 
	  (these files, especially the full sequence file, can be replaced)
		- files used
			- reference sequence file 	: ncbi_spike_ref.fasta
			- full sequence file 		: spikeprot1201.fasta
		- subdivides full sequence file into groups of 10000 sequnece
		- creates swarm files which mafft aligns subdivided sequences to ref seq
			- mafft  --thread -1 --keeplength --addfragments subset.fasta ref.fasta > aligned.fasta
2. -- ./submit_spike_align_swarm 
	- Swarm align broken up sequences
	- files used 
		- subdivided fasta fasta files 		: cov19_spike_<swarm#>.fasta
		- reference sequences fasta file	: ncbi_spike_ref.dasta
	- generates reference-aligned files from subdivided fasta files in directory ./fasta_files/

3. -- python concat_spike_fasta.py
	- concatenates swarm-aligned files 
	- assumes referennce sequence ID: YP_009724390.1 
		- removes redundant reference sequences from aligned files
	- saves full spike aligned file 		: cov19_spike_full_aligned.fasta 
	

4. -- ./submit_sbatch_variant_extraction.script
	- Maps aligned sequences to variant metadata
	- files used:
		- cov19_spike_full_aligned.fasta
		- ../variant_surveillance/variant_surveillance.tsv (from GISAID)
	- Generates MSA numpy arrays for data-processessing
		- only chooses sequences whose accession ID is in variant metadata (varian_surveillance.tsv)
	- creates dictionary to access aligned sequences by accession ID
		- accession2msa_dict.pickle
	- creates dictionary of dataframes for each variant 
		- variant_dfs.pickle

5. 00 ./submit_sbatch_variant_data-processing.script
	- processes variant MSAs 
	- reference sequence array is HARDCODED in script (cov_variant_data-processing.py)
	- files used:
		- <variant name>_s.npy (for each variant)
		- data_processing.py (data_processing.data_processing_experiment())
	- creates unique seuqnece MSA
		- <variant name>_unique_s0.npy			-- Unique sequences in full MSA
		- <variant name>_unique_index.npy		-- Indices of unique sequence set in full MSA
	- processes variant sequence data using gap cols/seq of .2 conserved cols of .95
	- creates processed data files for each variant
		- <varian_name>_processed_s0.np			-- unique data processed (dp) MSA numpy array no cols removed
		- <varian_name>_processed_sindex.npy 		-- index of non-conserved columns
		- <varian_name>_processed_ipdb.npy 		-- index of reference sequence in MSA numpy array
		- <varian_name>_processed_cols_removed.npy	-- index of bad columns in MSA numpy array
		- <varian_name>_processed_seqs_removed.npy 	-- index of bad sequences which were removed
	- creates dictionary of variant processed data 
		- spike_s0_data.pickle 				-- dp MSA numpy array interseciton of all (variant) bad cools removed 
		- spike_index_data.pickle 			-- index array with interseciton of all (variant) bad cools removed 
		- spike_ipdb_data.pickle			-- index of ref seq in for each variant MSA
		- spike_removed_data.pickle			-- array of removed cols for each variant




