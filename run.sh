# Data Harvesting: Download YADAMP pages and parse files for sequences/MIC data.
# Stores sequence data in residue_lm/data/residue_lstm_input.txt and sequence+MIC data in bilstm_pruning/data/bilstm_prune_input.txt
cd data_harvesting

# unzip the pages instead of downloading them from the web
# bash download_yadamp.sh
unzip YADAMP_pages.zip

mkdir ../residue_lm/data
mkdir ../bilstm_ranking/data
python parse_yadamp.py
cd ..

echo "*----- Data Harvesting complete ------*"


# Residue LM training: Train a residue-level language model over the input sequences and sample it to generate candidate sequences.
# Saves a .t7 model in the residue_lstm/cv/ directory which is used to sample.
# The output of the sampling stage is stored in clustal_pruning/sampled_seqs.txt
GPU_ID=1 #training is VERY slow on CPU. Set gpuid>0 for GPU.
cd residue_lm
th train.lua -gpuid $GPU_ID
th sample.lua $(echo cv/*.t7) -gpuid $GPU_ID
cd ..

echo "*----- Residue Level LM trained -----*"

# Distinct Sequence Pruning: Use ClustalW to remove very similar sequences from the set of generated sequences.
# The output of the pruning stage is stored in clustal_pruning/distinct_seqs.txt
# NOTE: sudo apt-get install clustalw before doing this
cd clustal_pruning
python prune_candidates.py
cd ..


echo "*-----Sequence Pruning Complete -----*"

# BiLSTM Ranking Model: Trains a BiLSTM model on YADAMP data and ranks the generated (unique) sequences according to predicted MIC.
# Saves a .t7 model to bilstm_ranking/cv/ and generates the final output-- bilstm_ranking/data/output_preds.txt.
# Each line of this file contains a sequence and its predicted MIC value (\t separated). This may be used to select top sequences (lowest predicted MIC) to synthesize.
cd bilstm_ranking 
th train.lua -gpuid $GPU_ID
th test.lua $(echo cv/*.t7) -gpuid $GPU_ID
cd ..

echo "*-----BLSTM Ranking Complete -----*"


