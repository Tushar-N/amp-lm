# AMP-LM: A residue-level Language model for AMP sequence generation

Code accompanying the paper: [Computational antimicrobial peptide design and evaluation against multidrug-resistant clinical isolates of
bacteria](http://www.jbc.org/content/early/2017/12/19/jbc.M117.805499)


There are 4 components to the code-base:
- `data_harvesting`: Download and parse pages from [YADAMP](http://yadamp.unisa.it/)
- `residue_lm`: Train and sample a residue level language model
- `clustal_pruning`: Remove redundant sequences from sampled sequences using [clustalW](http://www.genome.jp/tools/clustalw/)
- `bilstm_ranking`: Rank the resulting sequences according to predicted MIC values using a bidirectional LSTM model

Details about the inputs and outputs for each component can be found in the run script below.

### Requirements

- [torch7](http://torch.ch/docs/getting-started.html)
- python packages: `pip install numpy beautifulsoup4 joblib`
- clustal: `sudo apt-get install clustalw`


for blstm service:  
- `luarocks install https://raw.githubusercontent.com/benglard/htmlua/master/htmlua-scm-1.rockspec`
- `luarocks install https://raw.githubusercontent.com/benglard/waffle/master/waffle-scm-1.rockspec`


### Instructions to run
The ```run.sh``` script can be used for end-to-end execution of the AMP-LM pipeline. It is recommended that the script be edited to include the GPU ID of any CUDA capable device in the system (training is very slow on the CPU).

To run only the BiLSTM model as a service:
```sh
cd bilstm_ranking
th service.lua <model.t7> -port 1337
# usage: http://127.0.0.1:1337/blstm/GLKIGKKIGPFLKLVKK
```


---

### Contact

For queries regarding the LSTM algorithm, contact Tushar Nagarajan (<tushar.nagarajan@gmail.com>)  
For queries regarding the experimental data, contact Deepesh Nagarajan (<1337deepesh@gmail.com>)  
For academic queries, contact Prof. Nagasuma Chandra (<sumachandra@gmail.com>)  


If you find this repository useful, please consider citing:
```
@article{nagarajan2017computational,
  title={Computational antimicrobial peptide design and evaluation against multidrug-resistant clinical isolates of bacteria},
  author={Nagarajan, Deepesh and Nagarajan, Tushar and Roy, Natasha and Kulkarni, Omkar and Ravichandran, Sathyabaarathi and Mishra, Madhulika and Chakravortty, Dipshikha and Chandra, Nagasuma},
  journal={Journal of Biological Chemistry},
  pages={jbc--M117},
  year={2017},
  publisher={ASBMB}
}
```
