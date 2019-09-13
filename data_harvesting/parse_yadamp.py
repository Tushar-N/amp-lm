from bs4 import BeautifulSoup
import requests
import re
import cPickle as pickle

#------------------------ parse html pages ---------------------------------------#
AMPs=[]
for pep_i in range(1,2526):
	if pep_i%100==0:
		print pep_i

	with open('YADAMP_pages/peptide_%04d.html'%pep_i, 'r') as f:
		data=f.read()
	data_cells=BeautifulSoup(data, 'lxml').findAll('td', {'class':'_sibody'})
	MIC_cells=BeautifulSoup(data, 'lxml').findAll('td', {'class':'_sibodyMIC'})

	helicity=float(data_cells[3].string.strip())
	sequence=re.search(r'<b>([A-Z]+)<', str(MIC_cells[0])).groups()[0]
	MIC_data=[MIC_cells[k].string.strip() for k in range(1,8)]
	for i in range(len(MIC_data)):
		if len(MIC_data[i])<1:
			MIC_data[i]='0'

	MIC_data=[m.replace(',','.') for m in MIC_data if len(m)>0] #stupid commas
	MIC_data=[float(m) for m in MIC_data if len(m)>0]

	MIC_extra=[MIC_cells[k].string.strip() for k in range(8,11)]

	YADAMP_data=[helicity]+[sequence]+MIC_data+MIC_extra
	keys=['helicity','sequence','ecoli','aeruginosa','typhimurium','aureus','luteus', 'subtilis', 'albicans', 'other', 'gram+','gram-']
	AMPs.append({k:v for k,v in zip(keys, YADAMP_data)})

#--------------------------generate training data-----------------------------------#


with open('../residue_lm/data/residue_lstm_input.txt','w') as f:
	for amp in AMPs:
		if amp['helicity']>5 and len(amp['sequence'])<30:
			f.write('%s\n'%(amp['sequence']))

with open('../bilstm_ranking/data/bilstm_prune_input.txt','w') as f:
	amp_with_mic=[amp for amp in AMPs if amp['ecoli']>0]
	for amp in amp_with_mic:
		if amp['helicity']>5 and len(amp['sequence'])<30 and amp['ecoli']<=100:
			f.write('%s\t%f\t%f\n'%(amp['sequence'], amp['helicity'], amp['ecoli']))