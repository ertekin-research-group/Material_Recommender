import torch
import sys
sys.path.append('../MatSciBERT')
from normalize_text import normalize
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizerFast
from transformers import BertModel

from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import re
from tqdm import tqdm
import os
import numpy as np
#from parrot import Parrot

def sort_composition(compositions):
	"""_summary_

	Args:
		compositions (_type_): _description_

	Returns:
		_type_: _description_
	"""

	composition_name_new = []
	
	for name in compositions:
		composition = name
		composition_split = re.findall('[A-Z][^A-Z]*', composition)
		composition_split.sort()

		for i, element in enumerate(composition_split):
			if not element[-1].isnumeric():
				composition_split[i] = composition_split[i] + '1'
		composition_sorted = "".join(composition_split)
		composition_name_new.append(composition_sorted)

	return composition_name_new


def parse_para_phrase(sentence, para_list, n=5):
	"""_summary_

	Args:
		sentence (_type_): _description_
		para_list (_type_): _description_
		n (int, optional): _description_. Defaults to 5.

	Returns:
		_type_: _description_
	"""
	para_sentence = []
	phrases = sentence.split('. ')

	for i in range(n):
		para = []
		for j,phrase in enumerate(phrases):
			if para_list[j] is not None:
				if len(para_list[j])>i:
					para.append(para_list[j][i][0]+'.')
				else:
					if j == len(phrases)-1:
						para.append(phrase)
					else:
						para.append(phrase+'.')
			else:
				if j == len(phrases)-1:
					para.append(phrase)
				else:
					para.append(phrase+'.')

		para_sentence.append(' '.join(para))

	return para_sentence


def run_bert(data, model_type, model, tokenizer, para=False):
	"""_summary_

	Args:
		data (_type_): _description_
		model_type (_type_): _description_
		model_dir (_type_): _description_
		save_dir (_type_): _description_
		para (bool, optional): _description_. Defaults to False.
	"""


	# if model_type == 'matscibert':
	# 	model = AutoModel.from_pretrained(model_dir)
	# 	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	# elif model_type == 'matbert':
	# 	tokenizer = BertTokenizerFast.from_pretrained(model_dir, do_lower_case=True)
	# 	model = BertModel.from_pretrained(model_dir)

	#sentences = ['SiO2 is a network former.']
	embeddings = []
	for sentence in tqdm(data):


		if model_type == 'matscibert':
			norm_sents = [normalize(s) for s in [sentence]]
			tokenized_sents = tokenizer(norm_sents)
		elif model_type == 'matbert':
			tokenized_sents = tokenizer.batch_encode_plus([sentence])

		tokenized_sents = {k: torch.Tensor(v).long() for k, v in tokenized_sents.items()}

		token_size = int(tokenized_sents['input_ids'].shape[1])
		sentence_embedding = []

		for i in range(0,token_size,512):
			with torch.no_grad():
				last_hidden_state = model(input_ids=tokenized_sents['input_ids'][:,i:i+512],token_type_ids=tokenized_sents['token_type_ids'][:,i:i+512],
				attention_mask=tokenized_sents['attention_mask'][:,i:i+512])[0]

			sentence_embedding.append(last_hidden_state.detach().cpu().numpy()[0])
			#np.save(save_dir+'/{}.npy'.format(composition),last_hidden_state.detach().cpu().numpy()[0])

		sentence_embedding = np.concatenate(sentence_embedding).mean(0)

		if para:
			para_list = eval(sentence)
			para_sentences = parse_para_phrase(sentence,para_list,5)
			para_embedding = []
			for sentence in para_sentences:
				if model_type == 'matscibert':
					norm_sents = [normalize(s) for s in [sentence]]
					tokenized_sents = tokenizer(norm_sents)
				if model_type == 'matbert':
					tokenized_sents = tokenizer.batch_encode_plus([sentence])

				tokenized_sents = {k: torch.Tensor(v).long() for k, v in tokenized_sents.items()}
					
				token_size = int(tokenized_sents['input_ids'].shape[1])
				para_sentence_embedding = []
				for i in range(0,token_size,512):
					with torch.no_grad():
						last_hidden_state = model(input_ids=tokenized_sents['input_ids'][:,i:i+512],token_type_ids=tokenized_sents['token_type_ids'][:,i:i+512],
						attention_mask=tokenized_sents['attention_mask'][:,i:i+512])[0]

					para_sentence_embedding.append(last_hidden_state.detach().cpu().numpy()[0])
				
				para_sentence_embedding = np.concatenate(para_sentence_embedding).mean(0)
				para_embedding.append(para_sentence_embedding.reshape(1,-1))

			embeddings.append(np.concatenate((sentence_embedding.reshape(1,-1),np.concatenate(para_embedding))).mean(0).reshape(1,-1))

		else:
			embeddings.append(sentence_embedding.reshape(1,-1))

	return embeddings



if __name__ == "__main__":

	#save_dir = '../'
	data = pd.read_pickle('robo_descriptions.pkl')
	#data = pd.read_csv('../robo_data_kappa_para.csv')

	# if not os.path.isfile(save_dir+'robo_data_kappa_with_prop.pkl'):
	# 	robo_data = make_robodata(save_dir, data)
	# else:
	# 	robo_data = pd.read_pickle(save_dir+'robo_data_kappa_with_prop.pkl')

	save_dir = 'matbert_robo_descriptions_with_embed_noprop_composition.pkl'
	#model_dir = 'm3rg-iitd/matscibert'
	model = 'matbert'
	model_dir = '../matbert_ner_models/model_files/matbert-base-uncased'

	run_bert(data,model,model_dir,save_dir, False)