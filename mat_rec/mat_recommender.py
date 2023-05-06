import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from pymatgen.core import Structure
from robocrys import StructureCondenser, StructureDescriber
from transformers import BertTokenizerFast
from transformers import BertModel
from fastdist import fastdist 
import scipy.stats as ss
import mat_rec.model as model
from mat_rec.get_embedding_bert import *
from mat_rec.utils import *
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Recommender():

    def __init__(self, composition_embedding_dir=None, structure_embedding_dir=None, bert_model_dir=None):
        """_summary_

        Args:
            composition_embedding_dir (_type_, optional): _description_. Defaults to None.
            structure_embedding_dir (_type_, optional): _description_. Defaults to None.
            bert_model_dir (_type_, optional): _description_. Defaults to None.
        """

        self.composition_embeddings = None
        self.structure_embeddings = None
        self.search_results = dict()
        self.rank_results = dict()

        if composition_embedding_dir:
            composition_embedding_data = pd.read_hdf(composition_embedding_dir)
            self.composition_embeddings = np.concatenate(composition_embedding_data['embeddings'])
            self.composition_names = np.array(composition_embedding_data['composition_name'])
            print('composotion embeddings loaded successfully')

        if structure_embedding_dir:
            structure_embedding_data = pd.read_hdf(structure_embedding_dir)
            self.structure_embeddings = np.concatenate(structure_embedding_data['embeddings'])
            self.structure_names = np.array(structure_embedding_data['composition_name'])
            print('structure embeddings loaded successfully')

        if bert_model_dir:
            self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_dir, do_lower_case=True)
            self.lm = BertModel.from_pretrained(bert_model_dir)


    def get_mmoe_model(self, sparse_feature_dim=None, config=None, model_config=None, model_dir=None):
        """_summary_

        Args:
            config (_type_, optional): _description_. Defaults to None.
            model_config (_type_, optional): _description_. Defaults to None.
            model_dir (_type_, optional): _description_. Defaults to None.
        """
        self.mmoe_model = model.mat_MMoE(config, model_config)

        if model_dir:
            self.dense_feature_names = ['feature_'+str(i) for i in range(self.structure_embeddings.shape[1]*2)]
            if sparse_feature_dim:
                self.sparse_feature_names = ['feature_'+str(i+len(self.dense_feature_names)) for i in range(sparse_feature_dim)]

            self.mmoe_model.load_model(model_dir, self.dense_feature_names, self.sparse_feature_names)



    def get_description(self, cif_pths):
        
        condenser = StructureCondenser()
        describer = StructureDescriber()

        descriptions = []
    
        for cif_pth in tqdm(cif_pths):
            structure = Structure.from_file(cif_pth[1])

            condensed_structure = condenser.condense_structure(structure)
            description = describer.describe(condensed_structure)

            descriptions.append((cif_pth[0], description))
        
        return descriptions


    def get_embedding(self, descriptions):

        #compositions = list(descriptions.keys())
        compositions = [d[0] for d in descriptions]
        structure_descriptions = [d[1] for d in descriptions]
        composition_embedding = run_bert(compositions, 'matbert', self.lm, self.tokenizer)
        #structures = list(descriptions.values())
        structure_embedding = run_bert(structure_descriptions, 'matbert', self.lm, self.tokenizer)

        return composition_embedding, structure_embedding



    def query_composition(self, query_embedding, target_embedding, target_compositions, k=100):
        """_summary_

        Args:
            query_embedding (_type_): _description_
            target_embedding (_type_): _description_
            target_compositions (_type_): _description_
            k (int, optional): _description_. Defaults to 100.

        Returns:
            _type_: _description_
        """
    
        cos_sim = fastdist.vector_to_matrix_distance(query_embedding, target_embedding, fastdist.cosine,'cosine')
        #cos_sim = cosine_similarity(query_embedding,target_embedding)
        rank = len(cos_sim) - ss.rankdata(list(cos_sim),'ordinal')
        rank_data = pd.DataFrame({'similarity':cos_sim,'rank':rank,'composition_name':target_compositions})
        
        rank_data_topk = rank_data.sort_values('rank').iloc[:k]

        return list(rank_data_topk.index), rank_data_topk.reset_index()
    


    def rank_composition(self, query_data, output_data, query_composition):
        """_summary_

        Args:
            query_data (_type_): _description_
            output_data (_type_): _description_
            query_composition (_type_): _description_

        Returns:
            _type_: _description_
        """

        label = abs((query_data-output_data)/query_data)
        label = np.mean(np.array(label),1)
        rank = ss.rankdata(list(label),'ordinal')  
        rank_target = pd.DataFrame({'label':label,'rank':rank,
                                        'composition_name':query_composition})
        
        
        return rank_target
    

    def search_rank(self, cif_paths, k=100):

        descriptions = self.get_description(cif_paths)

        composition_embedding, structure_embedding = self.get_embedding(descriptions)
        compositions = [d[0] for d in descriptions]

        for i, comp in enumerate(compositions):

            candidate_idx, query_topk = self.query_composition(structure_embedding[i][0], self.structure_embeddings, self.structure_names, k=k)
            self.search_results[comp] = {'query_idx':candidate_idx,'query_topk':query_topk}

            query_output = []
            candidate_output = []

            #for idx in range(4):
    
            query_features = get_query_data(composition_embedding[i], structure_embedding[i])
            candidate_features = get_candidate_data(candidate_idx, self.composition_embeddings, self.structure_embeddings)

            x_query = {name: query_features[name].values for name in query_features.columns}
            x_candidate = {name: candidate_features[name].values for name in candidate_features.columns}

            query_output_temp = self.mmoe_model.predict(x_query)
            query_output.append(query_output_temp)
            candidate_output_temp = self.mmoe_model.predict(x_candidate)
            candidate_output.append(candidate_output_temp)

           # query_output = np.array(query_output).mean(0)
            #candidate_output = np.reshape(np.array(candidate_output),(4,100,5),'F').mean(0)

            rank_candidate = self.rank_composition(query_output[0], candidate_output[0], self.structure_names[candidate_idx])

            self.rank_results[comp] = rank_candidate.drop_duplicates('composition_name')

    

    


    


