import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def dcg(rank, rank_q, top_k):

    rel = (rank<=top_k).astype(int)
    dcg = 0
    for i in range(len(rank)):
        if np.log2(1+rank_q[i]) != 0:
            dcg += rel[i]/np.log2(1+rank_q[i])

    return dcg

def get_candidate_data(candidate_idx, composition_embeddings, structure_embeddings, temp_idx=3):    

    query_bert_features = structure_embeddings[candidate_idx]
    query_composition_features = composition_embeddings[candidate_idx]
    query_temp_features = []

    for i in candidate_idx:

        temp = np.array([[0,0,0,0]])
        temp[0,temp_idx] = 1
        query_temp_features.append(temp)

    candidate_features = pd.concat([pd.DataFrame(query_bert_features),
                                    pd.DataFrame(query_composition_features),
                                    pd.DataFrame(np.concatenate(query_temp_features))],axis=1)
    
    candidate_features.columns = range(candidate_features.columns.size)
    candidate_features.columns = 'feature_'+candidate_features.columns.astype(str)

    return candidate_features

def get_query_data(composition_embedding, structure_embedding, temp_idx = 3):

    temp = np.array([[0,0,0,0]])
    temp[0,temp_idx] = 1

    query_features = pd.concat([pd.DataFrame(structure_embedding),
                                pd.DataFrame(composition_embedding),
                                pd.DataFrame(temp)],axis=1)
    
    query_features.columns = range(query_features.columns.size)
    query_features.columns = 'feature_'+query_features.columns.astype(str)

    return query_features


def get_data(data_name):

    if data_name == 'fingerprint':
        robo_data=pd.read_pickle('robo_descriptions_with_fingerprint.pkl')
        robo_data['embeddings'] = robo_data['fingerprints']
    elif data_name == 'matscibert':
        robo_data=pd.read_pickle('robo_descriptions_with_embed_noprop.pkl')
    elif data_name == 'matbert':
        robo_data=pd.read_pickle('matbert_robo_descriptions_with_embed_noprop.pkl')

    property_data = pd.read_pickle('thermo_combined_label_data_host_temperature_ucsb+estm.pkl')
    property_data = property_data[property_data.count(1)>4]

    idx_use = np.array([i for i in range(len(property_data.host_material)) if property_data.host_material.values[i] in robo_data['composition_name'].values])

    data = robo_data.set_index('composition_name')
    selected_data = []
    selected_property = []
    temp_data = []

    for i in idx_use:
        comp_temp = property_data.index[i].split('_')
        host = property_data.iloc[i]['host_material']
        selected_data.append(data.loc[host].embeddings[0].reshape(1,-1))
        selected_property.append(property_data.iloc[i])
        if comp_temp[1] == '300':
            temp_data.append([1,0,0,0])
        elif comp_temp[1] == '600':
            temp_data.append([0,1,0,0])
        elif comp_temp[1] == '900':
            temp_data.append([0,0,1,0])
        else:
            temp_data.append([0,0,0,1])

    thermo_data = pd.DataFrame(selected_property)
    bert_feature = np.concatenate(selected_data,0)
    composition_feature = np.concatenate(thermo_data['embeddings'])
    temp_feature = np.stack(temp_data,0)

    return thermo_data, bert_feature, composition_feature, temp_feature



