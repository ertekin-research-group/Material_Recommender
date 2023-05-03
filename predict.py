import torch
from mat_rec import mat_recommender, model
import numpy as np
import pandas as pd
import glob
import json
import os 

def main():
    
    with open('train_config.json', 'r') as f:
        train_config = json.load(f)
    
    test_data_dir = train_config['test_data_dir']
    targets = train_config['targets']
    test_data = pd.read_csv(test_data_dir+'test_data.csv')
    id = test_data.id.values
    recommender = mat_recommender.Recommender(bert_model_dir = 'matbert_model_files/matbert-base-uncased')
    structure_dir = (row[1].composition, test_data_dir+row[1].id+'.cif' for row in test_data.iterrows())

    if os.path.isfile(test_data_dir+'test_embeddings.npy'):
        embedding_features = np.load(test_data_dir+'test_embeddings.npy')

    else:
        descriptions = recommender.get_description(structure_dir)
        composition_embedding,structure_embedding = recommender.get_embedding(descriptions)
        embedding_features = np.concatenate([np.concatenate(composition_embedding),np.concatenate(structure_embedding)],1)
        np.save(train_data_dir+'test_embeddings.npy',embedding_features)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    #data_use = data_use.reset_index()

    embedding_features = pd.DataFrame(embedding_features)
    embedding_features.columns = range(embedding_features.columns.size)
    embedding_features.columns = 'feature_'+embedding_features.columns.astype(str)
    property_model = model.mat_MMoE()
    property_model.model_config['n_task'] = len(targets)
    property_model.model_config['n_expert'] = train_config['n_expert']
    property_model.load_model('saved_model/model.pth',embedding_features.columns)
    test_data = test_data[targets].values


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_test = {name: embedding_features[name].values for name in embedding_features.columns}

    pred = property_model.predict(x_test)

    pred_df = pd.DataFrame()
    pred_df['id'] = id
    pred_df[targets] = pred

    pred_df.to_csv('output/pred_result.csv')

if __name__ == "__main__":
    main()