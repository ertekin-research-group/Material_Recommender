
import torch
from mat_rec import mat_recommender, model
import numpy as np
import pandas as pd
import glob
import os
import json
import warnings
warnings.filterwarnings("ignore")

def main():
    with open('train_config.json', 'r') as f:
        train_config = json.load(f)
    
    train_data_dir = train_config['train_data_dir']
    targets = train_config['targets']
    composition_only = json.loads(train_config['composition_only'].lower())
    train_data = pd.read_csv(train_data_dir+'train_data.csv')
    
    if composition_only:

        if os.path.isfile(train_data_dir+'train_embeddings_composition.npy'):
            embedding_features = np.load(train_data_dir+'train_embeddings_composition.npy')
        else:
            descriptions = [(row[1].composition, row[1].composition) for row in train_data.iterrows()]
            composition_embedding,structure_embedding = recommender.get_embedding(descriptions)

    else:
        structure_dir = [(row[1].composition,train_data_dir+row[1].id+'.cif') for row in train_data.iterrows()]

        if os.path.isfile(train_data_dir+'train_embeddings.npy'):
            embedding_features = np.load(train_data_dir+'train_embeddings.npy')
        else:
            descriptions = recommender.get_description(structure_dir)
            composition_embedding,structure_embedding = recommender.get_embedding(descriptions)
            embedding_features = np.concatenate([np.concatenate(composition_embedding),np.concatenate(structure_embedding)],1)
            np.save(train_data_dir+'train_embeddings.npy',embedding_features)



    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    property_model = model.mat_MMoE()

    #data_use = data_use.reset_index()

    embedding_features = pd.DataFrame(embedding_features)
    embedding_features.columns = range(embedding_features.columns.size)
    embedding_features.columns = 'feature_'+embedding_features.columns.astype(str)
    property_model.init_features(embedding_features.columns)
    train_data = train_data[targets].values


    property_model.model_config['n_task'] = len(targets)
    property_model.model_config['n_expert'] = train_config['n_expert']
    property_model.config['epoch'] = train_config['epoch']
    property_model.config['learning_rate'] = train_config['learning_rate']
    property_model.config['batch_size'] = train_config['batch_size']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
    if len(targets) >1:
        #multitask
        x_train, y_train, x_test, y_test = embedding_features, train_data, embedding_features, train_data
        mtl_trainer, output = property_model.train(x_train, y_train, x_test, y_test)

        np.save('output/multi_task_prediction_results.npy', output)
        np.save('output/multi_task_loss.npy', np.array([mtl_trainer.train_loss,mtl_trainer.val_loss],dtype=object))

    else:
        #single task
        x_train, y_train, x_test, y_test = embedding_features, train_data, embedding_features, train_data
        mtl_trainer, output = property_model.train(x_train, y_train.reshape(-1,1), x_test, y_test.reshape(-1,1), multi_task=False)
        all_target_results.append(output)
        np.save('output/single_task_prediction_results.npy', output)
        np.save('output/single_task_loss.npy', np.array([mtl_trainer.train_loss,mtl_trainer.val_loss],dtype=object))


if __name__ == "__main__":
    main()