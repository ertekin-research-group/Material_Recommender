import glob
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from mat_rec.multitask.utils.data import DataGenerator, PredictDataset
from mat_rec.multitask.models.MMOE import MMOE
from mat_rec.multitask.basic.features import DenseFeature, SparseFeature
from mat_rec.multitask.trainers import MTLTrainer
from torch.utils.data import Dataset, DataLoader

import torch
from sklearn.metrics import r2_score

from sklearn.utils import shuffle


class mat_MMoE():

    def __init__(self, config=None, model_config=None):

        if config:
            self.config = config
        else:
            self.config = {'learning_rate' : 1e-3,
                           'epoch' : 500, #10
                           'weight_decay' : 1e-5,
                           'batch_size': 64,
                           'num_workers':1
                           }
        if model_config:
            self.model_config = model_config
        else:
            self.model_config = {'n_task' : 5,
                                 'n_expert' : 8,
                                 'task_type':'regression',
                                 'expert_dim':[128,64,32],
                                 'tower_dim':[32,16]
                                 }
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'



    def init_features(self,dense_feature_names, sparse_feature_names=None):

        self.features = [DenseFeature(feat) for feat in dense_feature_names]

        if sparse_feature_names:
            self.features+=[SparseFeature(feat,2,embed_dim=4) for feat in sparse_feature_names]

        self.features = list(self.features)

        

    def load_model(self, model_dir, dense_feature_names, sparse_feature_names=None):

        self.init_features(dense_feature_names, sparse_feature_names)

        self.model = MMOE(self.features, [self.model_config['task_type']]*self.model_config['n_task'], self.model_config['n_expert'],
                      expert_params={"dims": self.model_config['expert_dim']}, tower_params_list=[{"dims": self.model_config['tower_dim']}]*self.model_config['n_expert'])
        
        self.model.load_state_dict(torch.load(model_dir,map_location=self.device))



    def get_kfold_cv(self, names, k, split_by_name=None):

        #sample_names = self.intens_mtx.index.values
        sample_names_shuffled = shuffle(names,random_state=39)
        sample_names_kfold_test = np.array_split(sample_names_shuffled,k)
        cv_test = sample_names_kfold_test

        return cv_test


    def  train(self, x_train, y_train, x_test, y_test, multi_task=True, model_path='./saved_model'):
        """_summary_

        Args:
            x_train (_type_): _description_
            y_train (_type_): _description_
            x_test (_type_): _description_
            y_test (_type_): _description_
            multi_task (bool, optional): _description_. Defaults to True.
            n_task (int, optional): _description_. Defaults to 5.
            n_expert (int, optional): _description_. Defaults to 8.

        Returns:
            _type_: _description_
        """

        parameters = self.config

        #features = list([DenseFeature(col) for col in x_train.columns[:-4]] + [SparseFeature(col,2,embed_dim=4) for col in x_train.columns[-4:]])

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, train_size=0.9, random_state=19)

        x_train, y_train = {name: x_train[name].values for name in x_train.columns}, y_train
        x_val, y_val = {name: x_val[name].values for name in x_val.columns}, y_val
        x_test, y_test = {name: x_test[name].values for name in x_test.columns}, y_test
        dg = DataGenerator(x_train, y_train)

        train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=parameters['batch_size'])
        
        if multi_task:
            self.model = MMOE(self.features, [self.model_config['task_type']]*self.model_config['n_task'], self.model_config['n_expert'],
                    expert_params={"dims": self.model_config['expert_dim']},  tower_params_list=[{"dims": self.model_config['tower_dim']}]*self.model_config['n_expert'])

            mtl_trainer = MTLTrainer(self.model, task_types=[self.model_config['task_type']]*self.model_config['n_task'], optimizer_params={"lr": parameters['learning_rate'], "weight_decay": parameters['weight_decay']},
            n_epoch=parameters['epoch'], earlystop_patience=500, device=self.device, model_path=model_path)

        else:
            self.model = MMOE(self.features, [self.model_config['task_type']], 1,
            expert_params={"dims": self.model_config['expert_dim']}, tower_params_list=[{"dims": self.model_config['tower_dim']}])

            mtl_trainer = MTLTrainer(self.model, task_types=[self.model_config['task_type']], optimizer_params={"lr": parameters['learning_rate'], "weight_decay": parameters['weight_decay']},
            n_epoch=parameters['epoch'], earlystop_patience=500, device=self.device,model_path=model_path)


        mtl_trainer.fit(train_dataloader, val_dataloader)
        mtl_trainer.evaluate(mtl_trainer.model, test_dataloader)

        predictions = self.evaluate(test_dataloader)
        
        print('r2 score :{}'.format(r2_score(y_test,predictions)))

        return mtl_trainer, [y_test,predictions]


    def evaluate(self, data_loader):

        self.model.eval()
        targets, predicts = list(), list()

        with torch.no_grad():
            tk0 = tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, ys) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
                ys = ys.to(self.device)
                y_preds = self.model(x_dict)
                targets.extend(ys.tolist())
                predicts.extend(y_preds.tolist())
        targets, predicts = np.array(targets), np.array(predicts)
        #scores = [self.evaluate_fns[i](targets[:, i], predicts[:, i]) for i in range(self.n_task)]
        return predicts


    def predict(self, data):

        dataset = PredictDataset(data)
        data_loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])

        self.model.eval()
        predicts =  list()
        with torch.no_grad():
            tk0 = tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
                y_preds = self.model(x_dict)
                predicts.extend(y_preds.tolist())
        predicts = np.array(predicts)
        #scores = [self.evaluate_fns[i](targets[:, i], predicts[:, i]) for i in range(self.n_task)]
        return predicts

