#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k as lightfm_precision_at_k
from rs.utils import prefilter_items, FILTERED_ID
# Для работы с матрицами
from scipy.sparse import csr_matrix, coo_matrix

from rs.metrics import precision_at_k as metrics_precision_at_k


# In[ ]:


class LightFMRecommender:
    
        def __init__(self, 
                no_components=40,
                loss='bpr', # "logistic","bpr", "warp"
                learning_rate=0.01, 
                item_alpha=0.4,
                user_alpha=0.1, 
                random_state=42,
                k=5,
                n=15,
                max_sampled=100):
            
            self.user_item_matrix = None
            self.sparse_user_item = None
            self.user_feat_lightfm = None
            self.item_feat_lightfm = None
            self.users_ids_row = None
            self.items_ids_row = None
            self.filtered_id = FILTERED_ID
            self.id_to_itemid=self.id_to_userid=self.itemid_to_id=self.userid_to_id = None


            self.model = LightFM(no_components=no_components,
                    loss=loss,
                    learning_rate=learning_rate, 
                    item_alpha=item_alpha,
                    user_alpha=user_alpha, 
                    random_state=random_state,
                    k=k,
                    n=n,
                    max_sampled=max_sampled)
            
            
        def fit(self, data: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame, epochs=20, num_threads=20, verbose=True):
            
            self.user_item_matrix = self.prepare_matrix(data)
            self.sparse_user_item = csr_matrix(self.user_item_matrix).tocsr()
            
            self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

            self.users_ids_row, self.items_ids_row = self.__get_ids_row(data)
            self.user_feat_lightfm = self.__prepare_user_features(user_features)
            self.item_feat_lightfm = self.__prepare_item_features(item_features)
            
            self.model.fit((self.sparse_user_item > 0) * 1,  # user-item matrix из 0 и 1
                      sample_weight=coo_matrix(self.user_item_matrix),
                      user_features=csr_matrix(self.user_feat_lightfm.values).tocsr(),
                      item_features=csr_matrix(self.item_feat_lightfm.values).tocsr(),
                      epochs=epochs,
                      num_threads=num_threads,
                      verbose=verbose)

        @staticmethod
        def filter_data(data: pd.DataFrame, take_n_popular=5000, item_features=None):
            return prefilter_items(data, take_n_popular=take_n_popular, item_features=item_features)


        def predict(self):
            # модель возвращает меру/скор похожести между соответствующим пользователем и товаром
            return self.model.predict(user_ids=self.users_ids_row,
                                item_ids=self.items_ids_row,
                                user_features=csr_matrix(self.user_feat_lightfm.values).tocsr(),
                                item_features=csr_matrix(self.item_feat_lightfm.values).tocsr(),
                                num_threads=10)


        @staticmethod
        def prepare_matrix(data_train: pd.DataFrame):

            user_item_matrix = pd.pivot_table(data_train, 
                                      index='user_id', columns='item_id', 
                                      values='quantity', # Можно пробовать другие варианты
                                      aggfunc='count', 
                                      fill_value=0
                                     )

            user_item_matrix = user_item_matrix.astype(float) 

            return user_item_matrix


        def __get_ids_row(self, data_train_filtered: pd.DataFrame):
            # подготавливаемм id для юзеров и товаров в порядке пар user-item
            users_ids_row = data_train_filtered['user_id'].apply(lambda x: self.userid_to_id[x]).values.astype(int)
            items_ids_row = data_train_filtered['item_id'].apply(lambda x: self.itemid_to_id[x]).values.astype(int)

            return users_ids_row, items_ids_row


        def __prepare_user_features(self, user_features):

            user_feat = pd.DataFrame(self.user_item_matrix.index)
            user_feat = user_feat.merge(user_features, on='user_id', how='left')
            user_feat.set_index('user_id', inplace=True)
            user_feat_lightfm = pd.get_dummies(user_feat, columns=user_feat.columns.tolist())
            
            return user_feat_lightfm
        
        
        def __prepare_item_features(self, item_features):

            item_feat = pd.DataFrame(self.user_item_matrix.columns)
            item_feat = item_feat.merge(item_features, on='item_id', how='left')
            item_feat.set_index('item_id', inplace=True)
            item_feat_lightfm = pd.get_dummies(item_feat, columns=item_feat.columns.tolist())
            
            return item_feat_lightfm
            
        
        @staticmethod
        def prepare_dicts(user_item_matrix):
            """Подготавливает вспомогательные словари"""

            userids = user_item_matrix.index.values
            itemids = user_item_matrix.columns.values

            matrix_userids = np.arange(len(userids))
            matrix_itemids = np.arange(len(itemids))

            id_to_itemid = dict(zip(matrix_itemids, itemids))
            id_to_userid = dict(zip(matrix_userids, userids))

            itemid_to_id = dict(zip(itemids, matrix_itemids))
            userid_to_id = dict(zip(userids, matrix_userids))

            return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
        
        
        def get_user_embeddings(self):
            return self.model.get_user_representations(features=csr_matrix(self.user_feat_lightfm.values).tocsr())
        
        def get_item_embeddings(self):
            return self.model.get_item_representations(features=csr_matrix(self.item_feat_lightfm.values).tocsr())
        
        def train_precision_at_k(self, k=5):
            return lightfm_precision_at_k(self.model, self.sparse_user_item,
                                 user_features=csr_matrix(self.user_feat_lightfm.values).tocsr(),
                                 item_features=csr_matrix(self.item_feat_lightfm.values).tocsr(),
                                 k=5).mean()

        @staticmethod
        def precision_at_k(df_result_for_metrics: pd.DataFrame, k=5):
            precision = df_result_for_metrics.apply(lambda row: metrics_precision_at_k(row['item_id'], row['actual'], k),
                                                    axis=1).mean()
            return precision

