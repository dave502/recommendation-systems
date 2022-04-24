#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from rs.metrics import precision_at_k, recall_at_k
from rs.utils import prefilter_items, FILTERED_ID

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

import random

class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data: pd.DataFrame, take_n_popular=5000, item_features=None):
        
        data_train = prefilter_items(data, take_n_popular, item_features)
       
        user_item_matrix = pd.pivot_table(data_train, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', # Можно пробовать другие варианты
                                  aggfunc='count', 
                                  fill_value=0
                                 )

        user_item_matrix = user_item_matrix.astype(float) 
        
        return user_item_matrix
    
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
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads,
                                             use_gpu=False)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        
        recs = self.own_recommender.recommend(userid=self.id_to_userid[user], 
                                user_items=csr_matrix(self.user_item_matrix).tocsr(), 
                                N=N,
                                filter_already_liked_items=False, 
                                filter_items=[self.itemid_to_id[999999]], 
                                recalculate_user=True)

        res = [self.id_to_itemid[rec[0]] for rec in recs]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        N_users = N+1 # количество выбранных пользователей
        N_recs = 1 # количество рекомендованных товаров для каждого пользователя

        similar_users = pd.DataFrame(self.model.similar_users(self.id_to_userid[user], N_users), columns=['users', 'similarity'])
        similar_users.drop([0], axis=0, inplace=True)  # убрать первого пользователя - самого user

        recs = similar_users['users'].apply(lambda user_id: [self.id_to_itemid[rec[0]]  
                                                              for rec in self.model.recommend(userid=self.userid_to_id[user_id], 
                                                                user_items=csr_matrix(self.user_item_matrix).tocsr(),   # на вход user-item matrix
                                                                N=N_recs, 
                                                                filter_already_liked_items=False, 
                                                                filter_items=[self.itemid_to_id[FILTERED_ID]], 
                                                                recalculate_user=True)]).values.tolist()
        
        res = [item for rec in recs for item in rec]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    
    def get_similar_users_items(self, data, user, N=5):
        """Рекомендуем топ-N товаров, среди популярных у похожих юзеров"""
        # data передаётся в параметре функции, чтобы не хранить большой объём информации в атрибуте класса
        
        N_users = N+1 # количество выбранных пользователей
        N_recs = 100 # количество популярных товаров, из которых случайно выбирается N рекомендуемых товаров

        similar_users = self.model.similar_users(self.id_to_userid[user], N_users)
        similar_users_ids = [user[0] for user in similar_users]

        # популярные товары у похожих пользователей
        popular = data[data['user_id'].isin(similar_users_ids[1:])].groupby('item_id')['sales_value'].sum().reset_index()
        popular.sort_values('sales_value', ascending=False, inplace=True)

        # N случайных товаров из топ-N_recs товаров похожих пользователей
        res = popular.head(N_recs).sample(N).item_id

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res.tolist()
