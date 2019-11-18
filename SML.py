# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import time
import numpy as np

class model(object):
    
    def __init__(self,hidden_size, user_num, prd_num,neg_samples, lamda,gama):

        self.hidden_size = hidden_size
        self.user_num = user_num
        self.prd_num = prd_num
        self.init = 1 / (self.hidden_size ** 0.5)
        self.lamda =lamda
        self.gama=gama
        self.neg_samples =neg_samples

        print("SML.")

    def build_network(self):

        self.userid = tf.placeholder(tf.int32, [None], name="user_id")
        self.prdid = tf.placeholder(tf.int32, [None], name="prd_id")
        self.neg_prdid = tf.placeholder(tf.int32,[None],name='neg_prdid')
        #self.neg_userid = tf.placeholder(tf.int32,[None],name='neg_userid')
        
        #self.neg_prdid = tf.placeholder(tf.int32,[None,self.neg_samples],name='neg_prdid')
        #self.neg_userid = tf.placeholder(tf.int32,[None,self.neg_samples],name='neg_userid')
 

        U = tf.Variable(tf.random_normal([self.user_num, self.hidden_size], stddev=self.init), dtype=tf.float32)
        P = tf.Variable(tf.random_normal([self.prd_num, self.hidden_size], stddev=self.init), dtype=tf.float32)
        # B =tf.cast(self.user_neighbours,dtype=tf.float32)
        with tf.name_scope(name="B"):
            B= tf.Variable(np.array([1.0]*self.user_num), dtype=tf.float32,trainable=True)
            B1=tf.Variable(np.array([1.0]*self.prd_num),dtype=tf.float32,trainable=True)
        bias = tf.nn.embedding_lookup(B,self.userid) 
        user_embedding = tf.nn.embedding_lookup(U, self.userid)
        pbias= tf.nn.embedding_lookup(B1,self.prdid)
        prd_embedding = tf.nn.embedding_lookup(P, self.prdid)
        neg_prd_embedding = tf.nn.embedding_lookup(P, self.neg_prdid)

        #neg_prd_embedding = tf.nn.embedding_lookup(P,tf.reshape(self.neg_prdid,[-1]))
        #u_temp = tf.reshape(tf.tile(user_embedding,[1,self.neg_samples]),[-1,self.hidden_size] )
        #p_temp_neg = tf.reshape(neg_prd_embedding, [-1,self.hidden_size] )
        #p_temp = tf.reshape(tf.tile(prd_embedding,[1,self.neg_samples]),[-1,self.hidden_size] )

        self.pred_distance =  tf.reduce_sum(tf.square(user_embedding-prd_embedding),1) 
        self.pred_distance_neg =  tf.reduce_sum(tf.multiply( user_embedding- neg_prd_embedding ,user_embedding- neg_prd_embedding),1) 
        self.pred_distance_PN =  tf.reduce_sum(tf.multiply(prd_embedding-neg_prd_embedding,prd_embedding- neg_prd_embedding ),1) 
        #self.pred_distance_neg =tf.reduce_sum(tf.square(u_temp-p_temp_neg),1) #tf.reduce_mean(tf.reduce_sum(tf.square(distance),2),1)

        # a =tf.reduce_sum(tf.log(tf.exp(self.margin)+tf.exp(self.pred_distance-self.pred_distance_neg)))
        # b = tf.reduce_sum(tf.log(tf.exp(self.margin)+tf.exp(self.pred_distance- self.pred_distance_PN)))

        a = tf.maximum(self.pred_distance - self.pred_distance_neg + bias, 0)
        b = tf.maximum(self.pred_distance-self.pred_distance_PN+pbias,0)

         
        #whole model
        self.loss= tf.reduce_sum(a)+self.lamda*tf.reduce_sum(b)
        self.loss=self.loss-1*(self.gama*(tf.reduce_mean(bias) +tf.reduce_mean( pbias)))

        tf.add_to_collection('user_embedding',user_embedding)
        tf.add_to_collection('prd_embedding',prd_embedding)

        self.clip_U = tf.assign(U, tf.clip_by_norm(U, 1.0, axes=[1]))
        self.clip_P = tf.assign(P, tf.clip_by_norm(P, 1.0, axes=[1]))
        self.clip_B = tf.assign(B,tf.clip_by_value(B,0,1.0))
        self.clip_B1=tf.assign(B1,tf.clip_by_value(B1,0,1.0))
 
 
