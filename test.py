#-*- coding: utf-8 -*-
from __future__ import unicode_literals
#author: Ming
 

import os, time, pickle,sys
import numpy as np
import tensorflow as tf
import math
from data_helpers import TrainDataset,TestDataset
import data_helpers
import SML
import datetime
import time
from RankingMetrics import *
import json
# Data loading params
tf.flags.DEFINE_integer("n_class", 5, "Numbers of class")
tf.flags.DEFINE_string("dataset", 'aiv', "The dataset")
tf.flags.DEFINE_integer("checkpoints",0,'checkpoints')
# Model Hyperparameters
tf.flags.DEFINE_integer("hidden_size", 50, "hidden_size of rnn")
tf.flags.DEFINE_float("lr", 0.05, "Learning rate")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 800, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on test set after this many steps")
tf.flags.DEFINE_integer("verbose", 20, "Evaluate model on test set after this many steps")
tf.flags.DEFINE_boolean("debug",False, "debug")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("neg_numbers",500,"number of negative sampels")
tf.flags.DEFINE_float("margin",0.5,"margin")
tf.flags.DEFINE_string("model","SML","CML or LRML")
tf.flags.DEFINE_string("cuda","0","0 or 1")
tf.flags.DEFINE_integer("neg_samples",20,'number of negative smaples ')
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

trainset = TrainDataset('../../data/'+FLAGS.dataset+'/train.txt',FLAGS.model,FLAGS.debug )
testset = TestDataset('../../data/'+FLAGS.dataset+'/test.txt',FLAGS.debug)
valset = TestDataset('../../data/'+FLAGS.dataset+'/dev.txt',FLAGS.debug)

def get_dicts(uname,tr,te,val ):
    ids = list(set(tr))
    print ("The number of "+str( uname)+ "in train dataset is "+str( len(ids)))
    ids1 = list(set(te))
    print ("The number of ", uname, "in test dataset is ", len(ids1))
    ids2 = list(set(val))
    print ("The number of ", uname, "in val dataset is ", len(ids2))

    out =ids+ids1+ids2
    out=list(set(out))
    print ("Total number of ",uname, "is ",len(out))
    return out
pids = get_dicts("product",trainset.t_prd,testset.t_prd,valset.t_prd)
uids = get_dicts("user",trainset.t_user,testset.t_user,valset.t_user)

path_checkpoint_file ="../checkpoints/"+FLAGS.dataset+"/"+FLAGS.model+"/"#
print path_checkpoint_file,path_checkpoint_file+FLAGS.model+"-"+str(FLAGS.checkpoints)
# Load data
checkpoint_file = tf.train.latest_checkpoint(path_checkpoint_file)
print checkpoint_file


with open(path_checkpoint_file+"usrdict.txt", 'rb') as f:
    userdict = pickle.load(f)
with open(path_checkpoint_file+"prddict.txt", 'rb') as f:
    prddict = pickle.load(f)


testset.predict_data(userdict, prddict)

trainset.get_negative_sample(userdict,prddict,testset.test_user_item_matrix,
                             testset.test_item_user_matrix,FLAGS.neg_numbers)
# trainbatches = trainset.batch_iter(userdict, prddict, FLAGS.n_class, FLAGS.batch_size,
#                                  FLAGS.num_epochs,FLAGS.neg_samples)

p_users = trainset.neighbour_prds
ks = list(range(0,len(p_users)))
print len(ks)
print 'product',len(prddict.keys())
print 'user',len(userdict.keys())
vl = p_users
# Final_user={}
# for  i in range(len(ks)):
#     for j in range(i+1,len(ks)):
#         if len(list((set(vl[i])- set(vl[j]))))<1 and len(set(vl[i]))>1:
#             out=  set(vl[i])-set([5130]) 
#             print "item",i,j, out

            # ii=list(prddict.keys())[list(prddict.values()).index(i)]
            # jj=list(prddict.keys())[list(prddict.values()).index(j)]
            # uu= [list(userdict.keys())[list(userdict.values()).index(u)] for u in out ]
            # print "item", ii, jj, uu

uid=1
p=list(set(p_users[uid])-set([5130]))
u=[uid]*len(p)
print u,p
p=  list(prddict.values())
u = list(userdict.values())

graph = tf.Graph()
with graph.as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    with sess.as_default():
        print checkpoint_file
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        userid = graph.get_operation_by_name("user_id").outputs[0]
        productid = graph.get_operation_by_name("prd_id").outputs[0]
 

        U=tf.get_collection("user_embedding")[0]
        P=tf.get_collection('prd_embedding')[0]
        # uu,pp = sess.run([ U,P],feed_dict={userid:u,productid:p})
        pp =sess.run([P],feed_dict={productid:p})
        with open(path_checkpoint_file + "/ItemMatrix.txt", 'wb') as f:
            pickle.dump(pp, f)
    

         

