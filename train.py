#-*- coding: utf-8 -*-
from __future__ import unicode_literals
#author: Ming Li

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

# Data loading params
tf.flags.DEFINE_integer("n_class", 5, "Numbers of class")
tf.flags.DEFINE_string("dataset", 'aiv', "The dataset")

# Model Hyperparameters
tf.flags.DEFINE_integer("hidden_size", 100, "hidden_size of rnn")
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
tf.flags.DEFINE_string("dist","L2","L2 or L1")
tf.flags.DEFINE_string("model","SML","CML or SML")
tf.flags.DEFINE_string("cuda","0","0 or 1")
tf.flags.DEFINE_integer("neg_samples",20,'number of negative smaples ')
tf.flags.DEFINE_float('lamda',0.01,'lamda')
tf.flags.DEFINE_float('gama',10,'gama')
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
# Load data
print("Loading data...")


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
print ('pids',len(pids),'uids',len(uids))
def get_user_prd_dict(uids,pids):
    userdict, prddict = dict(), dict()
    useridx, prdidx = 0, 0
    for u in uids:
        if u not in userdict:
            userdict[u] = useridx
            useridx += 1
    for p in pids:
        if p not in prddict:
            prddict[p] = prdidx
            prdidx += 1
    return userdict, prddict

userdict, prddict = get_user_prd_dict(uids,pids)
testset.predict_data(userdict, prddict)
valset.predict_data(userdict,prddict)
trainset.get_negative_sample(userdict,prddict,testset.test_user_item_matrix,\
                             testset.test_item_user_matrix,FLAGS.neg_numbers)
trainbatches = trainset.batch_iter(userdict, prddict, FLAGS.n_class, FLAGS.batch_size,
                                 FLAGS.num_epochs,FLAGS.neg_samples)


print("Loading data finished...")

user_num = len(userdict.keys()) 
prd_num = len(prddict.keys()) 
print(user_num,type(user_num))


def get_model():
    model=None
    if FLAGS.model == "SML":
        model=SML.model(
            hidden_size = FLAGS.hidden_size,
            user_num = len(userdict.keys()),
            prd_num = len(prddict.keys()),
            lamda = FLAGS.lamda,
            gama=FLAGS.gama,
            neg_samples = FLAGS.neg_samples,
            )
      
 
def get_feed_dict(model,data,type='train'):

    feed_dict={}
    if type=='train':
    
        u, p, y,neg_prdid,neg_userid=  list(data[0]),list(data[1]),list(data[2]),list(data[3]),list(data[4]) 
        
        if FLAGS.model =="FML" or (FLAGS.model =='MLP') or (FLAGS.model =='NeuMF'):
            u_list =list(set(u))
            ps=list(p)
            ys=[1]*len(u)
            us=list(u)
            for i in list(u):
                for  jj in range(FLAGS.inter_neg):
                    us.append(i)
                    ps.append(trainset.neg_prdids[i][jj])
                    ys.append(0)
            feed_dict = {
                model.userid: us,
                model.prdid: ps,
                model.y: ys
             }
        elif FLAGS.model =="SML":
            feed_dict = {
                model.userid: u,
                model.prdid: p,
                model.neg_prdid: neg_prdid,
             }

    elif type== 'test':
        u,p=zip(*data)
        list_model =['SML','BPR','MLP','NeuMF','TransMF']
        if FLAGS.model  in list_model:
            feed_dict = {
                model.userid: u,
                model.prdid: p,
             }
        elif FLAGS.model == "CML":
             feed_dict = {
                model.userid: u,
                model.prdid: p,
                model.keep_rate:1.0
             }
         
    return feed_dict


def m_evaluation(t_test):
    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    hit_at_5 =[]
    hit_at_10=[]

    top=[5,10]
    for u in t_test.test_users:
        userids = []
        user_neg_prds = trainset.neg_prdids[u]# trainset.neg_userids[i] for negative prds
        user_neg_prds = list(user_neg_prds) + list(t_test.test_user_item_matrix[u])
     
        prdids = []
        for j in user_neg_prds:
            prdids.append(j)
            userids.append(u)
          
        data=zip(userids,prdids)
        scores = predict(data)
         
        neg_item_index = list(zip(prdids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1])#ascending order
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:top[0]]
        pred_ratings_10[u] = pred_ratings[u][:top[1]]

        p_5, r_5, ndcg_5, hit_5= precision_recall_ndcg_at_k(top[0], pred_ratings_5[u], t_test.test_user_item_matrix[u])
        p_at_5.append(p_5)
        hit_at_5.append(hit_5)
        ndcg_at_5.append(ndcg_5)
        p_10, r_10, ndcg_10,hit_10 = precision_recall_ndcg_at_k(top[1], pred_ratings_10[u], t_test.test_user_item_matrix[u])
        p_at_10.append(p_10)
        hit_at_10.append(hit_10)
        ndcg_at_10.append(ndcg_10)

    

    result=[round(np.sum(hit_at_5)*1.0/t_test.data_size,4),round(np.sum(hit_at_10)*1.0/t_test.data_size,4),\
            round(np.mean(ndcg_at_5),4), round(np.mean(ndcg_at_10),4),\
           round(np.mean(p_at_5),4),round(np.mean(p_at_10),4)]

    str_result=["hit-"+str(top[0])+"","hit-"+str(top[1])+"", \
                "ndcg@"+str(top[0])+"","ndcg@"+str(top[1])+"", \
                "p@"+str(top[0])+"","p@"+str(top[1])+""]
    return str_result,result

with tf.Graph().as_default():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda
    session_config = tf.ConfigProto()
    # session_config.gpu_options.per_process_gpu_memory_fraction = 0.65
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    with sess.as_default():
        model = get_model() 
        model.build_network()
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        optimizer = tf.train.AdagradOptimizer(FLAGS.lr).minimize(model.loss,global_step=global_step) 
        # Save dict
        timestamp = str(int(time.time()))
        checkpoint_dir = os.path.abspath("./checkpoints/"+FLAGS.dataset+"/"+FLAGS.model)
        checkpoint_prefix = os.path.join(checkpoint_dir, FLAGS.model)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        with open(checkpoint_dir + "/usrdict.txt", 'wb') as f:
            pickle.dump(userdict, f)
        with open(checkpoint_dir + "/prddict.txt", 'wb') as f:
            pickle.dump(prddict, f)

        sess.run(tf.global_variables_initializer())

        # Training loop. For each batch...
        def predict(data):
            feed_dict=get_feed_dict(model,data,'test')
            pred= sess.run([model.pred_distance], feed_dict=feed_dict )[0] #pred_distance 1*B
            return pred


        predict_round=0 
        best_result=[0]*6
        final_test_result=[[0]*6]*6
        best_round=[0]*6

        for tr_batch in trainbatches:
            feed_dict=get_feed_dict(model,tr_batch,"train")
            start_time = time.time()
            if FLAGS.model=='SML':
                _,step, loss,_,_,b,b1= sess.run([optimizer,global_step,model.loss,model.clip_U,model.clip_P,model.clip_B,model.clip_B1],feed_dict)
            if math.isnan(loss):
                print("loss =NAN")
                sys.exit()
            if step % FLAGS.verbose ==0:
                print("time={}: step {}, loss {:g}".format(time.time() - start_time , step, loss))
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                predict_round += 1
                print("\nEvaluation round %d:" % (predict_round))
                                
                print('=======val=========')
                str_result,results1=m_evaluation(valset)
                print(str_result)
                print(results1)
                print('=======test=========')
                _,results2=m_evaluation(testset)
                print(results2)
                for i in range(6):
                    if results1[i] >best_result[i]:
                        best_result[i]=results1[i]
                        final_test_result[i]=results2
                        best_round[i]=predict_round
                        '''
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}={}\n".format(str_result[i],current_step))
                        # with open( "./"+FLAGS.dataset+"_user_margin.txt", 'wb') as f:
                        #     pickle.dump(b, f)
                        # with open("./"+FLAGS.dataset+"_item_margin.txt", 'wb') as f:
                        #     pickle.dump(b1, f)
                        # with open("./"+FLAGS.dataset+"_user_neighbour.txt",'wb') as f:
                        #      pickle.dump(user_neighbour_numbers,f)
                        # with open("./"+FLAGS.dataset+"_item_neighbour.txt",'wb') as f:
                        #      pickle.dump(item_neighbour_numbers,f)
                        '''
                      
                print('===============best_result=====')
                for i in range(6):
                    print(final_test_result[i])
                print(best_round)
  

                    

 

        
 

        






