from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.platform import gfile

import collections
import numpy as np
import tensorflow as tf
import os.path
import csv
import time

Dataset = collections.namedtuple('Dataset', ['data', 'label'])

# Hyper-Parameters
learning_rate = 0.005
training_epochs = 500 
batch_size = 50  
display_step = 1                                                                              
                
# Network Parameters
n_hidden_1 = 250	# 1st layer neurons 
n_hidden_2 = 250 	# 2nd layer neurons
n_input = 1500 		# input cound
n_classes = 20		# number of output classes

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def dense_to_prob(labels_dense, num_classes): 

  problabels = []
  labels = load_csv_general(target_dtype=np.float ,filename="data/probs.csv")

  labels_dense = labels_dense.astype(np.int64)
  for labelno in labels_dense:
  	labelno = labelno - 1   
        problabels.append(labels[labelno])
  return np.array(problabels) 

def dense_to_motion_labels(labels_dense, num_classes): 

  motionlabels = []

  labels = load_csv_general(target_dtype=np.float ,filename="data/motionlabels.csv")
     

  labels_dense = labels_dense.astype(np.int64)
  for labelno in labels_dense:
  	labelno = labelno - 1   
        motionlabels.append(labels[labelno])
  return np.array(motionlabels)

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""  
  labels_dense = labels_dense - 1
  labels_dense =  labels_dense.astype(int)
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot      

def load_csv_general(target_dtype, filename):
  data = []
 
  if os.path.isfile(filename):	
	with gfile.Open(filename) as csv_file:
	    data_file = csv.reader(csv_file)  
	    for ir in data_file:
		data.append(ir)
  return data

# Loading the training csv files 
def load_csv(target_dtype, label_column=0, trainfiles=True):

  data = []  
  filename = "data/training.csv"

  if not trainfiles:  
     filename = "data/test.csv"

  print ("CSV files are being loaded from : " + filename)
 
  if os.path.isfile(filename):	
	with gfile.Open(filename) as csv_file:
	    data_file = csv.reader(csv_file)  
	    for ir in data_file:
		data.append(ir)
  print ('Returning ' + str(len(data)) + " number of records")

  return data

# Create model
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    y = tf.nn.softmax(out_layer)
    return y

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],seed=0)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],seed=0)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes],seed=0))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1],seed=0)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2],seed=0)),
    'out': tf.Variable(tf.random_normal([n_classes],seed=0))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    print ('Loading training CSV files')
    wtrndata = load_csv(target_dtype=np.float, label_column=0, trainfiles=True)	
    wifidata = np.array(wtrndata, np.float32)

    n_records = len(wtrndata)
    print("Done loading " + str(n_records) + " number of records")	

    seed = 0
    # Training cycle
    start = time.time()
    for epoch in range(training_epochs):
	avg_cost = 0.
	total_batch = int(n_records/batch_size)
        seed = seed + 1
	# Loop over all batches
	for i in range(total_batch):
    
            np.random.seed(seed*1000 + i)

	    random_set = wifidata[np.random.choice(wifidata.shape[0],batch_size, replace=False)]
	    dense_labels = random_set[:,0]
            batch_y = dense_to_prob(dense_labels,n_classes)
	    batch_x = np.delete(random_set, 0, axis=1)
	    
	    # Run optimization op (backprop) and cost op (to get loss value)  
	    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
	    # Compute average loss
	    avg_cost += c / total_batch           
	# Display logs per epoch step
	if epoch % display_step == 0:
	    print("Epoch:", '%04d' % (epoch+1), "cost=", \
		"{:.9f}".format(avg_cost))
    end = time.time()
    print("Optimization Finished!")
    print("It took ", str(end - start), " seconds to train the network")

    ind_max = tf.argmax(pred,1)
    flat_y = tf.reshape(y, [-1])
    flat_ind_max = ind_max + tf.cast(tf.range(tf.shape(y)[0]) * tf.shape(y)[1], tf.int64)
    y_ = tf.gather(flat_y, flat_ind_max)
    accuracy = tf.reduce_mean(tf.cast(y_, "float"))   
    
    wtstdata = load_csv(target_dtype=np.float, label_column=0, trainfiles=False)	 
    set_tst = np.array(wtstdata, np.float32)	
    
    dense_labels_tst = set_tst[:,0]
    batch_ytst = dense_to_motion_labels(dense_labels_tst,n_classes)  
    
    batch_xtst = np.delete(set_tst, 0, axis=1)         
    transprobsmat = load_csv_general(target_dtype=np.float ,filename="data/motionmodel.csv")
    transprobs = np.array(transprobsmat)
    transprobs = transprobs.astype(np.float64)
    prev_region = 0    
    score = 0;
    
    resultlines = np.empty((0,5),np.float)

    start_train = time.time()
    for j in range(0,200):
        xts = batch_xtst[j,:]
        xtst = np.reshape(xts,(1,1500))
        yts = batch_ytst[j,:]
        ytst = np.reshape(yts, (1,20))   
        acc,preds = sess.run([accuracy, pred], feed_dict={x: xtst, y: batch_ytst})
        motionvec = np.reshape(transprobs[prev_region,:], (1,20))
        finalanswer_vector = np.multiply(preds, motionvec)
        ytst = ytst.astype(np.int64)          
        finalanswer = tf.argmax(finalanswer_vector,1).eval()[0]
        prev_region = finalanswer
        truepos = tf.argmax(ytst,1).eval()[0]
        flat = preds.flatten()
        flat.sort()
        maxprob = flat[-1]
        secmaxprob = flat[-2]
        tempflat = preds.flatten()
        secindex = 0
        cou = 0
        for elem in np.nditer(tempflat):
            cou = cou + 1
            if elem == secmaxprob:
               secindex = cou 
        templine = np.concatenate(([dense_labels_tst[j]],[finalanswer+1],[maxprob],[secmaxprob],[secindex]),axis=1)
        resultlines = np.vstack([resultlines,templine])
        print(resultlines.shape)
        if finalanswer == truepos:  
           score = score + 1
    trueaccuracy = score/200     
    end_train = time.time()
    print("It took ", str(end_train - start_train), " seconds to test the network")

    print('Accuracy: ',trueaccuracy)  

    np.savetxt("primsecond.csv", resultlines, delimiter=",")

