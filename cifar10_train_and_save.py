# NAI

# import dependencies
print "Import Dependencies..."
from matplotlib import pyplot
import numpy as np 
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe 
from caffe2.python import core,model_helper,net_drawer,optimizer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter

##################################################################################
# MAIN

print "Entering Main..."

##################################################################################
# Gather Inputs
train_lmdb = "./train_lmdb"
predict_net_out = "predict_net.pb" # Note: these are in PWD
init_net_out = "init_net.pb"
training_iters = 1000

# Make sure the training lmdb exists
if not os.path.exists(train_lmdb):
	print "ERROR: train lmdb NOT found"
	exit()

##################################################################################
# Create model helper for use in this script

# specify that input data is stored in NCHW storage order
arg_scope = {"order":"NCHW"}

# create the model object that will be used for the train net
# This model object contains the network definition and the parameter storage
train_model = model_helper.ModelHelper(name="train_model", arg_scope=arg_scope)

##################################################################################
#### Step 1: Add Input Data

# Go read the data from the lmdb and format it properly
# Since the images are stored as 8-bit ints, we will read them as such
# We are using the TensorProtosDBInput because the lmdbs were created with a TensorProtos object
data_uint8, label = train_model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=50, db=train_lmdb, db_type='lmdb')
# cast the 8-bit data to floats
data = train_model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
# scale data from [0,255] -> [0,1]
data = train_model.Scale(data,data,scale=float(1./256))
# enforce a stopgradient because we do not need the gradient of the data for the backward pass
data = train_model.StopGradient(data,data)

##################################################################################
#### Step 2: Add the model definition the the model object
# This is where we specify the network architecture from 'conv1' -> 'softmax'
# [Arch: data->conv->pool->...->fc->relu->softmax]

# Ouput size as fxn of input width (W), kernel size (K), padding (P), and stride (S)
# This can be used for calculating OFM size of conv and pooling layers
# O = ( (W-K+2P) / S ) + 1

def Add_Original_CIFAR10_Model(model, data):
    # data size = 3x101x101
    conv1 = brew.conv(model, data, 'conv1', dim_in=3, dim_out=32, kernel=5, stride=1, pad=2)
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=3, stride=2)
    relu1 = brew.relu(model, pool1, 'relu1')
    # data size = 32x50x50
    conv2 = brew.conv(model, relu1, 'conv2', dim_in=32, dim_out=32, kernel=5, stride=1, pad=2)
    relu2 = brew.relu(model, conv2, 'relu2')
    pool2 = brew.average_pool(model, relu2, 'pool2', kernel=3, stride=2)
    # data size = 32x25x25
    conv3 = brew.conv(model, pool2, 'conv3', dim_in=32, dim_out=64, kernel=5, stride=1, pad=2)
    relu3 = brew.relu(model, conv3, 'relu3')
    pool3 = brew.average_pool(model, relu3, 'pool3', kernel=3, stride=2)
    # data size = 64x12x12
    fc1 = brew.fc(model, pool3, 'fc1', dim_in=64*11*11, dim_out=64)
    fc2 = brew.fc(model, fc1, 'fc2', dim_in=64, dim_out=2)
    softmax = brew.softmax(model,fc2, 'softmax')

def Add_Custom_Model(model, data):
    # data size = 64x101x101
    conv1 = brew.conv(model, data, 'conv1', dim_in=3, dim_out=64, kernel=5, stride=1, pad=2)
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=3, stride=2)
    relu1 = brew.relu(model, pool1, 'relu1')
    # data size = 64x50x50
    conv2 = brew.conv(model, relu1, 'conv2', dim_in=64, dim_out=64, kernel=5, stride=1, pad=2)
    relu2 = brew.relu(model, conv2, 'relu2')
    pool2 = brew.average_pool(model, relu2, 'pool2', kernel=3, stride=2)
    # data size = 64x25x25
    conv3 = brew.conv(model, pool2, 'conv3', dim_in=64, dim_out=128, kernel=5, stride=1, pad=2)
    relu3 = brew.relu(model, conv3, 'relu3')
    pool3 = brew.average_pool(model, relu3, 'pool3', kernel=3, stride=2)
    # data size = 128x11x11
    conv4 = brew.conv(model, pool3, 'conv4', dim_in=128, dim_out=128, kernel=5, stride=1, pad=2)
    relu4 = brew.relu(model, conv4, 'relu4')
    pool4 = brew.average_pool(model, relu4, 'pool4', kernel=3, stride=2)
    # data size = 128x5x5
    #fc1 = brew.fc(model, pool4, 'fc1', dim_in=128*5*5, dim_out=128)
    fc1 = brew.fc(model, pool4, 'fc1', dim_in=128*4*4, dim_out=128)
    fc2 = brew.fc(model, fc1, 'fc2', dim_in=128, dim_out=2)
    softmax = brew.softmax(model,fc2, 'softmax')

#softmax=Add_Original_CIFAR10_Model(train_model, data)
softmax=Add_Custom_Model(train_model, data)

##################################################################################
#### Step 3: Add training operators to the model
# TODO: use the optimizer class here instead of doing sgd by hand

xent = train_model.LabelCrossEntropy(['softmax', 'label'], 'xent')
loss = train_model.AveragedLoss(xent, 'loss')
brew.accuracy(train_model, ['softmax', 'label'], 'accuracy')
train_model.AddGradientOperators([loss])

optimizer.build_sgd(train_model,base_learning_rate=0.01, policy="step", stepsize=1, gamma=0.999)

##################################################################################
#### Run the training procedure

# run the param init network once
workspace.RunNetOnce(train_model.param_init_net)
# create the network
workspace.CreateNet(train_model.net, overwrite=True)
# Set the total number of iterations and track the accuracy and loss
total_iters = training_iters
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
# Manually run the network for the specified amount of iterations
for i in range(total_iters):
	workspace.RunNet(train_model.net)
	accuracy[i] = workspace.FetchBlob('accuracy')
	loss[i] = workspace.FetchBlob('loss')
	print "Iter: {}, loss: {}, accuracy: {}".format(i, loss[i], accuracy[i])

# After execution is done lets plot the values
pyplot.plot(loss,'b', label='loss')
pyplot.plot(accuracy,'r', label='accuracy')
pyplot.legend(loc='upper right')
pyplot.show()

##################################################################################
#### Save the trained model for testing later

# save as two protobuf files (predict_net.pb and init_net.pb)
# predict_net.pb defines the architecture of the network
# init_net.pb defines the network params/weights
print "Saving the trained model to predict/init.pb files"
deploy_model = model_helper.ModelHelper(name="cifar10_deploy", arg_scope=arg_scope, init_params=False)
Add_Custom_Model(deploy_model, "data")

# Use the MOBILE EXPORTER to save the deploy model as pbs
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter_test.py
workspace.RunNetOnce(deploy_model.param_init_net)
workspace.CreateNet(deploy_model.net, overwrite=True) # (?)
init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)
with open(init_net_out, 'wb') as f:
    f.write(init_net.SerializeToString())
with open(predict_net_out, 'wb') as f:
    f.write(predict_net.SerializeToString())

print "Done, exiting..."





