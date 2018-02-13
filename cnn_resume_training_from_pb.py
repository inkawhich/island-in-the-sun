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
import random
import skimage.io
from skimage.color import rgb2gray

##################################################################################
# MAIN

print "Entering Main..."

##################################################################################
# Gather Inputs
train_dictionary = "../dataset/train_dictionary.txt"
init_net_in = "cnn_init_net.pb"
init_net_out = "cnn_init_net_32epoch.pb"
batch_size = 50
num_epochs = 10





def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]

def prepare_image(img_path):
    
    img = skimage.io.imread(img_path)
    img = skimage.img_as_float(img)
    #img = rescale(img, 227, 227)
    img = crop_center(img, 90, 90)
    #img = img.swapaxes(1, 2).swapaxes(0, 1)    # HWC to CHW dimension
    #img = img[(2, 1, 0), :, :]                 # RGB to BGR color order
    #img = img * 255 - 128                      # Subtract mean = 128
    
    #pyplot.imshow(img)
    #pyplot.show()
    
    img = rgb2gray(img)
    
    # Create a new one to horizontal flip
    img2 = np.copy(img)
    img2 = np.expand_dims(img2, axis=2)          # expand dims for greyscale images
    img2 = np.fliplr(img2)
    img2 = img2.swapaxes(1, 2).swapaxes(0, 1)
    
    
    #pyplot.imshow(img, cmap='gray')
    #pyplot.show()
    img = np.expand_dims(img, axis=2)          # expand dims for greyscale images
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    
    #exit()
    #return img.astype(np.float32)
    return img.astype(np.float32),img2.astype(np.float32)



def make_batch(iterable, batch_size=1):
    length = len(iterable)
    for index in range(0, length, batch_size):
        yield iterable[index:min(index + batch_size, length)]

class Island_Dataset(object):
    def __init__(self, dictionary_file=train_dictionary):
        self.image_files = [line.split()[0] for line in open(dictionary_file)]
        self.labels = [line.split()[1] for line in open(dictionary_file)]
    
    def __getitem__(self, index):
        image,image2 = prepare_image(self.image_files[index])
        label = self.labels[index]
        return image, image2,label
    
    def __len__(self):
        return len(self.labels)
    
    def read(self, batch_size=50, shuffle=True):
        """Read (image, label) pairs in batch"""
        num = int(len(self)) // batch_size
        num_batches = batch_size * num
        order = list(range(num_batches))
        
        if shuffle:
            random.shuffle(order)
        for batch in make_batch(order, batch_size):
            images, labels = [], []
            
            
            for index in batch:
                image, image2,label = self[index]
                images.append(image)
                labels.append(label)
                
                images.append(image2)
                labels.append(label)
            yield np.stack(images).astype(np.float32), np.stack(labels).astype(np.int32).reshape((2*batch_size,))

##################################################################################
# Create model helper for use in this script

# specify that input data is stored in NCHW storage order
arg_scope = {"order":"NCHW"}

# create the model object that will be used for the train net
# This model object contains the network definition and the parameter storage
train_model = model_helper.ModelHelper(name="train_model", arg_scope=arg_scope, init_params=False)

##################################################################################
#### Add the model definition the the model object
# This is where we specify the network architecture from 'conv1' -> 'softmax'
# [Arch: data->conv->pool->...->fc->relu->softmax]

# Ouput size as fxn of input width (W), kernel size (K), padding (P), and stride (S)
# This can be used for calculating OFM size of conv and pooling layers
# O = ( (W-K+2P) / S ) + 1

def AddLeNetModel(model, data):
    # Size = 3x90x90
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    pool1 = brew.max_pool(model, conv1, 'pool1',kernel=2,stride=2)
    relu1 = brew.relu(model, pool1, 'relu1')
    # Size = 20x45x45
    conv2 = brew.conv(model, relu1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    pool2 = brew.max_pool(model, conv2, 'pool2',kernel=2, stride=2)
    relu2 = brew.relu(model, pool2, 'relu2')
    
    conv3 = brew.conv(model, relu2, 'conv3', dim_in=50, dim_out=50, kernel=5)
    pool3 = brew.max_pool(model, conv3, 'pool3',kernel=2, stride=2)
    relu3 = brew.relu(model, pool3, 'relu3')
    
    #drop3 = brew.dropout(model,relu3,'drop3',ratio=0.4,is_test=0)
    
    # Size = 50x22x22
    #fc4 = brew.fc(model, relu3, 'fc4', dim_in=50*19*19, dim_out=500)
    fc4 = brew.fc(model, relu3, 'fc4', dim_in=50*7*7, dim_out=500)
    relu4 = brew.relu(model, fc4, 'relu4')
    
    #drop4 = brew.dropout(model,relu4,'drop4',ratio=0.2,is_test=0)
    
    # Size = 1x500
    pred = brew.fc(model,relu4,'pred',500,2)
    softmax = brew.softmax(model,pred, 'softmax')



##################################################################################
##################################################################################
##################################################################################

softmax=AddLeNetModel(train_model, 'data')

##################################################################################
##################################################################################
##################################################################################

# Populate the model obj with the init net stuff, which provides the parameters for the model
init_net_proto = caffe2_pb2.NetDef()
with open(init_net_in, "rb") as f:
    init_net_proto.ParseFromString(f.read())
tmp_param_net = core.Net(init_net_proto)
train_model.param_init_net = tmp_param_net



##################################################################################
#### Add training operators to the model

xent = train_model.LabelCrossEntropy(['softmax', 'label'], 'xent')
loss = train_model.AveragedLoss(xent, 'loss')
brew.accuracy(train_model, ['softmax', 'label'], 'accuracy')
train_model.AddGradientOperators([loss])

optimizer.build_sgd(train_model,base_learning_rate=0.01, policy="step", stepsize=10000, gamma=0.1, momentum=0.9)

##################################################################################
#### Run the training procedure

# Initialization.
train_dataset = Island_Dataset(dictionary_file=train_dictionary)

# Prime the workspace with some data so we can run init net once
for image, label in train_dataset.read(batch_size=1):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break


# run the param init network once
workspace.RunNetOnce(train_model.param_init_net)
# create the network
workspace.CreateNet(train_model.net, overwrite=True)
# Set the total number of iterations and track the accuracy and loss


# Set the total number of iterations and track the accuracy and loss
accuracy = []
loss = []

# Manually run the network for the specified amount of iterations
for epoch in range(num_epochs):

    for index, (image, label) in enumerate(train_dataset.read(batch_size)):

        # image.shape = [bsize, 20, 100, 100]
        workspace.FeedBlob("data", image)
        workspace.FeedBlob("label", label)
        workspace.RunNet(train_model.net) 
        curr_acc = workspace.FetchBlob('accuracy')
        curr_loss = workspace.FetchBlob('loss')
        accuracy.append(curr_acc)
        loss.append(curr_loss)
        print "[{}][{}/{}] loss={}, accuracy={}".format(epoch, index, int(len(train_dataset) / batch_size),curr_loss, curr_acc)

##################################################################################
#### Save the trained model for testing later

# save as two protobuf files (predict_net.pb and init_net.pb)
# predict_net.pb defines the architecture of the network
# init_net.pb defines the network params/weights
print "Saving the trained model to predict/init.pb files"
deploy_model = model_helper.ModelHelper(name="cifar10_deploy", arg_scope=arg_scope, init_params=False)
AddLeNetModel(deploy_model, "data")

# Use the MOBILE EXPORTER to save the deploy model as pbs
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter_test.py
workspace.RunNetOnce(deploy_model.param_init_net)
workspace.CreateNet(deploy_model.net, overwrite=True) # (?)
init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)
with open(init_net_out, 'wb') as f:
    f.write(init_net.SerializeToString())

print "Done, exiting..."


# After execution is done lets plot the values
pyplot.plot(loss,'b', label='loss')
pyplot.plot(accuracy,'r', label='accuracy')
pyplot.legend(loc='upper right')
pyplot.show()


