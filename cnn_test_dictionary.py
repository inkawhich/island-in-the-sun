##################################################################################
# NAI
#
# This script is what we will use for testing our trained model based on a labeled
#   test dictionary. This does not format a csv for submission but can be used to
#   quickly see how the model we just trained does on our test set. Note, the test
#   dictionary is labeled, so this will output a % ACCURACY
#
##################################################################################

# import dependencies
print ("Import Dependencies...")
from matplotlib import pyplot
import numpy as np 
import os
import shutil
import operator
import caffe2.python.predictor.predictor_exporter as pe 
from caffe2.python import core,model_helper,net_drawer,optimizer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
import random
import skimage.io
from skimage.color import rgb2gray

##################################################################################
# Gather Inputs
test_dictionary = "../dataset/test_dictionary.txt"
predict_net = "./submission/cnn_predict_net.pb"
init_net = "./submission/cnn_init_net.pb"


##################################################################################
# Image formatting functions
# We do not need all of the augmentation like we do in training but we may want
#   to play with the data before feeding it to classifier
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

    # Create horizontal flip
    img2 = np.copy(img)
    img2 = np.fliplr(img2)

    # Create rotate 90
    img3 = np.copy(img)
    img3 = np.rot90(img3)

    img = img.swapaxes(1, 2).swapaxes(0, 1)    # HWC to CHW dimension
    img = img[(2, 1, 0), :, :]                 # RGB to BGR color order
    img = img * 255 - 128                      # Subtract mean = 128
    img /= 255.
    #pyplot.imshow(img)
    #pyplot.show()

    # Create horizontal flip
    img2 = img2.swapaxes(1, 2).swapaxes(0, 1)    # HWC to CHW dimension
    img2 = img2[(2, 1, 0), :, :]                 # RGB to BGR color order
    img2 = img2 * 255 - 128                      # Subtract mean = 128
    img2 /= 255.
    
    # Rot90
    img3 = img3.swapaxes(1, 2).swapaxes(0, 1)    # HWC to CHW dimension
    img3 = img3[(2, 1, 0), :, :]                 # RGB to BGR color order
    img3 = img3 * 255 - 128                      # Subtract mean = 128
    img3 /= 255.

    return img.astype(np.float32),img2.astype(np.float32),img3.astype(np.float32)

##################################################################################
# Bring up the network from the .pb files
with open(init_net,"rb") as f:
    init_net = f.read()
with open(predict_net,"rb") as f:
    predict_net = f.read()

p = workspace.Predictor(init_net, predict_net)


##################################################################################
# Loop through the test dictionary and run the inferences

test_dict = open(test_dictionary,"r")

num_correct = 0
total = 0


# For each line in the test dictionary file
for line in test_dict:
    # Split the line into its image and label components
    split_line = line.split()
    img = split_line[0]
    label = int(split_line[1].rstrip())

    # Format the image to feed into the net
    sample,sample1,sample2 = prepare_image(img)
    sample = sample[np.newaxis, :, :, :].astype(np.float32)
    sample1 = sample1[np.newaxis, :, :, :].astype(np.float32)
    sample2 = sample2[np.newaxis, :, :, :].astype(np.float32)

    # run the net and return prediction
    results = np.asarray(p.run([sample]))
    results1 = np.asarray(p.run([sample1]))
    results2 = np.asarray(p.run([sample2]))

    #print results[0,0,:]
    #print results1[0,0,:]
    #print results2[0,0,:]

    avg_results = (results[0,0,:] + results1[0,0,:] + results2[0,0,:])/3.
    #print avg_results
    #exit()
    
    print ("Image: ",img)
    print ("Label: ", label)
    #print "results: ",results

    # turn it into something we can play with and examine which is in a multi-dimensional array
    #results = np.asarray(results)
    #print "results shape: ", results.shape

    #results = results[0,0,:]
    #print "results shape: ", results.shape

    max_index, max_value = max(enumerate(avg_results), key=operator.itemgetter(1))

    print ("Prediction: ", max_index)
    print ("Confidence: ", max_value)

    if max_index == label:
        num_correct += 1

    total += 1
    
print ("\n**************************************")
print ("Accuracy = {}".format(num_correct/float(total)))


