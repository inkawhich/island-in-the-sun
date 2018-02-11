# NAI

# This script will be use to generate the submission files. It takes the model pb's and 
#  the name of the testing directory as input and outputs a csv file in the format for 
#   submission

# The only changes we should have to make to this file is the name of the model pb's

# import dependencies
print "Import Dependencies..."
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
import glob


##################################################################################
# Gather Inputs

test_data_dir = os.path.abspath("../dataset/testing")
test_csv = "t3_submission.csv"

predict_net = "cnn_predict_net.pb"
init_net = "cnn_init_net.pb"



##################################################################################
# Image formatting functions
def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]

def prepare_image(img_path):
    
    img = skimage.io.imread(img_path)
    img = skimage.img_as_float(img)
    img = crop_center(img, 90, 90)
    img = rgb2gray(img)
    img = np.expand_dims(img, axis=2)          # expand dims for greyscale images
    img = img.swapaxes(1, 2).swapaxes(0, 1)

    return img.astype(np.float32)
    





##################################################################################
# Bring up the network from the .pb files
with open(init_net) as f:
    init_net = f.read()
with open(predict_net) as f:
    predict_net = f.read()

p = workspace.Predictor(init_net, predict_net)





##################################################################################
# Loop through the test directory and classify all images

f = open(test_csv,"w")
f.write("id,score\n")

for img in glob.glob(test_data_dir + "/*"):
    
    print "Testing: ",img
    img_id = os.path.basename(img).split(".")[0]

    # Format the image to feed into the net
    sample = prepare_image(img)
    sample = sample[np.newaxis, :, :, :].astype(np.float32)

    # run the net and return prediction
    results = p.run([sample])

    # turn it into something we can play with and examine which is in a multi-dimensional array
    results = np.asarray(results)
    #print "results shape: ", results.shape

    results = results[0,0,:]
    #print "results shape: ", results.shape

    max_index, max_value = max(enumerate(results), key=operator.itemgetter(1))

    print "Prediction: ", max_index
    print "Confidence: ", max_value
    
    f.write(img_id + "," + str(max_index) + "\n")

    

