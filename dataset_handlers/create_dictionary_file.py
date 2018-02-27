# NAI

import os
import glob
import numpy as np


# CSV Label file [of format (id,score)]
#labels_file = os.path.abspath("../../dataset/labels_training.csv")

# Directory where images are
#data_dir = os.path.abspath("../../dataset/training")

# Output file name
#output_dict = os.path.abspath("../../dataset/equalized_training_dictionary.txt")


labels_file = os.path.abspath("../../dataset/labels_testing.csv")
data_dir = os.path.abspath("../../dataset/testing")
output_dict = os.path.abspath("../../dataset/test_dictionary.txt")

###############################################################
# There is an imbalance in the number of 1's and 0's
# Set this to one if you want to balance the number
#   Note: If you set this you will be ignoring some samples
equalize = 0
###############################################################


f = open(labels_file,"r")
out = open(output_dict,"w")

# Get rid of the header line in the csv file
f.readline()

# Stat keepers
ignored_zeros = 0
cnt = 0
label_cnt = np.zeros(shape=(1,2))

for line in f:
    split = line.split(",")
    dat = split[0]
    label = split[1].rstrip()
    
    # Keep stats
    label_cnt[0,int(label)] += 1
    
    # If we are equalizing and the number of 0's has exceeded the number of 1's, then ignore this zero
    if ((equalize == 1) and (label_cnt[0,0] > 505) and label == '0'):
        #print ("Too many 0's, Ignoring this one")
        ignored_zeros += 1
        continue

    #print ("fname: '{}' -> label: '{}'".format(dat,label))
    # Form the output string to write to the file
    out_str = data_dir + "/" + dat + ".tif " + label
    #print (out_str)
    # Write the string to the file
    out.write(out_str + "\n")
    cnt += 1

#print "\n*************************"
#print "Number of 1's = {}".format(label_cnt[0,1])
#print "Number of 0's = {}".format(label_cnt[0,0] - ignored_zeros)



