# NAI

import os
import glob

# Label file for training data
training_labels_file = "../labels_training.csv"

# Directory where images are
training_data_dir = "/Users/nathaninkawhich/Documents/ECE590_Intro_ML/island-in-the-sun/training"

# Output file name
training_dict = "full_dictionary.txt"

f = open(training_labels_file,"rb")
out = open(training_dict,"w")

# Get rid of the header line in the csv file
f.readline()

for line in f:
    split = line.split(",")
    dat = split[0]
    label = split[1].rstrip()
    print "'{}' -> '{}'".format(dat,label)
    out_str = training_data_dir + "/" + dat + ".tif " + label
    print out_str
    out.write(out_str + "\n")




