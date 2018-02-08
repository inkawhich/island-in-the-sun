# NAI
# This script takes a full dictionary as input
# The input dictionary is of the form: </path/to/jpg> <label>
# Once read in, we shuffle the lines then split them into two output
# files: a train dictionary and a test dictionary

import random
import os

# Directory where images are
input_dict = "/Users/nathaninkawhich/Documents/ECE590_Intro_ML/island-in-the-sun/caffe2/full_dictionary.txt"
output_train_dict = "/Users/nathaninkawhich/Documents/ECE590_Intro_ML/island-in-the-sun/caffe2/train_dictionary.txt"
output_test_dict = "/Users/nathaninkawhich/Documents/ECE590_Intro_ML/island-in-the-sun/caffe2/test_dictionary.txt"

# percentage of the file from the full dictionary that will go to train dictionary
percent_train = .8

# Read in all the lines from the full dictionary
lines = open(input_dict).readlines()

# Randomize the order of the lines
random.shuffle(lines)

# Decide how many will go to train
num_lines = len(lines)
num_train = int(num_lines*percent_train)

print "total lines read in: ", len(lines)
print "num train: ",len(lines[:num_train])
print "num test: ",len(lines[num_train:])
print "total lines written: ", (len(lines[:num_train])+len(lines[num_train:]))

# write the two separate files
open(output_train_dict, 'w').writelines(lines[:num_train])
open(output_test_dict, 'w').writelines(lines[num_train:])
