 #Team 3 

 ##island-in-the-sun competition submission

**Team 3 Members**

- Nathan Inkawhich
- Lisa Sapozhnikov
- Yifan Li

## How to use this code...

1. Edit the inputs section of the *create_dictionary_file.py* script to point to your *labels_training.csv* and *training* data directory that you downloaded from Kaggle

2. Run *create_dictionary_file.py* which will create a txt file called *train_dictionary.txt*

3. Run *cnn_train_and_save.py* which will train a model on the data described in the train dictionary and then save the trained model. This step will output two files: *cnn_init_net.pb* and *cnn_predict_net.pb* which is Caffe2's way of saving trained models. If you want to change how long the model trains for, change the *num_epochs* variable in the input section. (**NOTE: THIS STEP WILL TAKE A WHILE!**)

4. Edit *cnn_test_directory.py* and set the path to the test data directory where all of the test .tif files are.

5. Run *cnn_test_directory.py*, which will classify each of the .tif files in the specified directory and write the predictions to a csv file which can be directly used as a Kaggle submission.

