# code-released

### Directory Remarks
 
 - You should place your static graph of your model in the directory named model
 - You will get the tensorboard summary in the directory of ./checkpoint/tensorboard/*, meaning you will be allowd to show your data visualization in this root directory from the specific url like ip:6006 or ip:6006 (ip probably is your localhost in your local browser)
 - You can get the detail of your model validated of accuracy and loss from the folder of ./log/valid_report.txt
 - You should just run python ./train.py to train your specific network or run python ./predict.py to get the trained model's predition in your test dataset and get your submission in the csv format. Absolutely, you should make your model be visual in the ./utils.py.
 - You should just set the specific network in the config.json. You can change your hyperparameters in the process of train to change it by your ideas. This will be in work immediately. And when the try_to_validate is set true, the train will be in a state of validation in validation dataset. Make sure the try_to_validate item is set to false before the validation is over.
 
 ### Hyperparameters Details of the config.json
 
   1. "learning_rate": If the value is negtive, the train will change by the setting before according the train epoch. If you change in the json file when the train is running, the setting will work in a second. Try to change it by the assist of the tensorboard.
   2. "weights_regularize_lambda": A hyperparater controls the fully connected layers' weights regularization. A effective controlling parameter to avoid the overfit.
   3. "three_class_regularize_lambda": A loss set as a form of regularization for the specifid aim of putting three labels for one image. It will represents the difference of the sum of the submitted three results' probabilities between 1.0.
   4. "try_to_validate": if the value is set to true, the process will go into a validation at once. Make sure to turn it to false before the validation is over.
   5. "keep_prob": A method of avoiding overfit. But maybe replaced by the batch normalization. It is deprecated in most of my models.
   6. "batch_size": Control the size of input. The setting will worked only at the beginning of the train.
   7. "input_image_size": Control the input image size of your model. It shouldn't be changed when your saved your model ever.
   8. "lambda_level1": A parameter to control the weight of the product tree level 1 loss of a sub-classification task.
   9. "lambda_level2": A parameter to control the weight of the product tree level 2 loss of a sub-classification task.
   10. "lambda_level3": A parameter to control the weight of the product tree level 3 loss of a sub-classification task.
   11. "lambda_label": A parameter to control the weight of the image label loss of the main-classification task.
