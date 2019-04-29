# code-released

### Directory Remarks
 
 - You should place your static graph of your model in the directory named model
 - You will get the tensorboard summary in the directory of ./checkpoint/tensorboard/*, meaning you will be allowd to show your data visualization in this root directory from the specific url like ip:6006 or ip:6006 (ip probably is your localhost in your local browser)
 - You can get the detail of your model validated of accuracy and loss from the folder of ./log/valid_report.txt
 - You should just run python ./train.py to train your specific network or run python ./predict.py to get the trained model's predition in your test dataset and get your submission in the csv format. Absolutely, you should make your model be visual in the ./utils.py.
 - You should just set the specific network in the config.json. You can change your hyperparameters in the process of train to change it by your ideas. This will be in work immediately. And when the try_to_validate is set true, the train will be in a state of validation in validation dataset. Make sure the try_to_validate item is set to false before the validation is over.
 
 ### Hyperparameters Detail of the config.json
