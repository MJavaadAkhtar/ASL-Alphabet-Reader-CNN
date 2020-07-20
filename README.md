# ASL Alphabet Reader CNN
<div style="padding-left:10px; padding-right:15px;">A Convulutional Neural Network that identifies American Sign Language Alphabets from the user. </div>
<hr>

### Table of Contents
1. [Overview](#Overview)
3. [Model Training](#model-training)
3. [How To Use It](#how-to-use-it)
3. [How To Read Output](#how-to-read-output)

 
### Overview:
This is a CNN model that allows users to do fundamental CNN analysis:
- Provides the users the ability to train CNN model on any type of imageset. 
- It provides users the ability to create complex mini-batches of large data.
- Allows users to train model, plot graphs and saves checkpoint for early stopping.

### Model Training:
The Neural Network is trained on dataset that contains alphabets of ASL, including sign language for `del` and `space`. The dataset contains a total of 87,000 images, and was divided into following 3 categories:
- `Training set`: Containing 52,200 images (60% of dataset)
- `Validation set  `: Containing 17,400 images (20% of dataset)
- `Testing set`: Containing 17,400 images (20% of dataset)

<!-- The model was trained on 100 epochs with batches of 128 and a learning rate of 0.001. The training accuracy is shown in the graph below:


The loss graph is shown below: -->


### How To Use It
#### If you want to train your own model
There are 3 major steps that needs to be done before CNN is ready to use. These are as follows:

<ol>
<li>
First you need to create a python virtual environment and install all the required pip libraries. You can do it as follows:

```sh
virtualenv venv
source ./venv/bin/activate

pip install -r requirements.txt
```
</li>

<li>
Now you need to specify the location for your training, testing and validation dataset. You need to change the following variables:

- `dataPreparation.py`  : train_path
- `dataPreparation.py`  : valid_path
- `dataPreparation.py`  : test_path

</li>

<li>
Change the number of distinct classes the images have. You can do that by changing the following variables:

- `models.py`: NumberOfUniqueClasses
</li>
</ol>

Now you are ready to train your CNN model. Just run `python train.py` to train the model. 
<div style='color:red'> <b>Note:</b> Make sure your image is 200px by 200px </div>

#### If you want to use a trained model
If you want to use a train model, you need to do the following steps:
 <ul>
 
 <li>

 First you need to place you images inside `Test` folder. 
 
 </li>
 
 <li>

 The CNN is placed inside the `model` folder. If you change the CNN model location, you need to update the following variable inside `model-load.py` file:
 - `model_data = torch.load('locationOfCNN Model')`

 </li>

 <li>

Now you are ready. Run the model on your images by running `python model-load.py` file.
 </li>

 </ul>

#### How To Read Output
It might be confusing at first how the output is displayed. The output of the model itself is displayed in numbers. Each number corresponds to a specific ASL alphabet. The mapping table is shown below:

| OUTPUT | MAPPING OF ALPHABET |
|--------|---------------------|
| O      | A                   |
| 1      | B                   |
| 2      | C                   |
| 3      | D                   |
| 4      | E                 |
| 5      | F                   |
| 6      | G                   |
| 7      | H                   |
| 8      | I                   |
| 9      | J                   |
| 10     | K                   |
| 11     | L                   |
| 12     | M                   |
| 13     | N                   |
| 14     | O                   |
| 15     | P             |
| 16     | Q                   |
| 17     | R                   |
| 18     | S                   |
| 19     | T                   |
| 20     | U                   |
| 21     | V               |
| 22     | W                   |
| 23     | X                   |
| 24     | Y                   |
| 25     | Z                   |
| 26     | del                   |
| 27     | nothing                   |
| 28     | space                   |