# Dina
## Installation
```shell
git clone git@github.com:gamcoh/Dina.git
cd Dina
pip install -r requirements.txt
```
Then install [TensorFlow](https://www.tensorflow.org/install)

## Download the 20bn jester dataset
[Here](https://20bn.com/datasets/download)'s a link for downloading the dataset.
Once you finish downloading, you will need to move the frames into the dataset folder, you can use the `move_data.ipynb` script.

## Training the model
I'm using the resnet3d 101 model but you can use whatever model you want.
```shell
python train.py
```
This will save a `.h5` file.
