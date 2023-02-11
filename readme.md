# Code for course project of Advanced Machine Learning (AML) AA 2022/2023
"Sim-to-Real transfer of Reinforcement Learning policies in robotics" exam project.

*Starting code has been forked from [this repo](https://github.com/gabrieletiboni/aml22-rl), check it out for basic information.*

# Organization
There are 2 main folders besides the predefined files:
- train
- test

In `/train` folder it's possible to find:

1. `train_main.ipynb`, that is the Jupyter notebook mainly used for training models
2. `/environments` in which there is a collection of all the complete environments used for different training purposes (e.g.: Domain Randomization, images, etc)
3. `/training_logs` in which there are the logs relative to the training of different models. Logs can be visualized using tensorboard.
4. `custom_CNN_classes` in which there are some attempts to design a customCNN to be used as policy. In particular a working customCNN with LSTM cell has been tried, results were very poor.

In `/test` folder there are:

1. `eval_model.py`, basic script to evaluate the mean reward and std of reward of standard models (i.e. w/out images as observations)
2. `/trained_models` in which a collection of trained models trained with different hyperparameters and using different techniques can be found.

Besides models in `/trained_models` folder, other trained models related to visual-based training can be found at the [link](https://drive.google.com/drive/folders/14z6EJqn3G7yBsEv2RhbwGj9gCIfF5rv9?usp=sharing).

