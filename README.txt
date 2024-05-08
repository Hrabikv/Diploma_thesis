This guide is describing setting up the environment and testing different setups for classification.
The project is written in Python version 3.10. The project was tested on laptop with Windows 10.

First, you need all the files from the GitHub repository: https://github.com/Hrabikv/Diploma_thesis
You can download it via GitHub UI, or you can clone it.

Installation
In the root folder is file requirements.txt, where are all used external libraries.
To get those, you need to install Python 3.10.
It is recommended that a virtual environment be created to make sure that there are no collisions in the version of libraries.
Run this command to create one:

python -m venv venv

This created a virtual environment named venv. To activate this environment, run this command:

venv\Scripts\activate

For installation of all libraries to the environment, run this command:

pip install -r requirements.txt

Work with the Project
The entry point of the program is mani.py in the src folder. To start a program, run this command:

python src/main.py

In the same virtual environment as the libraries you installed above,

Parameters of the Project
The project has seven parameters that you can change without having to open source code files.
All parameters are in config.txt file. All these parameters are mandatory.
The first is NUMBER_OF_CLASSES, which indicates what type of classification is wanted.
The number "2" is for binary classification, and the number "3" is for multi-class classification.
The second TRAINING_INFO and third TESTING_INFO parameters are about the amount of information printed into a console during the run.
The fourth parameter, called CLASSIFIERS, is an array of classification techniques that will be used during classification.
This parameter can have multiple values. The fifth parameter, TYPE_OF_DATA, is the choice of the dataset that will be used.
The FEATURE_VECTOR parameter is about the representation of the samples of the chosen dataset.
The last parameter, CUDA_USE, is the switch of the CUDA Toolkit environment.
All possible values of all parameters are shown in config.txt file.