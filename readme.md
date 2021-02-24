## Messier Object Identification
* [General info](#general-info)
* [Features](#features)
* [Technologies](#technologies)
* [Project repository](#project-repository)
* [Keras Model](#keras-model)
* [Dataset](#dataset)
* [Site](#site)
* [Setup](#setup)

## General info
Messier Object Identification is a program that I have created for my Master Thesis project: Identification of complex objects in astronomical images. It's main purpose is to identificate photos of messier objects and provide probability in percents of affiliation to existing object. Application is using a Convolutional Neural Network Keras model with Tensorflow backend. Model that is provided prior to the application has an accuracy of correct guesses at about 80%. Besidese that user can creat it's own Keras model in the app and try to teach the network with provided data. 

## Features
* App was created with Windows users in mind
* GUI was built via PyQT library
* User can upload photo of Messier object and see it's corresponding number (if he/she doesn't know what was photographed)
* Uploaded photo can be processed - converted to grayscale, resized or disturbed with noise of different types - Salt&Peper, Speckle, Gauss
* Possibilty to create a model of Convolutional Neural Network
* User, after creating model or using default one, can teach it with provided data
* Application provides catalogue of Messier objects with detailed data

## Technologies
Project was created with:
* Python 3.7.8
* PyQT 5
* Tensorflow
* Keras
* Numpy, Scipy

## Project repository
[Link](https://github.com/matchodura/Messier-Object-Identification)

## Keras Model

Model is a file with .h5 extension. It is recommended to use it at start of using the application.

[Download](https://drive.google.com/file/d/18ZlbySvh5kxF2jDv-vA6JLm6C_6SAvrj/view?usp=sharing)


## Dataset

Dataset that was used for teaching the network is a collection of 110 Messier Objects which translate to about 3000 (of original, not processed photos) .png images of 294px x 294px in size. To increase generalization of the model, images were subjected to some image processing tools like converting to grayscale, noising or mirroring. For the teaching process they have been divided in to train, test and validations sets. All of the image collections are available in the link below:



## Screenshots

When app is launched user can select from side navbar interesting options.

![](https://drive.google.com/uc?export=view&id=1V6PnL41_fxaVytf_zbkZdMZbkK06ACGd)


"Klasyfikator" means Classificator. It is the most important part of the program. Here user can upload Keras model, choose image of Messier Object and see probability of correct guess.

![](https://drive.google.com/uc?export=view&id=1F-R6vSWP7J59Y0DLmKC93et-J2bxdHSR)

"Uczenie sieci" alias network teaching is a tool that enables feature of creating new Convolutional Neural Network or relearning provided one with another Keras model. The dataset must be in correct folder format - /Train, /Test, /Val. Based on number of photos, batch size and number epochs and user's GPU time of process varies greatly. When epoch is finished graphs display validation error and accuracy.

![](https://drive.google.com/uc?export=view&id=1ZaMFWxKlo5VLnXmebp1mjjq1j_jiWQaV)

"Narzędzia pomocnicze" is an app section when user can edit photo with some image processing tools - resizing, noising (Gauss, Poisson, Speckle or Salt&Pepper), converting to grayscale, rotating, mirroring and finally saving it to desired location.

![](https://drive.google.com/uc?export=view&id=1IMpjCZoGGeOW-IszT_luLlRwaDtN5ST5)

"Katalog Obiektów Messiera" is a catalogue of Messier Objects. User can navigate between all (110) objects and see some trivia about them.

![](https://drive.google.com/uc?export=view&id=1pwtMqah1_aRSeZZamTcin0cBzwi2tPBh)

"Pomoc" is a page with information and step-by-step guides of how to use provided by the app tools.

![](https://drive.google.com/uc?export=view&id=1Ke_QBXsIfv2o26ce-UzMPpJ9oFl2KEPW)

"Kontakt" is a contact page. Feel free to contact me at any time!

![](https://drive.google.com/uc?export=view&id=1W5sqk9NQEwxXBLOJ_yU_NcegWAEpImQe)
	
## Setup
If you are interested in running this project, two ways are available for you:

First way (recommended)

0. Ensure Python 3.7.8 is installed
1. Clone or download repository to desired folder
2. From command prompt navigate to folder 
Messier Object Identification/gui
3. In project directory, write command
```
c:\MessierObjectIdentification\gui>python gui.py
```

Second way (not recommended, as it wasn't tested for every system). Download from link below:

[Download](https://drive.google.com/drive/folders/14A8dVAiegNHl6VE00j1V5XAzOI0-ZZrr?usp=sharing)
