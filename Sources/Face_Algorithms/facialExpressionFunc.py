### Function supports to facial expression detection
### Last modified: 2018-09
### *** Credit: https://github.com/serengil/tensorflow-101/blob/master/python/facial-expression-recognition.py *** ###
### *** Credit: Kaggle challange fer2013.csv file *** ###

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


#----------------------------------------------------------------------------------#
# getFacialExpressionModel(image_shape, num_classes):
#    Type: shape, int ==> keras CNN model
#    Input: input image shape, number of classes
#    Output: keras CNN model for facial expression detection
#    Side effects: None
#    Purposes: Generate model that recognize facial expressions
#    Note: None
#    Credit: https://github.com/serengil/tensorflow-101/blob/master/python/facial-expression-recognition.py
def getFacialExpressionModel(image_shape = (48, 48, 1),num_classes = 7):
    # construct model
    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape = image_shape))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # flatten model
    model.add(Flatten())

    # fully connected layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    # output layer: shape as (num_of_instances, num_class)
    model.add(Dense(num_classes, activation='softmax'))

    return model
#----------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------#
# trainFacialExpressionModel(datasetLoadPath, modelLoadPath, modelSavePath, num_classes, batch_size, epochs):
#    Type: string, string, string, int, int, int ==> void
#    Input: path to dataset, path to model load location, path to model save location, number of classes, batch size, training iterations
#    Output: None
#    Side effects: Store trained model to specified location
#    Purposes: Train model that recognize facial expressions
#    Note: by default, classes are: angry, disgust, fear, happy, sad, surprise, neutral
def trainFacialExpressionModel(datasetLoadPath = '../Data/Facial_Expression_Datasets/fer2013/fer2013.csv', 
                               modelLoadPath = '../Data/Facial_Expression_Models/model_1/facial_expression_model_weights.h5',
                               modelSavePath = '../Data/Facial_Expression_Models/model_2/facial_expression_model_weights.h5',
                               num_classes = 7, batch_size = 256, epochs = 5, usingExistModel = False):
    # cpu - gpu configuration
    config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)     
    
    # read dataset (fer2013.csv)
    with open(datasetLoadPath) as fer2013:
        file_content = fer2013.readlines()
        lines = np.array(file_content)
        num_of_instances = lines.size
    print("Number of instances: %d" % (num_of_instances))
    print("Instance length: %d" % (len(lines[1].split(",")[1].split(" ")))) 
    
    # initialize then load training set and testing set
    x_train, y_train, x_test, y_test = [], [], [], [] 
    
    for i in range(1,num_of_instances):
        try:
            emotion, img, usage = lines[i].split(",")    
            val = img.split(" ")            
            pixels = np.array(val, 'float32') 
            # one hot encoding
            emotion = keras.utils.to_categorical(emotion, num_classes)   
            if 'Training' in usage:
                y_train.append(emotion)
                x_train.append(pixels)
            elif 'PublicTest' in usage:
                y_test.append(emotion)
                x_test.append(pixels)
        except:
            print("Exception")
    
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')
    x_train /= 255 
    x_test /= 255
    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')   
    print("Number of training samples: %d" % (x_train.shape[0]))
    print("Number of testing samples: %d" % (x_test.shape[0]))

    # generate CNN model
    model = getFacialExpressionModel(num_classes = num_classes)
    
    # batch generator
    gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)    
    
    if usingExistModel:
        model.load_weights(modelLoadPath)
    
    # train and save model
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam() , metrics=['accuracy'])    
    model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)  
      # model.fit_generator(x_train, y_train, epochs=epochs) # train for all trainset
    model.save_weights(modelSavePath)
#----------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------#
# emotion_analysis(emotions):
#    Type: list of num ==> void
#    Input: list of num (length 7)
#    Output: None
#    Side effects: Plot graph
#    Purposes: Drawing bar chart for emotion preditions
#    Note: None
#    Credit: https://github.com/serengil/tensorflow-101/blob/master/python/facial-expression-recognition.py
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))  
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.show()
#----------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------#
# facialPredictionWithImagePath(imageLoadPath, modelLoadPath, num_classes):
#    Type: string, string, int ==> void
#    Input: path to image, path to model location, number of classes
#    Output: None
#    Side effects: Graphical analysis of emotion
#    Purposes: Recognize facial expressions
#    Note: by default, emotions are: angry, disgust, fear, happy, sad, surprise, neutral
def facialPredictionWithImagePath(imageLoadPath, modelLoadPath = '../Data/Facial_Expression_Models/model_0/facial_expression_model_weights.h5', num_classes = 7):
    # generate CNN model
    model = getFacialExpressionModel(num_classes = num_classes)
    
    # load model weights
    model.load_weights(modelLoadPath)
    
    # load img to PIL format
    img = image.load_img(imageLoadPath, grayscale=True, target_size=(48, 48))
    
    # convert PIL img to numpy array, then convert the array as input
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    
    # Predict image through model
    custom = model.predict(x)
    print(custom[0])
    emotion_analysis(custom[0])
#----------------------------------------------------------------------------------#

    