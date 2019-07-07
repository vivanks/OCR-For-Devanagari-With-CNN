#  [OCR-For-Devanagari-With-Image-Processing](https://github.com/vivanks/OCR-For-Devanagari-With-CNN)


This project based on keras's CNN for modeling the devanagiri data set based upon **preprocessed data**. Keras is a wrapper library who's backend can be either Tensorflow, CNTK or Theano. It's very friendly for quick prototyping and testing ideas (of course, provided your model doesn't take days to compile as well :D ) I use the tensorflow-gpu backend for Keras.

# Dataset

-   The dataset was created by extraction and manual annotation of thousands of characters from handwritten documents. Creator Name: Shailesh Acharya, Email:  [sailes437@gmail.com](mailto:sailes437@gmail.com), Institution: University of North Texas, Cell: +19402200157 Creator Name: Prashnna Kumar Gyawali, Email:  [gyawali.prasanna@gmail.com](mailto:gyawali.prasanna@gmail.com), Institution: Rochester Institute of Technology
    
-   Data Type: GrayScale Image The image dataset can be used to benchmark classification algorithm for OCR systems. The highest accuracy obtained in the Test set is 98.47%. Model Description is available in the paper.
    
-   Image Format: .png Resolution: 32 by 32 Actual character is centered within 28 by 28 pixel, padding of 2 pixel is added on all four sides of actual character.
    
-   S. Acharya, A.K. Pant and P.K. Gyawali â€œDeep Learning Based Large Scale Handwritten Devanagari Character Recognitionâ,In Proceedings of the 9th International Conference on Software, Knowledge, Information Management and Applications (SKIMA), pp. 121-126, 2015.

## Installation
Install the dependencies and devDependencies and start the program.

    $ python3
    $ tensorflow
    $ keras
Import relevant libraries

    # Standard useful data processing imports
    import random
    import numpy as np
    import pandas as pd
    # Visualisation imports
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Scikit learn for preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    # Keras Imports - CNN
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
    from keras.optimizers import Adam
    from keras.utils.np_utils import to_categorical

# Development of Model

Let's first define a model. Here we shall use the keras Sequential model, which essentially involves us adding a layer one after the other.... sequentially.

    cnn = Sequential()
The real fun part is defining the LAYERS.

The first layer, a.k.a the input layer requires a bit of attention in terms of the shape of the data it will be looking at.

So just for the first layer, we shall specify the input shape, i.e., the shape of the input image - rows, columns and number of channels.

Keras also has this neat API that joins the convolutional and activation layers into 1 API call.

    kernelSize = (3, 3)
    ip_activation = 'relu'
    ip_conv_0 = Conv2D(filters=32, kernel_size=kernelSize, input_shape=im_shape, activation=ip_activation)
    cnn.add(ip_conv_0)
So a common trick here used when developing CNN architectures is to add two Convolution+Activation layers back to back BEFORE we proceed to the pooling layer for downsampling.

This is done so that the kernel size used at each layer can be small.

when multiple convolutional layers are added back to back, the overall effect of the multiple small kernels will be similar to the effect produced by a larger kernel, like having two 3x3 kernels instead of a 7x7 kernel. (Reference link coming soon!)

    # Add the next Convolutional+Activation layer
    ip_conv_0_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
    cnn.add(ip_conv_0_1)
    
    # Add the Pooling layer
    pool_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
    cnn.add(pool_0)

Let's do this again. i.e, One more ConvAct + ConvAct + Pool layer sequence.

    ip_conv_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
    cnn.add(ip_conv_1)
    ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
    cnn.add(ip_conv_1_1)
    
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
    cnn.add(pool_1)
Here's an interesting part. Two common problems with most models is that they either underfit, or they overfit. With a simple CNN such as this, there is a high probability that your model would begin overfitting the data. i.e, it relies on the training data too much.

There are multiple ways to address the problem of overfitting. This is a pretty neat  [site](https://towardsdatascience.com/deep-learning-3-more-on-cnns-handling-overfitting-2bd5d99abe5d)  which talks about this succintly.

We have 92000 data points to play with. I'm avoiding augmentation for the same reason.

For our case, we will use a simple Dropout layer to make sure the network does not depend on the training data too much.

    # Let's deactivate around 20% of neurons randomly for training
    drop_layer_0 = Dropout(0.2)
    cnn.add(drop_layer_0)

We are done with the Convolutional layers, and will proceed to send this data to the Fully Connected ANN. To do this, our data must be a 1D vector and not a 2D image. So we flatten it.

    flat_layer_0 = Flatten()
    cnn.add(Flatten())
And proceed to the ANN -

    # Now add the Dense layers
    h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')
    cnn.add(h_dense_0)
    # Let's add one more before proceeding to the output layer
    h_dense_1 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')
    cnn.add(h_dense_1)

Our problem is classification. So succeeding the final layer, we use a softmax activation function to classify our labels.

    op_activation = 'softmax'
    output_layer = Dense(units=n_classes, activation=op_activation, kernel_initializer='uniform')
    cnn.add(output_layer)
Almost done. Now we just need to define the optimizer and loss functions to minimize and compile the CNN

    opt = 'adam'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    # Compile the classifier using the configuration we want
    cnn.compile(optimizer=opt, loss=loss, metrics=metrics)

# Results

    Accuracy: 98.30%


