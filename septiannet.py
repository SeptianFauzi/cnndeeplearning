from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class Septiannet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
            model = Sequential()
            inputShape = (height, width, depth)

            model.add(Conv2D(32, (3, 3), strides=1, data_format="channels_last", input_shape=inputShape,use_bias=False))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3),use_bias=False,strides=1))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3),use_bias=False,strides=1))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            # first (and only) set of FC => RELU layers
            model.add(Flatten())
            model.add(Dense(256))
            model.add(Activation("relu"))
            # softmax classifier
            model.add(Dense(classes))
            model.add(Activation("softmax"))
    
            # return the constructed network architecture
            return model
