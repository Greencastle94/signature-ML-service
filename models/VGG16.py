from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import RMSprop

def create_model():
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    img_input = layers.Input(shape=(150, 150, 3))

    x = layers.ZeroPadding2D((1,1))(img_input)
    x = layers.Convolution2D(64, 3, 3, activation='relu')(x)
    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(64, 3, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2,2), strides=(2,2))(x)

    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(128, 3, 3, activation='relu')(x)
    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(128, 3, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2,2), strides=(2,2))(x)

    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(256, 3, 3, activation='relu')(x)
    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(256, 3, 3, activation='relu')(x)
    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(256, 3, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2,2), strides=(2,2))(x)

    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(512, 3, 3, activation='relu')(x)
    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(512, 3, 3, activation='relu')(x)
    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(512, 3, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2,2), strides=(2,2))(x)

    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(512, 3, 3, activation='relu')(x)
    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(512, 3, 3, activation='relu')(x)
    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.Convolution2D(512, 3, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2,2), strides=(2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Create output layer with a single node and sigmoid activation
    output = layers.Dense(1, activation='softmax')(x)

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully
    # connected layer + sigmoid output layer
    model = Model(img_input, output)

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc'])

    # model.summary()

    return model
