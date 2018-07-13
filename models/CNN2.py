from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import RMSprop

def create_model():
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    img_input = layers.Input(shape=(150, 150, 3))

    # First convolution extracts 128 filters that are 5x5
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(128, 5, activation='relu')(img_input)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Dropout(0.5)(x)

    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = layers.Flatten()(x)

    x = layers.Dense(96, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(54, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = layers.Dropout(0.25)(x)

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
