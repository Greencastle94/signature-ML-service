import os
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
# Here you choose which model to use
import models.CNN as net

def evaluate(history):
    # Retrieve a list of accuracy results on training and test data
    # sets for each training epoch
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')

    plt.figure()

    # Plot training and validation loss per epoch
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')

    plt.show()

def main():
    dataset = 'new'            # Which dataset to use
    batch_size = 12
    epochs = 50

    model = net.create_model()

    current_dir = os.path.dirname(__file__)
    training_dir = os.path.join(current_dir, 'datasets/', dataset, 'training/')
    validation_dir = os.path.join(current_dir, 'datasets/', dataset, 'validation/')

    # Calculate training & validation steps
    num_images = 0
    for folder in ['true/', 'false/']:
        dir = os.path.join(training_dir, folder)
        num_images += len(os.listdir(dir))
    train_steps = num_images/batch_size
    num_images = 0
    for folder in ['true/', 'false/']:
        dir = os.path.join(validation_dir, folder)
        num_images += len(os.listdir(dir))
    val_steps = num_images/batch_size

    print('Dataset: ' + dataset)
    print('batch_size: ' + str(batch_size))
    print('train_steps: ' + str(train_steps))
    print('val_steps: ' + str(val_steps))
    print('epochs: ' + str(epochs))

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            training_dir,  # This is the source directory for training images
            target_size=(150, 150),  # All images will be resized to 150x150
            batch_size=batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = valid_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')

    # Train model
    history = model.fit_generator(
          train_generator,
          steps_per_epoch=train_steps,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=val_steps,
          verbose=2)

    evaluate(history)


if __name__ == '__main__':
    main()
