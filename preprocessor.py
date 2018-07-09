import cv2
import numpy as np
import os


def prepare(input):
    # preprocessing the image input
    clean = cv2.fastNlMeansDenoising(input)
    ret, tresh = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)
    img = crop(tresh)   # Crop the non-informative pixels out

    ### NOT USED ##############################################################
    # # 40x10 image as a flatten array
    # flatten_img = cv2.resize(img, (40, 10), interpolation=cv2.INTER_AREA).flatten()
    #
    # # resize to 400x100
    # resized = cv2.resize(img, (400, 100), interpolation=cv2.INTER_AREA)
    # columns = np.sum(resized, axis=0)  # sum of all columns
    # lines = np.sum(resized, axis=1)  # sum of all lines
    #
    # h, w = img.shape
    # aspect = w / h

    # Amount of output elements
    #print("Flatten img: " + str(len(flatten_img)))  # 400 elements
    #print("Columns: " + str(len(columns)))          # 400 elements
    #print("Lines: " + str(len(lines)))              # 100 elements
    #print("Aspect: 1")                              # 1 element

    #return [*flatten_img, *columns, *lines, aspect]
    ###########################################################################
    return img


def crop(img):
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y+h, x: x+w]

### Functions I propably will use ###
# os.rename() - To move files or change directory
# os.mkdir() - Crate a new directory
# os.path.exists() - Check if directory exists

def process_and_write_img(in_path, out_path, genuine):
    """Preprocesses all the images and saves them in a another directory"""
    if genuine:
        title = 'genuine'
    else:
        title = 'forged'

    save_path = os.path.join(out_path, title)

    print('Preprocessing images...')
    i=1
    percent = 10
    img_folder = os.path.join(in_path, title)
    for filename in os.listdir(img_folder):
        img = cv2.imread(os.path.join(img_folder, filename), 0)
        if img is not None:
            preprocessed_img = prepare(img)
            cv2.imwrite(os.path.join(save_path , title + '-' + str(i) + '.png'), preprocessed_img)

        ### Progress indicator ###
        #print('Total images: ' + str(len(os.listdir(img_folder))))
        #print(len(os.listdir(img_folder))*percent/100)
        #print(i)
        if i >= len(os.listdir(img_folder))*percent/100:
            print(str(percent) + ' %...')
            percent += 10
        i+=1

    print('Images preprocessed and saved!\n')

def main():
    author = 'SigComp11'
    current_dir = os.path.dirname(__file__)
    training_dir = os.path.join(current_dir, 'data/training/', author)
    validation_dir = os.path.join(current_dir, 'data/validation/', author)

    print('Preprocess starts now...\n')
    new_dir_name = 'preproc-SigComp11'
    isTraining = True
    for in_path in [training_dir, validation_dir]:
        if isTraining:
            print('--- Working with TRAINING images ---')
        else:
            print('--- Working with VALIDATION images ---')
        # Creating a new directory to save all the preprocessed images
        out_path = os.path.join(os.path.split(in_path)[0], new_dir_name)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
            os.mkdir(os.path.join(out_path, 'genuine'))
            os.mkdir(os.path.join(out_path, 'forged'))

        process_and_write_img(in_path, out_path, genuine=True)
        process_and_write_img(in_path, out_path, genuine=False)
    print('Preprocess done!')


if __name__ == '__main__':
    main()
