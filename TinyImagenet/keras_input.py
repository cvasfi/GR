import os
import numpy as np
from PIL import Image

# Custom

def get_annotations_map():
    valAnnotationsPath = './Data/val/val_annotations.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}
    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]
    return valAnnotations

def load_images(path, num_classes):
    # Load images

    print('Loading ' + str(num_classes) + ' classes')

    X_train = np.zeros([num_classes * 500, 64, 64,3], dtype='uint8')
    y_train = np.zeros([num_classes * 500], dtype='uint8')

    trainPath = path + '/train'

    print('loading training images...');

    i = 0
    j = 0
    annotations = {}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
        annotations[sChild] = j
        for c in os.listdir(sChildPath):
            X = np.array(Image.open(os.path.join(sChildPath, c)))
            if len(np.shape(X)) == 2:
                X_train[i] = np.transpose(np.array([X, X, X]),[2,1,0])
            else:
                X_train[i] = X
            y_train[i] = j
            i += 1
        j += 1
        if (j >= num_classes):
            break

    print('finished loading training images')

    val_annotations_map = get_annotations_map()

    X_test = np.zeros([num_classes * 50,  64, 64, 3], dtype='uint8')
    y_test = np.zeros([num_classes * 50], dtype='uint8')

    print('loading test images...')

    i = 0
    testPath = path + '/val/images'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X = np.array(Image.open(sChildPath))
            if len(np.shape(X)) == 2:
                X_test[i] = np.transpose(np.array([X, X, X]),[2,1,0])
            else:
                X_test[i] = X
            y_test[i] = annotations[val_annotations_map[sChild]]
            i += 1
        else:
            pass

    print('finished loading test images') + str(i)

    return (X_train, y_train), (X_test, y_test)
