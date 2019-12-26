import cv2
import os
import numpy as np
import pyautogui
import time
import csv
from PIL import ImageGrab
from pynput import keyboard
# Imports for testing
import pandas as pd
import numpy as np
import cv2
import ast

keys = []
isThreadRunning = False

def imageprocessing(image):
    processed_image = cv2.cvtColor(np.float32(image),cv2.COLOR_BGR2GRAY)
    #processed_image = cv2.Canny(processed_image,threshold1=200,threshold2=200)
    return processed_image

def writecsv(o1):
    with open('log.csv','a') as fp:
        writer = csv.writer(fp,delimiter=',')
        writer.writerow(o1)

def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        keys.append(key.char)
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
    key))
    if key == keyboard.Key.esc:
    # Stop listener
        return False

def key_check():
    # Non blocking way
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()
    isThreadRunning = True
    # Collect events until released
    '''with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()
    '''
    return keys

 # code copied from neural network code revert the changes after test   
def load_data():

    data_df = pd.read_csv('Trainingdata/log.csv', names=['image','input'])
    X1 = data_df[['image']].values
    Y1 = data_df['input'].values
    #print("Image values read from csv are %s " %X1)
    X = []
    Y = []
    p=0
    for add, out in zip(X1,Y1):
        image_path= os.path.join('Trainingdata',add[0])
        print("Path is %s" %image_path)
        img = cv2.imread(image_path, 0)
        #cv2.imshow('image',img)
        #cv2.waitKey(25)
        k = ast.literal_eval(out)
        if k == [0,0,1]:
            X.append(img)
            Y.append(k)
            X.append(cv2.flip( img, 1 ))
            Y.append([1,0,0])
        elif k==[1,0,0]:
            X.append(img)
            Y.append(k)
            X.append(cv2.flip( img, 1 ))
            Y.append([0,0,1])

        elif k == [0,1,0]:
            if p % 3 == 0:
                X.append(img)
                Y.append(k)
            else:
                pass
        p += 1
    print(Y)
    print(X)
    X = np.array(X, dtype=np.uint8)
    Y = np.array(Y, dtype=np.uint8)
    X = X.reshape(X.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    X = X.astype('float32')
    X /= 255
    cv2.destroyAllWindows()

    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid

def grab_screen(region=None):
    count = 0
    print("Grab screen called")
    #os.rmdir('Dataset')
    os.mkdir('Test')
    recordingstart = raw_input('do you want to start recording ')
    time.sleep(2)
    while count <= 400:
        image1 = ImageGrab.grab()
        image1 = imageprocessing(image1)
        image1 = cv2.resize(image1, (400, 150))
        image = "image%s.png" %count
        full_name_image = os.path.join('Test', image)
        cv2.imwrite(full_name_image,image1)
        count = count + 1
        print("Image path is %s" %full_name_image)
        img = cv2.imread(full_name_image,0)
        cv2.imshow('image',img)
        #keypressed = getkey()
        #print('alphanumeric key {0} pressed'.format(keypressed))
        #print('alphanumeric key %s global variable is ' %keys)

        #writecsv([full_name_image,keypressed])

def getkey():
    key = key_check()
    output = [0,0,0]
    #AWD

    if 'A' in key:
        output[0] = 1
    elif 'D' in key:
        output[2] = 1
    else:
        output[1] = 1

    return output

def runMyCarStraight(self):
    while(self.run):
        pyautogui.typewrite('w')

if __name__ == "__main__":
    time.sleep(2)
    runMyCarStraight()
    #grab_screen(None)