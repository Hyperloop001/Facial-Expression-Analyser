### Face detection using dlib 
### Last modified: 2018-09

import dlib
import numpy as np


#----------------------------------------------------------------------------------#
# drawFaceDetectionWithImagePath(imageLoadPath, faceDetector):
#    Type: string, dlib detector ==> void
#    Input: path to image, dlib face detector
#    Output: None
#    Side effects: Graphical result of face detection
#    Purposes: Detect face from a image
#    Note: None
def drawFaceDetectionWithImagePath(imageLoadPath, faceDetector = dlib.get_frontal_face_detector()):
    detector = faceDetector
    win = dlib.image_window()
    
    img = dlib.load_rgb_image(imageLoadPath)
    dets = detector(img, 1)
    
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))    
        
    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()
#----------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------#
# getFacePositionWithImagePath(imageLoadPath, faceDetector):
#    Type: string, dlib detector ==> numpy array
#    Input: path to image, dlib face detector
#    Output: numpy array that contains face position data (shape = (?, 4))
#    Side effects: None
#    Purposes: Detect face from a image, then return face positions
#    Note: None
def getFacePositionWithImagePath(imageLoadPath, faceDetector = dlib.get_frontal_face_detector()):
    detector = faceDetector

    img = dlib.load_rgb_image(imageLoadPath)
    dets = detector(img, 1)

    result = np.empty((0,4), int)

    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
        xmin, ymin, xmax, ymax = d.left() - 10, d.top() - 10, d.right() + 10, d.bottom() + 10
        result = np.append(result, np.array([[xmin if xmin > 0 else 0, ymin if ymin > 0 else 0, xmax if xmax > 0 else 0 , ymax if ymax > 0 else 0]]), axis=0)

    return result
#----------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------#
# getFacePositionWithNumpyImage(npImage, faceDetector):
#    Type: numpy image, dlib detector ==> numpy array
#    Input: image in numpy format, dlib face detector
#    Output: numpy array that contains face position data (shape = (?, 4))
#    Side effects: None
#    Purposes: Detect face from a image, then return face positions
#    Note: None
def getFacePositionWithNumpyImage(npImage, faceDetector=dlib.get_frontal_face_detector()):
    detector = faceDetector
    dets = detector(npImage, 1)

    result = np.empty((0, 4), int)
    for i, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
        xmin, ymin, xmax, ymax = d.left() - 10, d.top() - 10, d.right() + 10, d.bottom() + 10
        result = np.append(result, np.array(
            [[xmin if xmin > 0 else 0,
              ymin if ymin > 0 else 0,
              xmax if xmax > 0 else 0,
              ymax if ymax > 0 else 0]]),
                           axis=0)
    return result
#----------------------------------------------------------------------------------#


if __name__ == '__main__':
    getFacePositionWithImagePath("facial01.jpg")
    