### Facial expression detector
### The way it works: 1. detect faces and draw frame
###                   2. create subimage for each face
###                   3. apply facial expression CNN on each subimage
###                   4. show the most possible emotions to user
### Last modified: 2018-09

import sys
import cv2
import dlib
from Face_Algorithms.facialExpressionFunc import *
from Face_Algorithms.faceDetectionFunc import *
from PIL import Image
from timeit import time
from keras.preprocessing import image


def main(runDetector = False,
         trainModel = False,
         testModel = False,
         usingExistModel = True,
         exportVideo = False,
         num_classes = 7,
         batch_size = 256,
         epochs = 5,
         testModelImagePath = "test1.jpg",
         datasetLoadPath = '../Data/Facial_Expression_Datasets/fer2013/fer2013.csv',
         modelLoadPath = '../Data/Facial_Expression_Models/model_0/facial_expression_model_weights.h5',
         modelSavePath = '../Data/Facial_Expression_Models/model_2/facial_expression_model_weights.h5',
         videoSavePath = '../Data/Output_Videos/output01.mp4'):
    # Train facial expression CNN model
    if trainModel:
        print("Train facial expression CNN model...")
        trainFacialExpressionModel(datasetLoadPath = datasetLoadPath, modelLoadPath = modelLoadPath,
                                   modelSavePath = modelSavePath, usingExistModel = usingExistModel,
                                   num_classes = num_classes, batch_size = batch_size, epochs = epochs)
        print("Done!")
        sys.exit(0)

    # Test facial expression CNN model
    if testModel:
        print("Test facial expression CNN model...")
        facialPredictionWithImagePath(testModelImagePath, modelLoadPath)
        print("Done!")
        sys.exit(0)

    # Run detector on video stream(s)
    if runDetector:
        print("Run facial expression CNN model on video stream(s): ")

        # generate face detector
        print("Loading face detector...")
        faceDetector = dlib.get_frontal_face_detector()

        # generate and initialize facial expression CNN model
        print("Loading facial expression CNN model...")
        facialExpressionModel = getFacialExpressionModel(num_classes = num_classes)
        facialExpressionModel.load_weights(modelLoadPath)

        # detection procedure
        print("Program initializing...")
        video_capture = cv2.VideoCapture(0)
        if exportVideo:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_writer = cv2.VideoWriter(videoSavePath, fourcc, 8.0, (640, 480))
        print("Video captured!")
        print("Start processing video: ")
        fps = 0.0
        while True:
            # read in video frame
            video_status, npImage = video_capture.read()
                # print(npImage.shape)
                # npImage = cv2.resize(npImage,(640,480), interpolation=cv2.INTER_LINEAR)
            npImage = cv2.flip(npImage, 1)
            if video_status != True:
                break;

            # start timing
            time_start = time.time()

            # run face detector
            pilImage = Image.fromarray(npImage)
            result = getFacePositionWithNumpyImage(npImage, faceDetector = faceDetector)
            numFaceDetected = result.shape[0]
            print("Number of face detected: %d" % (numFaceDetected))

            # detect emotion and draw label out faces
            for i in range(numFaceDetected):
                # get the area that contain face
                xmin, ymin, xmax, ymax = result[i]
                xmin = xmin if xmin < 640 else 640
                ymin = ymin if ymin < 480 else 480
                xmax = xmax if xmax < 640 else 640
                ymax = ymax if ymax < 480 else 480


                # generate 48 * 48 face image in PIL format, then convert to CNN input
                pilFaceImage = ((pilImage.crop((xmin, ymin, xmax, ymax))).resize((48, 48), Image.BILINEAR)).convert(mode = 'L')
                npFaceImage = image.img_to_array(pilFaceImage)
                npFaceImage = np.expand_dims(npFaceImage, axis=0)
                npFaceImage /= 255

                """
                # generate 48 * 48 face image in numpy format
                # capture subimage
                npFaceImage = npImage[ymin:ymax, xmin:xmax, :]
                # greyscale subimage
                npFaceImage = np.dot(npFaceImage, [0.299, 0.587, 0.114])
                # resize subimage
                npFaceImage = cv2.resize(npFaceImage,(48,48), interpolation=cv2.INTER_LINEAR)
                # expand subimage dimension
                npFaceImage = np.expand_dims(npFaceImage, axis=2)
                # normalize subimage
                npFaceImage /= 255
                # detect emotion with CNN model
                """

                # emotion prediction with CNN model
                emotion_pribability = (facialExpressionModel.predict(npFaceImage))[0]

                # get the emotion with largest probility
                sortedIndex = np.argsort(emotion_pribability)[::-1]
                index_1, index_2 = sortedIndex[0], sortedIndex[1]
                emotion_1, emotion_2 = "unknown", "unknown"

                # get first emotion
                if index_1 == 0:
                    emotion_1 = "angry"
                elif index_1 == 1:
                    emotion_1 = "disgust"
                elif index_1 == 2:
                    emotion_1 = "fear"
                elif index_1 == 3:
                    emotion_1 = "happy"
                elif index_1 == 4:
                    emotion_1 = "sad"
                elif index_1 == 5:
                    emotion_1 = "surprise"
                elif index_1 == 6:
                    emotion_1 = "neutral"

                # get second emotion
                if index_2 == 0:
                    emotion_2 = "angry"
                elif index_2 == 1:
                    emotion_2 = "disgust"
                elif index_2 == 2:
                    emotion_2 = "fear"
                elif index_2 == 3:
                    emotion_2 = "happy"
                elif index_2 == 4:
                    emotion_2 = "sad"
                elif index_2 == 5:
                    emotion_2 = "surprise"
                elif index_2 == 6:
                    emotion_2 = "neutral"

                # draw result
                cv2.putText(npImage, emotion_1, (xmin, ymin - 35), 0, 5e-3 * 150, (0, 255, 0), 2)
                cv2.putText(npImage, emotion_2, (xmin, ymin - 10), 0, 5e-3 * 150, (0, 255, 0), 2)
                cv2.rectangle(npImage, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # end timing
            time_end = time.time()
            time_used = time_end - time_start
            if time_used != 0:
                fps = (fps + (1. / time_used)) / 2
            print("fps = %f" % (fps))

            # show and store modified frame
            if exportVideo:
                video_writer.write(npImage)
            cv2.imshow('GooseBusters', npImage)

            # Press "q" to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # release video captured
        video_capture.release()
        if exportVideo:
            video_writer.release()
        print("Video released!")
        print("Done!")
        sys.exit(0)


if __name__ == '__main__':
    main(runDetector = True,
         trainModel = False,
         testModel = False,
         usingExistModel = True,
         exportVideo = False,
         num_classes = 7,
         batch_size = 256,
         epochs = 4,
         testModelImagePath = "test1.jpg",
         datasetLoadPath = '../Data/Facial_Expression_Datasets/fer2013/fer2013.csv',
         modelLoadPath = '../Data/Facial_Expression_Models/model_3/facial_expression_model_weights.h5',
         modelSavePath = '../Data/Facial_Expression_Models/model_4/facial_expression_model_weights.h5',
         videoSavePath = '../Data/Output_Videos/output02.mp4')





