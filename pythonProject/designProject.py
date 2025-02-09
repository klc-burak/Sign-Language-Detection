import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Video akışının alınması
videoCapture = cv2.VideoCapture(0)

# Yalnızca tek bir elin tespit edilmesi sağlanır
handDetector = HandDetector(maxHands=1)

# Önceden eğitilmiş modelin yüklenmesi
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

# Gerekli değişkenlerin tanımlanması
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
folder = "data/z"
imageSize = 300
offset = 20
count = 0

while True:
    # Video akışının kare kare okunmasıyla elin tespit edilmesi
    success, image = videoCapture.read()
    imageOutput = image.copy()
    hands, image = handDetector.findHands(image)

    # Görüntünün dinamik olarak boyutlandırılması ve sınıflandırılması
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imageWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imageCrop = image[y - offset: y + h + offset, x - offset: x + w + offset]
        imageCropShape = imageCrop.shape
        aspectRatio = h/w

        if aspectRatio > 1:
            k = imageSize / h
            wCal = math.ceil(k*w)
            imageResize = cv2.resize(imageCrop, (wCal, imageSize))
            imageResizeShape = imageResize.shape
            wGap = math.ceil((imageSize - wCal) / 2)
            imageWhite[:, wGap:wCal + wGap] = imageResize
            prediction, index = classifier.getPrediction(imageWhite)
            print(prediction, index)

        else:
                k = imageSize / w
                hCal = math.ceil(k * h)
                imageResize = cv2.resize(imageCrop, (imageSize, hCal))
                imageResizeShape = imageResize.shape
                hGap = math.ceil((imageSize - hCal) / 2)
                imageWhite[hGap:hCal + hGap, :] = imageResize
                prediction, index = classifier.getPrediction(imageWhite)

        cv2.putText(imageOutput, letters[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.imshow("ImageCrop", imageCrop)
        cv2.imshow("ImageWhite", imageWhite)

    cv2.imshow("Image", imageOutput)
    cv2.waitKey(1)