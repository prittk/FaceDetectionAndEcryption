import base64
import cv2
import numpy as np
import mediapipe as mp
import math
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes


# constantly used
distanceUsedForKey = 0
leftUsedForKey = 0
rightUsedForKey = 0
# key saved to ecrypt
encryptDist = distanceUsedForKey
encryptLeft = leftUsedForKey
encryptRight = rightUsedForKey
hashedPlainText=""

# Text to be encrypted and deciphered
plaintext = "Hello World"
ciphertext = ""
cipherDistAES = 0

decrypt = False


def openWebCam():
    global decrypt
    webCam = cv2.VideoCapture(0)  # open webcame
    detectFace = cv2.CascadeClassifier('../../Downloads/haarcascade_frontalface_default.xml')
    cv2.namedWindow('frame')
    switch = 'Encrypt 0 : OFF \n1 : ON'
    switch2 = 'Decipher 0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'frame', 0, 1,
                       encryptLandmarks)  # cv2 doesnt have button support through pip install, for ease of use for everyone i made a switch trackbar
    cv2.createTrackbar(switch2, 'frame', 0, 1, startDecrypt)
    faces = []  # stores the unique data of the face
    i = 0
    while True:
        frame = detectFaces(webCam, detectFace, faces)

        text_pane = np.zeros((100, frame.shape[1], 3), dtype=np.uint8)

        cv2.putText(text_pane, plaintext, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined_frame = np.vstack((frame, text_pane))

        if decrypt:
            print("Trying to decrypt")
            decryptLandmarks()

        cv2.imshow("frame", combined_frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    webCam.release()
    cv2.destroyAllWindows()


def detectFaces(webCam, detectFace, facedata):
    i = 0

    ret, frame = webCam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detectFace.detectMultiScale(gray, 1.3, 5)

    # Detect facial landmarks

    for (x, y, w, h) in faces:
        getFaceData(x, y, w, h, gray, frame)

        crop_img = frame[y:y + h, x:x + w,
                   :]  # select region of x to x+h row, y to y+h column, and all of z color channel
        resized_img = cv2.resize(crop_img, (50, 50))
        # add face to facedata
        if len(facedata) <= 100 and i % 10 == 0:
            facedata.append(resized_img)
        i += i
        cv2.putText(frame, str(len(facedata)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 50, 255),
                    1)  # put a red box around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    return frame

    # end face detection with the q key


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)


def getFaceData(x, y, w, h, gray, frame):

    global distanceUsedForKey, leftUsedForKey, rightUsedForKey

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            #Predict Landmark for facial features of users
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

            # Draws a circle where the landmarks are.
            for idx, (x_lm, y_lm) in enumerate(landmarks):
                cv2.circle(frame, (x_lm, y_lm), 1, (0, 255, 0), -1)
            #This defines the key features including the left, right, and between_eyes
            left_eye = (landmarks[33], landmarks[133])
            right_eye = (landmarks[362], landmarks[263])
            between_eyes = (landmarks[133], landmarks[362])

            #Normalizes the face width vector
            face_width = np.linalg.norm(
                np.array(landmarks[234]) - np.array(landmarks[454]))
            left = distanceCalc(*left_eye) / face_width
            right = distanceCalc(*right_eye) / face_width
            distanceBetween = distanceCalc(*between_eyes) / face_width

            print(f"Left eye width {left}, Right eyes width {right}")

            distanceUsedForKey = distanceBetween
            leftUsedForKey = left
            rightUsedForKey = right


def drawPoints(image, faceLandmarks, startpoint, endpoint, isClosed=False):
    points = []
    for i in range(startpoint, endpoint + 1):
        point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

    ##calculate distance


def distanceCalc(point1, point2):
    return math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))  # sqrt((x2-x1)**2 + (y2-y1)**2)


def getEyeWidth(landmarks):
    ##left eye left point == 37 left eye right point == 40
    # https://www.studytonight.com/post/dlib-68-points-face-landmark-detection-with-opencv-and-python
    leftOuter = (landmarks.part(36).x, landmarks.part(36).y)  # 36 is the number for left eye outer
    leftInner = (landmarks.part(39).x, landmarks.part(39).y)  # 39 inner
    ##right eye left point == 43   right eyes right point == 46
    rightInner = (landmarks.part(42).x, landmarks.part(42).y)
    rightOuter = (landmarks.part(45).x, landmarks.part(45).y)

    face_width = math.sqrt((landmarks.part(0).x - landmarks.part(
        16).x) ** 2)  # Width across cheeks used for normalizing the image so distance doesnt affect result
    leftWidth = distanceCalc(leftOuter, leftInner) / face_width  # left eye width fixed fro distance
    rightWidth = distanceCalc(rightInner, rightOuter) / face_width  # right eye width

    ##find distasnce between eye inners
    distanceBetweenBoth = distanceCalc(leftInner, rightInner) / face_width
    print(f"Distance between inner cornias = {distanceBetweenBoth}")

    return leftWidth, rightWidth, distanceBetweenBoth


def encryptLandmarks(x):
    global plaintext  # access the global
    global ciphertext
    global cipherDistAES
    global encryptDist
    global saved_iv
    global firstBitskeyDist
    global hashedPlainText


    encryptDist = distanceUsedForKey
    encryptLeft = leftUsedForKey
    encryptRight = rightUsedForKey

    keyDist = str(encryptDist)
    keyLeft = str(encryptLeft)
    keyRight = str(encryptRight)

    hashedKeyDist = hashlib.sha256(keyDist.encode()).digest()
    hashedLeft = hashlib.sha256(keyLeft.encode()).digest()
    hashedRight = hashlib.sha256(keyRight.encode()).digest()
    # get 32 bytes
    firstBitskeyDist = hashedKeyDist[:32]
    firstBitsLeft = hashedLeft[:32]  # 0 - 31
    firstBitsRight = hashedRight[:32]

    cipherDistAES = AES.new(firstBitskeyDist, AES.MODE_CTR)
    saved_iv=cipherDistAES.nonce
    cipherLeftAES = AES.new(firstBitsLeft, AES.MODE_CTR)
    cipherRightAES = AES.new(firstBitsRight, AES.MODE_CTR)
    plaintext_bytes = plaintext.encode()  # have to make string byte form to encrypt

    ciphertext = cipherDistAES.encrypt(plaintext_bytes)
    ciphertext_b64 = base64.b64encode(ciphertext).decode()
    plaintext = ciphertext_b64
    hashedPlainText = plaintext

    print(f"Hashed key Distance= {hashedKeyDist}")


def decryptLandmarks():
    global cipherDistAES
    global ciphertext
    global encryptDist
    global distanceUsedForKey
    global firstBitskeyDist
    global plaintext, hashedPlainText
    global saved_iv
    global decrypt


    # 10 % greater or less match to decrypt
    print(f"Distance key {distanceUsedForKey}")
    print(f"Encrypted Distance {encryptDist}")
    encryptDist=float(encryptDist)
    print(f"encrypt distance less than 10% {encryptDist- (encryptDist*.01)}, distanceUsedForKey {distanceUsedForKey}, encrypt distance greater than 10% {encryptDist*.01 + encryptDist}")



    if (encryptDist - (encryptDist * .01)) <= distanceUsedForKey <= ((encryptDist * .01) + encryptDist):

        decrypt_cipher = AES.new(firstBitskeyDist, AES.MODE_CTR,nonce=saved_iv)

        decrypted_bytes = decrypt_cipher.decrypt(ciphertext)

        print("Decrypted Data:",decrypted_bytes)

        plaintext = decrypted_bytes.decode()
        print(plaintext)
    else:
        print("back to encrypting")
        plaintext = hashedPlainText



def startDecrypt(x):
    global decrypt
    decrypt = True


openWebCam()
