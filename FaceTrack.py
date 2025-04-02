import cv2
import numpy as np
import dlib
import math
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad




def openWebCam():
    webCam = cv2.VideoCapture(0) #open webcame
    detectFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    faces = [] #stores the unique data of the face
    i = 0
    while True:
        frame = detectFaces(webCam,detectFace,faces)
        
        text_pane = np.zeros((100, frame.shape[1],3),dtype = np.uint8)
        
        cv2.putText(text_pane, "This is the text pane!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        combined_frame = np.vstack((frame, text_pane))
        
        cv2.imshow("Frame", combined_frame)
    
        k = cv2.waitKey(1)
        if k== ord('q'):
            break
        
    webCam.release()
    cv2.destroyAllWindows()
    
    
def detectFaces(webCam, detectFace, facedata):
    i = 0
     
    ret, frame = webCam.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = detectFace.detectMultiScale(gray, 1.3, 5)
    
    # Detect facial landmarks
    
    
    for (x,y,w,h) in faces:
        getFaceData(x,y,w,h,gray,frame)

        crop_img =frame[y:y+h, x:x+w, :] #select region of x to x+h row, y to y+h column, and all of z color channel
        resized_img = cv2.resize(crop_img, (50,50))
        #add face to facedata 
        if len(facedata) <= 100 and i% 10 == 0:
            facedata.append(resized_img)
        i += i
        cv2.putText(frame,str(len(facedata)), (50,50), cv2.FONT_HERSHEY_COMPLEX,1, (0,50,255),1) #put a red box around detected face
        cv2.rectangle(frame,(x,y), (x+w,y+h), (50,50,255),1)
    
    return frame
    
        #end face detection with the q key
        
def getFaceData(x,y,w,h,gray,frame):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = predictor(gray, rect) #get unique featers from the face
        
        faces = detector(gray)
    
        # Draw landmarks on the frame of a circle for expected part detected
        for n in range(68):  # 68 landmarks
            x_landmark_leftEye = landmarks.part(n).x
            y_landmark_rightEye = landmarks.part(n).y
            
            
            cv2.circle(frame, (x_landmark_leftEye, y_landmark_rightEye), 2, (0, 255, 0), -1)
          
        left, right, distanceBetween =getEyeWidth(landmarks)
        widthOfEye = [left,right]
        print(f"Left eye width {widthOfEye[0]}, Right eyes width {widthOfEye[1]}")
        
        
        distanceUsedForKey = distanceBetween #TODO: Use this to decrypt by saying if distance is within 10%
        
        key = str(distanceBetween)
        hashedKey = hashlib.sha256(key.encode()).digest()
        
        FirstBitskey = hashedKey[:32]
        cipher = AES.new(FirstBitskey, AES.MODE_CBC)
        print(f"Hashed key = {hashedKey}")
        
        

        
        
       
  
       
def distanceCalc(point1,point2):
    
    return math.sqrt(((point1[0] - point2[0])**2) + ((point1[1] - point2[1])**2)) #sqrt((x2-x1)**2 + (y2-y1)**2)


def getEyeWidth(landmarks):
    ##left eye left point == 37 left eye right point == 40
    #https://www.studytonight.com/post/dlib-68-points-face-landmark-detection-with-opencv-and-python
    leftOuter = (landmarks.part(37).x, landmarks.part(37).y) # 36 is the number for left eye outer
    leftInner = (landmarks.part(40).x, landmarks.part(40).y) #39 inner
    ##right eye left point == 43   right eyes right point == 46
    rightInner = (landmarks.part(43).x, landmarks.part(43).y)
    rightOuter = (landmarks.part(46).x, landmarks.part(46).y)
    
    
    face_width = math.sqrt((landmarks.part(1).x - landmarks.part(17).x)**2)  # Width across cheeks used for normalizing the image so distance doesnt affect result

    
    leftWidth = distanceCalc(leftOuter,leftInner)/face_width
    rightWidth = distanceCalc(rightInner,rightOuter)/face_width
    
    
    ##find distasnce between eye inners
    distanceBetweenBoth = distanceCalc(leftInner,rightInner)/face_width
    print(f"Distance between inner cornias = {distanceBetweenBoth}")
    

    return leftWidth, rightWidth, distanceBetweenBoth



            
openWebCam()

