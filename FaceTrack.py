import cv2
import numpy as np
import dlib
import math




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
    
        # Draw landmarks on the frame of a circle for expert part detected
        for n in range(68):  # 68 landmarks
            x_landmark = landmarks.part(n).x
            y_landmark = landmarks.part(n).y
            cv2.circle(frame, (x_landmark, y_landmark), 2, (0, 255, 0), -1)
            
        left, right =getEyeWidth(landmarks)
        distanceBetweenEyes = distanceCalc(right, left)
        print(distanceBetweenEyes)

def distanceCalc(point1,point2):
    return math.sqrt(((point1[0] - point2[0])**2) + ((point1[1] - point2[1])**2)) #sqrt((x2-x1)**2 + (y2-y1)**2)

def getEyeWidth(landmarks):
    leftOuter = (landmarks.part(36).x, landmarks.part(36).y)
    leftInner = (landmarks.part(39).x, landmarks.part(39).y)
    
    rightOuter = (landmarks.part(42).x, landmarks.part(42).y)
    rightInner = (landmarks.part(45).x, landmarks.part(45).y)
    
    leftWidth = distanceCalc(leftOuter,leftInner)
    rightWidth = distanceCalc(rightOuter,rightInner)

    return leftWidth, rightWidth


def getDistanceBetweenEyes(landmarks):
    leftCenter = ((landmarks.part(36).x, landmarks.part(39).y)//2, (landmarks.part(36).y+ landmarks.part(39).y)//2) ##// for floor (whole number) division
    
    rightCenter = ((landmarks.part(42).x, landmarks.part(45).x)//2, (landmarks.part(42).y, landmarks.part(45).y)//2)
    
    distance =distanceCalc(leftCenter, rightCenter)
    
    return distance
    

            
openWebCam()

