import cv2
import numpy as np
import dlib
from PIL import Image

cap=cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,360)
cap.set(10,100)
detector =dlib.get_frontal_face_detector()
predictor= dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

prop1=Image.open('props/mask.png')
caps=Image.open('props/hat.png')

cascPath = "haarcascade_frontalface_default.xml"

# cascade classifier object
faceCascade = cv2.CascadeClassifier(cascPath)


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def resizetofit(prop_source,main_img):
    gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)

# detect faces in grayscale image
    faces = faceCascade.detectMultiScale(gray, 1.15)

# convert cv2 imageto PIL image
    background = Image.fromarray(main_img)

    for (x,y,w,h) in faces:

        resized_mask = prop_source.resize((w+(w-int(w/3)),h+(h-int(w/3))), Image.ANTIALIAS)
        resized_cap=caps.resize((w+10,int(h)), Image.ANTIALIAS)
        offset = (x-int((w-int(w/3))/2),y+int((h-int(h/3))/2.8))
        caps_offset=(x,y-int(h/1.5))
        background.paste(resized_cap, caps_offset, resized_cap)
        background.paste(resized_mask, offset, resized_mask)

    return np.asarray(background)




while True:
    flag,img=cap.read()
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    d_face=img.copy()
    p_face = img.copy()
    m_face= img.copy()
    for face in faces:
        x1,y1,x2,y2=face.left(),face.top(),face.right(),face.bottom()
        cv2.rectangle(d_face,(x1,y1),(x2,y2),(0,0,255),2)
        landmarks= predictor(gray,face)
        for n in range(0,68):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            cv2.circle(p_face,(x,y),1,(0,255,0),3)
            if n==0:
                p1=(x,y)
            elif n==16:
                p2=(x,y)
            elif n==14:
                p3=(x,y)
            elif n==2:
                p4=(x,y)
            elif n==36:
                p5=(x,y)
            elif n==45:
                p6=(x,y)

        p_final=resizetofit(prop1,m_face)
        # cv2.line(m_face,p1,p2,(255,0,0),1)
        # cv2.line(m_face,p1,p4,(255, 0, 0),1)
        # cv2.line(m_face,p2,p3,(255, 0, 0),1)
        # cv2.line(m_face, p3, p4, (255, 0, 0), 1)
        # cv2.line(m_face, p5, p6, (0, 0, 255), 2)


    stack = stackImages(.8, ([img, d_face],[p_face,p_final]))

    cv2.imshow("Output", stack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break