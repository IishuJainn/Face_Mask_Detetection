import pickle
model=pickle.load(open('model1.pkl','rb'))
def detect_face_mask(img):
    y_pred = model.predict(img.reshape(1,224,224,3))
    return y_pred

def draw_label(img,text,pos,bg_color):
    text_size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
    end_x=pos[0] + text_size[0][0] + 2
    end_y=pos[1] + text_size[0][1] - 2
    cv2.rectangle(img, pos, (end_x,end_y),bg_color, cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

def detect_face(img):
    haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cood = haar.detectMultiScale(img)
    return cood

import cv2
cap=cv2.VideoCapture(0)
while True:
    ret, frame =cap.read()
    frame = cv2.flip(frame, 1)
    coods =detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    for x,y,w,h in coods:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
    img=cv2.resize(frame,(224,224))
    y_pred =detect_face_mask(img)
    print(y_pred)
    if y_pred < 0.001:
        draw_label(frame,"Mask",(30,30),(0,255,0))
    else:
        draw_label(frame,"No Mask", (30, 30), (0,0, 255))
    cv2.imshow("window",frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows()