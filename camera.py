import cv2
# faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        alg = "haarcascade_frontalface_default.xml"
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + alg)
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(grayImg)
        ret,frame=self.video.read()
        
        for (x, y, w, h) in faces:   
            text = "Face Detected"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()