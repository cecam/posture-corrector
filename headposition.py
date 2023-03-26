import pathlib
import cv2
from playsound import playsound

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

video_capture = cv2.VideoCapture(0)

def reproduceSound(): 
    playsound('audio/canto_angelical.mp3')

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # get vcap property 
    width  = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH )   
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT )

    print(width, height)

    # Draw a rectangle around the faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frames, (x, y), (x+width, y+height), (255, 255, 0) if y  < 10 else (0, 0, 255), 2)
        cv2.putText(frames, str({x,y}), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        if width > 400:
            cv2.putText(frames, str({'ancho', width}), (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            reproduceSound()

    # Display the resulting frame
    cv2.imshow('Video', frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()


