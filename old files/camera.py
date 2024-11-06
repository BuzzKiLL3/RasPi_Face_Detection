import cv2
import cProfile
# " cv2.VideoCapture() " IS A cv2 FUNCTION OPENING A VIDEO CAPTURE FOR VIDEO
cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # 
cap.set(3, 280)
cap.set(4, 464)
armed = False


def cycle(frame):
    armed = False
    # FINDS FACES ON IMAGES
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cv2.imwrite("om.jpg", frame)
    img = cv2.imread('om.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if len(faces) == 0:
        cv2.imshow('Om', img)
        # GIVES POSITIONS ON WHERE TO DRAW A BOX AND LINES AROUND FACE
    else:
        (x, y, w, h) = faces[0]
        left = x
        bottom = y
        right = x + w
        top = y + h
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        # Display the resulting frame
        height = frame.shape[0]
        width = frame.shape[1]
        screen_center_x = int(width / 2)
        screen_center_y = int(height / 2)
        box_center_x = int((left + right) / 2)
        box_center_y = int((top + bottom) / 2)
        vector_x = screen_center_x - box_center_x
        vector_y = screen_center_y - box_center_y
        x_adjust = screen_center_x - box_center_x
        y_adjust = screen_center_y - box_center_y
        distance = ((screen_center_x - box_center_x) ** 2 + (screen_center_y - box_center_y) ** 2) ** 0.5
        print(distance)
        if distance <= 15:
            armed = True
        # IF FACE IS FOUND DRAWS BOX AND LINES
        if armed:
            pass
            cv2.line(img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.line(img, (left, bottom), (right, top), (0, 0, 255), 2)
            cv2.line(img, (box_center_x, top), (box_center_x, bottom), (0, 0, 255), 3)
            cv2.line(img, (left, box_center_y), (right, box_center_y), (0, 0, 255), 3)
            cv2.line(img, (box_center_x, box_center_y), (screen_center_x, screen_center_y), (0, 0, 255), 2)
            cv2.imshow('om', img)
        else:
            cv2.imshow('om', img)


def main():
    # EXITS PROGRAM IF PRESSED : q
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cycle(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


cProfile.run('main()')
