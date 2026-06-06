import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Show current working directory
print("File will be saved in:", os.getcwd())

# Start webcam
video_capture = cv2.VideoCapture(0)

# -------- LOAD IMAGES --------

divya_image = face_recognition.load_image_file(r"C:\Users\rohit\Pictures\eshu\divya.jpeg")
divya_encoding = face_recognition.face_encodings(divya_image)[0]

mansi_image = face_recognition.load_image_file(r"C:\Users\rohit\Pictures\eshu\mansi.jpeg")
mansi_encoding = face_recognition.face_encodings(mansi_image)[0]

known_face_encodings = [divya_encoding, mansi_encoding]
known_face_names = ["Divya","Mansi"]

students = known_face_names.copy()

# -------- CREATE CSV FILE --------

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

file_path = r"C:\Users\rohit\Desktop\\" + current_date + ".csv"

f = open(file_path,"w",newline="")
lnwriter = csv.writer(f)

lnwriter.writerow(["Name","Time"])

print("CSV file created at:", file_path)

# -------- CAMERA LOOP --------

while True:

    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)

        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:

            name = known_face_names[best_match_index]

            cv2.putText(frame,f"{name} Present",(20,100),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            if name in students:

                students.remove(name)
                current_time = datetime.now().strftime("%H:%M:%S")
                lnwriter.writerow([name,current_time])
                print(name,"attendance saved")

    cv2.imshow("Camera",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()

print("Attendance file saved successfully")