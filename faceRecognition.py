import face_recognition
import cv2, os, mysql.connector
import numpy as np
def takePassport(self):
        faceCascade = cv2.CascadeClassifier(self.cascPath)
        video_capture = cv2.VideoCapture(0)
        while (True):
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                ret, frame = video_capture.read()
                cv2.imshow("Captured", frame)
                cv2.imwrite(filename="knownImage\\"+self.fName.get()+".jpg", img=frame)
                self.passval.set(os.path.join(os.path.dirname(os.path.abspath("knownimage")),self.fName.get()+".jpg"))
                cv2.waitKey(1650)
                break
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

def identifyImage(self, visit, amount):
        self.known_face_names = []
        self.known_face_image = []
        self.known_face_encodings = []
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "knownImage")
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root, file)
                    # Create arrays of known face encodings and their names
                    self.known_face_names.append(os.path.basename(path).split(".")[0])
                    self.known_face_image.append(path)

        # Get a reference to webcam #0 (the default one)
        video_capture = cv2.VideoCapture(0)
        # Load a sample picture and learn how to recognize it.
        for face_image in self.known_face_image:
            _image = face_recognition.load_image_file(face_image)
            # Create arrays of known face encodings and their names
            self.known_face_encodings.append(face_recognition.face_encodings(_image)[0])

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        name = ""
        process_this_frame = True
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    face_names.append(name)
            process_this_frame = not process_this_frame
            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # Display the resulting image
            cv2.imshow('Video', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()