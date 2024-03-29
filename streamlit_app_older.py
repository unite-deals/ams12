import streamlit as st
import cv2
import os
import numpy as np
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import time
import queue 

hide_github_link_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visiblity: hidden;}
    header {visibility: hidden;}
        .viewerBadge_container__1QSob {
            display: none !important;
        }
    </style>
"""
st.markdown(hide_github_link_style, unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
model_path = 'static/face_recognition_model.pkl'
# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img_rgb = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def set_timezone():
    current_local_time = datetime.now()
    timezone_js = """
        <script>
            var timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
            var data = { timezone: timezone };
            fetch("/set_timezone", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            });
        </script>
    """
    st.markdown(timezone_js, unsafe_allow_html=True)

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# extract the face from an image
def extract_faces(img):
    if img != []:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray.reshape(1, -1))

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')

    # Check if there are users with images
    if not userlist:
        print("No users found for training.")
        return

    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    faces = np.array(faces)

    # Check if faces array is not empty
    if faces.size == 0:
        print("No faces found for training.")
        return

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

# Streamlit app
def main():
    st.title("Attendance Management System")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Home", "Take Attendance", "Add Student"])

    if page == "Home":
        home_page()
    elif page == "Take Attendance":
        take_attendance_page()
    elif page == "Add Student":
        add_student_page()

def home_page():
    names, rolls, times, l = extract_attendance()
    st.write(f"## Today's Attendance ({datetoday2})")
    df_home = pd.DataFrame({"Name": names, "Roll": rolls, "Time": times})
    st.table(df_home)

    st.write(f"Total Registered Students: {totalreg()}")

def take_attendance_page():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        st.warning("There is no trained model in the static folder. Please add a new face to continue.")
        return

    st.write("## Taking Attendance from Live Video Streaming")

    webrtc_ctx = webrtc_streamer(
        key="video",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceDetectionProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    if webrtc_ctx.video_receiver:
        st.video(webrtc_ctx.video_receiver)
    else:
        st.warning("Failed to create video receiver. Please check your webcam connection.")


def add_student_page():
    st.title("Capture Images for New Student")
    newusername = st.text_input('Enter new username:')
    newuserid = st.text_input('Enter new user ID:')
    userimagefolder = 'static/faces/' + newusername + '_' + 'ID:' + str(newuserid)

    # Check if the user folder already exists
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    image_place = st.empty()
    i = 0
    
    webrtc_ctx = webrtc_streamer(
    key="video-sendonly",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True},
    video_receiver_size=3  # Set a larger receiver size, adjust as needed
)

    try:
        while i < 3:
            if webrtc_ctx.video_receiver:
                try:
                    video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
                except queue.Empty:
                    st.warning("Queue is empty. Abort.")
                    break

                img_rgb = video_frame.to_ndarray(format="rgb24")
                image_place.image(img_rgb)
                image_array = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # Convert the frame to grayscale
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

                # Detect faces in the grayscale frame
                faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                # Draw rectangles around the detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image_array, f'Images Captured: {i + 1}/10', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 20), 2, cv2.LINE_AA)

                    # Save the captured image
                    name = f'{newusername}_{i}.jpg'
                    cv2.imwrite(os.path.join(userimagefolder, name), image_array[y:y + h, x:x + w])

                # Display the resulting frame
                st.image(image_array, channels="BGR", use_column_width=True)

                # Increment the counter
                i += 1

            # Sleep for a short duration to prevent capturing too quickly
            time.sleep(1)

    except Exception as e:
        st.warning(f"An error occurred: {e}")

    finally:
        # Release the VideoCapture object
        cap.release()

    # Train the model after capturing images
    train_model()

    st.success("Images Captured. Training the model...")
    st.success("Training complete. New student added.")
if __name__ == "__main__":
    main()