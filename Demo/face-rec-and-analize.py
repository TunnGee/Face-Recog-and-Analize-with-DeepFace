###### Step 1: Install Python libraries ######
#pip install deepface
#pip install --upgrade deepface

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt


###### Step 2: Define Titles for Faces ######
titles = ["Face 1"]   #Set your own Image Name


###### Step 3: Load and Display the Input Image ######
img_path = "image.png"
img = cv2.imread(img_path)
plt.imshow(img[:, :, ::-1])   #Convert BGR to RGB
plt.show()


###### Step 4: Extract Faces from the Image ######
faces = DeepFace.extract_faces(img_path, detector_backend='opencv')


###### Step 5: Display Each Detected Face with Title ######
for i, face in enumerate(faces):
    face_img = face["face"]
    plt.imshow(face_img[:, :, ::-1])
    title = titles[i] if i < len(titles) else f"Face {i+1}"
    plt.title(title)
    plt.show()


###### Step 6: Analyze Detected Faces for Emotions, Age, and Gender ######
objects = DeepFace.analyze(img_path)


###### Step 7: Print Analysis Results for Each Detected Face ######
for i, object in enumerate(objects):
    title = titles[i] if i < len(titles) else f"Face {i+1}"
    emotion = object["dominant_emotion"]
    age = object["age"]
    gender = object["gender"]

    print(f'{title}:')
    print(f'  Feeling: {emotion}')
    print(f'  Age: {age}')
    print(f'  Gender: {gender}')