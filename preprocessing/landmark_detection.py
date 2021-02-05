import face_alignment
import numpy as np
import cv2
from tqdm import tqdm
import os

print(os.getcwd())
video_path = "../datasets/s2_v2_u4_Trim.mp4"
init_frame = []
LMs = []

vidcap = cv2.VideoCapture(video_path)
count = 0
success, image = vidcap.read()
u_max, v_max = image.shape[:2]
while success:
    init_frame.append(image)
    success, image = vidcap.read()
    count += 1


# Extracting landmarks
# We assume only one person on each frame
#print("Extracting landmarks:")
#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')
#for i in tqdm(range(len(init_frame)), position=0, leave=True):
#    LMs.append(fa.get_landmarks(init_frame[i])[0])

#with open('../landmarks/LMs_test.npy', 'wb') as f:
#    np.save(f, LMs)

with open('../landmarks/s2_v2_u4_Trim.npy', 'rb') as f:
    LMs = np.load(f)

print(LMs.shape)

