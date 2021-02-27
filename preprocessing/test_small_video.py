import cv2
import numpy as np


path = "datasets/LRW/test/test/ABOUT_00001.mp4"
frontal_frame = []
save_path = "datasets/LRW/test/test/test_00001.mp4"

vidcap = cv2.VideoCapture(path)
count = 0
success, image = vidcap.read()
while success:
    frontal_frame.append(image)
    success, image = vidcap.read()
    count += 1

frontal_frame = np.array(frontal_frame)
frontal_frame.shape


# convert 256x256 to 200x200
frames = []
for i in range(len(frontal_frame)):
    img = frontal_frame[i]
    frames.append(cv2.resize(img, (200, 200)))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_path, fourcc, 25.0, (200, 200))

for i in range(len(frames)):
    out.write(np.uint8(frames[i]))
out.release()