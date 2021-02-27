import face_alignment
import os
import cv2
import registration_utils as regut
import numpy as np
from BFM.morphable_model import MorphabelModel
from tqdm import tqdm
from numpy import linalg as LA
# Some constants
bfm = MorphabelModel('../../lip_frontalization/BFM/data/BFM.mat')
sp = bfm.get_shape_para('zero')
ep = bfm.get_exp_para('zero')
vertices = bfm.generate_vertices(sp, ep)

generic_model = vertices[bfm.kpt_ind].copy()
# Flipping the model to turn it to the standard coordiante system
generic_model[:, 1:] = - generic_model[:, 1:]
# Frontal reference model
LM_ref = generic_model.copy()

device = 'cpu'

def get_head_pose(frame, fa):
    maxiter =50
    lm = fa.get_landmarks(frame)[0]
    lm[:, 2] = - lm[:, 2]
    R_init = np.eye(3)
    t_init = np.zeros((3, 1))
    s_init = 1
    R, s, Sig_in, T, w = regut.robust_Student_reg(LM_ref.transpose(), lm.transpose(), R_init, s_init, t_init, maxiter)

    angle_R = np.arccos((R.trace()-1)/2)
    angle_R = angle_R/np.pi * 180

    return angle_R
for label in ['AFFECTED', 'AFRICA', 'AFTER', 'AFTERNOON', 'AGAIN', 'AGAINST']:
    with open("head_pose/{}.txt".format(label), "w") as file:
        for ind_vid in range(1, 51):
            video_path = "../datasets/LRW/{}/test/{}_{:05d}.mp4".format(label, label,ind_vid)
            frames = []
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=device)

            vidcap = cv2.VideoCapture(video_path)
            count = 0
            success, image = vidcap.read()
            while success:
                frames.append(image)
                success, image = vidcap.read()
                count += 1
            pose_range_min = -1
            pose_range_max = -1
            for i in tqdm(range(0, len(frames), 6)):
                angle = np.absolute(get_head_pose(frames[i], fa))
                min_cur = angle // 30
                max_cur = min_cur + 1
                if pose_range_min == -1:
                    pose_range_min = min_cur
                    pose_range_max = max_cur
                else:
                    pose_range_min = min(pose_range_min, min_cur)
                    pose_range_max = max(pose_range_max, max_cur)

            if pose_range_max - pose_range_min == 1:
                head_pose_type = pose_range_min
            elif pose_range_max - pose_range_min == 2:
                head_pose_type = pose_range_min + 3
            else:
                head_pose_type = 5
            print("Video", ind_vid, ": ", head_pose_type)

            file.write("{}\n".format(head_pose_type))
