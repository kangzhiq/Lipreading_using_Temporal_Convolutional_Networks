import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F

dir_path = "../datasets/LRW_h96w96_mouth_crop_gray_frontalized"
label = "ABOUT"
num = 44
file_path = os.path.join(dir_path, label, "test", "{}_{:05d}.npz".format(label, num))
loaded = np.load(file_path, allow_pickle=True)['data']
print(loaded.shape)
#fig, ax = plt.subplots(1, 1)
#ax.imshow(loaded[15, :, :].astype(int))

for i in range(loaded.shape[0]):
    img = loaded[i, :, :]
    cv2.imwrite("temp/frame{:03d}.jpg".format(i), img)

# Test with data
def get_model():
    args_loaded = load_json( args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                    'kernel_size': args_loaded['tcn_kernel_size'],
                    'dropout': args_loaded['tcn_dropout'],
                    'dwpw': args_loaded['tcn_dwpw'],
                    'width_mult': args_loaded['tcn_width_mult'],
                  }

    return Lipreading( modality=args.modality,
                       num_classes=args.num_classes,
                       tcn_options=tcn_options,
                       backbone_type=args.backbone_type,
                       relu_type=args.relu_type,
                       width_mult=args.width_mult,
                       extract_feats=args.extract_feats).cuda()

model = get_model()
model.load_state_dict( torch.load(args.model_path)["model_state_dict"], strict=True)
