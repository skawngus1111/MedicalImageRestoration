import argparse

import torch
import torch.fx  # 반드시 timm import보다 먼저

from MIRExperiment.medical_image_restoration import MedicalImageRestoration

def main(args):
    print("Hello! We start experiment for Medical Image Restoration!")

    medical_image_restoration_experiment = MedicalImageRestoration(args)
    if args.train: medical_image_restoration_experiment.fit()
    else: medical_image_restoration_experiment.inference()

    print("Done!!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')

    parser.add_argument('--data_path', type=str, default='/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/MIR_Dataset')
    parser.add_argument('--save_path', type=str, default='/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/MyExperiment')

    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--plot_inference', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--modality_list',  nargs='+') # PET CT MRI

    args = parser.parse_args()

    for modality_list in [["PET", "CT", "MRI"], ["PET"], ["CT"], ["MRI"]]:
    # for modality_list in [["PET", "CT", "MRI"], ["CT"]]:
        if len(modality_list) == 3: args.plot_inference = True
        else: args.plot_inference = False
        args.modality_list = modality_list
        main(args)