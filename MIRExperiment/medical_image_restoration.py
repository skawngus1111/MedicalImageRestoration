import os

from torch.utils.data import DataLoader
from tqdm import tqdm

import torch

import numpy as np
import pandas as pd
import imageio.v2 as imageio

from MIRExperiment._base import BaseExperiment
from utils.misc import transformData, dataIO, save_png_vis_final, to_uint8_vis
from utils.calculate_metrics import compute_measure
from utils.get_functions import get_save_path
from utils.save_functions import save_model
from utils.load_functions import load_model
from dataset.medical_image_restoration_dataset import MedicalImageRestorationTestDataset

class MedicalImageRestoration(BaseExperiment):
    def __init__(self, args):
        super(MedicalImageRestoration, self).__init__(args)

        self.io = dataIO()
        self.transformData = transformData()
        self.eval_metrics = {
            "psnr": [],
            "ssim": [],
            "rmse": []
        }
        self.psnr_max = 0
        self.save_model_path, self.save_plot_path = get_save_path(self.args)

    def fit(self):
        if self.args.train:
            self.loss_list = []
            pbar = tqdm(total=int(self.args.total_iteration))

            print("################ Train ################")
            for iteration in list(range(1, int(self.args.total_iteration) + 1)):
                output_batch = self.train_iteration()

                if iteration % self.args.val_iteration == 0: self.valid_iteration(iteration)

                pbar.set_description("loss:{:6} | psnr:{:6}".format(output_batch['loss'].item(), self.eval_metrics['psnr'][-1] if len(self.eval_metrics['psnr']) > 0 else 0))
                pbar.update()
        print("################ Inference ##############")
        self.inference()

    def train_iteration(self):
        self.model.train()
        data_batch = next(self.train_sampler)
        output_batch = self.forward(data_batch)
        self.backward(output_batch['loss'])
        self.loss_list.append(output_batch['loss'].item())

        return output_batch

    def valid_iteration(self, iteration):
        self.model.eval()

        psnr, ssim, rmse = 0, 0, 0
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            for counter, data_batch in enumerate(tqdm(self.valid_loader)):
                output_batch = self.forward(data_batch)
                gen_img = self.transformData.denormalize(output_batch['prediction'], data_batch['modality'][0])
                v_label_pic = self.transformData.denormalize(data_batch['HQ_batch'], data_batch['modality'][0])

                gen_img = self.transformData.truncate_test(gen_img, data_batch['modality'][0])
                v_label_pic = self.transformData.truncate_test(v_label_pic, data_batch['modality'][0])

                data_range = v_label_pic.max() - v_label_pic.min()
                oneEval = compute_measure(gen_img, v_label_pic, data_range=data_range)

                psnr += oneEval[0]
                ssim += oneEval[1]
                rmse += oneEval[2]

                # self.io.save(gen_img.cpu().clone().numpy().squeeze(),
                #         os.path.join(self.save_model_path, "plot_results", "{}_{}.nii".format(data_batch['file_name'][0], data_batch['modality'][0])))

            c_psnr = psnr / (counter + 1)
            c_ssim = ssim / (counter + 1)
            c_rmse = rmse / (counter + 1)

            self.eval_metrics['psnr'].append(c_psnr)
            self.eval_metrics['ssim'].append(c_ssim)
            self.eval_metrics['rmse'].append(c_rmse)

            # save_model(G_net_model=self.model, save_dir=self.save_model_path, optimizer_G=None, ex="_iteration_{}".format(iteration))
            if c_psnr >= self.psnr_max:
                self.psnr_max = c_psnr
                self.io.save("Best Iteration: {}, PSNR: {}, SSIM:{}, RMSE:{}".format(iteration, c_psnr, c_ssim, c_rmse),
                        os.path.join(self.save_model_path, 'test_reports', "best.txt"))
                save_model(G_net_model=self.model, save_dir=self.save_model_path, optimizer_G=self.optimizer, ex="_best")

    def inference(self):
        self.model = load_model(self.args, self.model)
        self.model.eval()
        for modality_name in self.args.modality_list:
            self.plot_count = 0
            test_dataset = MedicalImageRestorationTestDataset(root_dir=self.args.data_path, modality_list=[modality_name], target_folder="test")
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers)

            psnr_list, ssim_list, rmse_list, name_list = [], [], [], []
            ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
            with ctx():
                for counter, data_batch in enumerate(tqdm(test_loader)):
                    output_batch = self.forward(data_batch)
                    gen_img = self.transformData.denormalize(output_batch['prediction'], data_batch['modality'][0])
                    v_label_pic = self.transformData.denormalize(data_batch['HQ_batch'], data_batch['modality'][0])

                    gen_img = self.transformData.truncate_test(gen_img, data_batch['modality'][0])
                    v_label_pic = self.transformData.truncate_test(v_label_pic, data_batch['modality'][0])

                    data_range = v_label_pic.max() - v_label_pic.min()
                    oneEval = compute_measure(gen_img, v_label_pic, data_range=data_range)

                    psnr_list.append(oneEval[0])
                    ssim_list.append(oneEval[1])
                    rmse_list.append(oneEval[2])
                    name_list.append(data_batch['file_name'][0])

                    # self.io.save(gen_img.cpu().clone().numpy().squeeze(),
                    #              os.path.join(self.save_plot_path, "plot_results", "test_results", "{}.nii".format(data_batch['file_name'][0])))

                    if self.args.plot_inference and self.plot_count < 100:
                        mod = data_batch['modality'][0]
                        name = data_batch['file_name'][0]

                        pred = gen_img
                        gt = v_label_pic

                        base_dir = os.path.join(self.save_plot_path, "plot_results", "test_results_png", mod)
                        pred_dir = os.path.join(base_dir, "pred")
                        cmp_dir = os.path.join(base_dir, "compare")

                        os.makedirs(pred_dir, exist_ok=True)
                        os.makedirs(cmp_dir, exist_ok=True)

                        pred_path = os.path.join(pred_dir, f"{name}.png")
                        cmp_path = os.path.join(cmp_dir, f"{name}_GT_PRED.png")  # Diff 안 넣으면 이름도 정리

                        # pred 저장
                        save_png_vis_final(pred, pred_path, mod, self.transformData)

                        # GT는 파일로 저장 안 하고 compare용으로만 사용
                        gt_np = gt.detach().float().cpu().numpy().squeeze()

                        # GT도 동일한 vis rule 적용
                        gt_vis_path = pred_path.replace(".png", "_gt_tmp.png")
                        save_png_vis_final(gt, gt_vis_path, mod, self.transformData)

                        pred_u8 = imageio.imread(pred_path)
                        gt_u8 = imageio.imread(gt_vis_path)

                        # GT | Pred
                        grid = np.concatenate([gt_u8, pred_u8], axis=1)
                        imageio.imwrite(cmp_path, grid)

                        os.remove(gt_vis_path)  # 임시 파일 제거

                        self.plot_count+=1

            psnr_list = np.array(psnr_list)
            ssim_list = np.array(ssim_list)
            rmse_list = np.array(rmse_list)
            name_list = np.array(name_list)

            c_psnr = psnr_list.mean()
            c_ssim = ssim_list.mean()
            c_rmse = rmse_list.mean()

            print(" ^^^Final Test  {}   psnr:{:.6}, ssim:{:.6}, rmse:{:.6} ".format(modality_name, c_psnr, c_ssim,
                                                                                    c_rmse))

            result_dict = {
                "NAME": name_list,
                "PSNR": psnr_list,
                "SSIM": ssim_list,
                "RMSE": rmse_list,
            }

            result = pd.DataFrame({key: pd.Series(value) for key, value in result_dict.items()})
            result.to_csv(os.path.join(self.save_model_path, "test_reports", "{}_result.csv".format(modality_name)))

            # --- add: save as TXT ---
            txt_path = os.path.join(self.save_model_path, "test_reports", f"{modality_name}_result.txt")

            with open(txt_path, "w", encoding="utf-8") as f:
                # summary
                f.write(f"Final Test {modality_name}\n")
                f.write(f"PSNR: {c_psnr:.6f}\n")
                f.write(f"SSIM: {c_ssim:.6f}\n")
                f.write(f"RMSE: {c_rmse:.6f}\n")
                f.write("\n")

                # per-sample table (tab-separated)
                f.write("NAME\tPSNR\tSSIM\tRMSE\n")
                for n, p, s, r in zip(name_list, psnr_list, ssim_list, rmse_list):
                    f.write(f"{n}\t{p:.6f}\t{s:.6f}\t{r:.6f}\n")