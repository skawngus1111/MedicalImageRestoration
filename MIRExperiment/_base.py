from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.get_functions import get_device
from model_list import medical_image_restoration_model
from dataset.medical_image_restoration_dataset import (
    MedicalImageRestorationTrainDataset,
    MedicalImageRestorationTestDataset,
    DataSampler)

class BaseExperiment(object):
    def __init__(self, args):
        super(BaseExperiment, self).__init__()

        self.args = args
        self.args.device = get_device()
        self.args.total_iteration = 200000
        self.args.val_iteration = 1000
        self.args.batch_size = 4
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_sampler, self.valid_loader = self.dataloader_generator()

        self.model = medical_image_restoration_model(self.args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-08)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.args.total_iteration, eta_min=1.0e-6)

    def forward(self, data_batch):
        data_batch = self.cpu_to_gpu(data_batch)
        ctx = torch.cuda.amp.autocast() if self.args.amp else nullcontext()
        with ctx: return self.model(data_batch)

    def backward(self, loss):
        self.optimizer.zero_grad()

        if self.args.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

    def cpu_to_gpu(self, data):
        dev = self.args.device if isinstance(self.args.device, torch.device) else torch.device(str(self.args.device))
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(dev, non_blocking=True)
        return data

    def dataloader_generator(self, shuffle=True):
        train_dataset = MedicalImageRestorationTrainDataset(root_dir=self.args.data_path, modality_list=self.args.modality_list)
        valid_dataset = MedicalImageRestorationTestDataset(root_dir=self.args.data_path, modality_list=self.args.modality_list, use_num=32)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=shuffle, drop_last=True, num_workers=self.args.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

        train_sampler = DataSampler(train_loader)

        return train_sampler, valid_loader
