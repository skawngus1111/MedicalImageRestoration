import os

import torch

def save_model(G_net_model, save_dir, optimizer_G=None, ex=""):
    save_path=os.path.join(save_dir, "model_weights")
    G_save_path = os.path.join(save_path,'Generator{}.pth'.format(ex))
    torch.save(G_net_model.state_dict(), G_save_path)

    if optimizer_G is not None:
        opt_G_save_path = os.path.join(save_path,'Optimizer{}.pth'.format(ex))
        torch.save(optimizer_G.state_dict(), opt_G_save_path)