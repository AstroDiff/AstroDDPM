## Standard imports
import os
import json
import argparse
import torch
import tqdm
## Relative imports
from astroddpm.runners import config_from_id
from astroddpm.datahandler.dataset import get_dataset_and_dataloader
from astroddpm.diffusion.stochastic.sde import get_sde
from astroddpm.diffusion.power_spectra.powerspec_sampler import get_ps_sampler
from astroddpm.moment.models import TparamMomentNetwork, MomentModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, train_dataloader, epochs,test_dataloader = None, ckpt_path = None, ckpt_step = 5, scheduler = None):
    train_losses = []
    test_losses = []
    progress_bar = tqdm.tqdm(range(epochs))
    if model.conetwork is not None:
        for param in model.conetwork.parameters():
            param.requires_grad_(False)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            if len(batch.shape) == 3:
                batch = batch.unsqueeze(1)
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        if test_dataloader is not None:
            model.eval()
            test_loss = 0
            for batch in test_dataloader:
                if len(batch.shape) == 3:
                    batch = batch.unsqueeze(1)
                batch = batch.to(device)
                loss = model.loss(batch)
                test_loss += loss.item()
            test_loss /= len(test_dataloader)
            test_losses.append(test_loss)
        log = "Epoch {} | Train loss: {:2f} | Test loss: {:2f}".format(epoch, train_loss, test_loss)
        if scheduler is not None:
            scheduler.step()
        progress_bar.update(1)
        progress_bar.set_description(log)
        if ckpt_path is not None and (epoch % ckpt_step == 0):
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": model.config}
            torch.save(ckpt, os.path.join(ckpt_path, f"ckpt.pt"))
    if model.conetwork is not None:
        for param in model.conetwork.parameters():
            param.requires_grad_(True)
    return train_losses, test_losses

def main(args):
    diffuser_ID = args.diffuser_id
    config = config_from_id(diffuser_ID)
    dataloaders_config = config['dataloaders']
    power_spectrum_config = config['diffusion_model']['ps']
    sde_config = config['diffusion_model']['sde']
    _, _, train_dataloader, test_dataloader = get_dataset_and_dataloader(dataloaders_config)
    ps_sampler = get_ps_sampler(power_spectrum_config)
    sde = get_sde(sde_config)

    sde.beta_0 = args.beta_0
    sde.beta_T = args.beta_T
    sde.beta_schedule = args.beta_schedule
    CKPT_PATH = args.ckpt_path
    MODEL_ID = args.model_id
    CKPT_DIR = os.path.join(CKPT_PATH, MODEL_ID)
    os.makedirs(CKPT_DIR, exist_ok=True)

    #network = TparamMomentNetwork( in_channels = args.in_channels, dim_param = 2, in_size = 256, order = 1, num_blocks = 3, first_channels = 10, time_embed_dim = 100,padding_mode="circular", 
                #normalize="GN", group_c=1, skiprescale = True, discretization = "continuous", embedding_mode = None, n_steps = 1000,dropout=0.0)
    network = TparamMomentNetwork(in_channels=args.in_channels, dim_param=args.dim_param, in_size=args.in_size, order=args.order, num_blocks=args.num_blocks, first_channels=args.first_channels, time_embed_dim=args.time_embed_dim, padding_mode=args.padding_mode, normalize=args.normalize, group_c=args.group_c, skiprescale=args.skiprescale, discretization=args.discretization, embedding_mode=args.embedding_mode, n_steps=args.n_steps, dropout=args.dropout)
    network = network.to(device)
    model = MomentModel(network, sde, ps_sampler)
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    else:
        raise NotImplementedError
    
    model= model.to(device)

    train_losses, test_losses = train(model, optimizer, train_dataloader, epochs= args.epochs ,test_dataloader = test_dataloader, ckpt_path = CKPT_DIR, ckpt_step = 5)

    ## Save the losses in the same folder as the ckpt
    with open(os.path.join(CKPT_DIR, "losses.json"), "w") as f:
        json.dump({"train_losses": train_losses, "test_losses": test_losses}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a moment network')
    parser.add_argument('--model_id', type=str, default='MomentModel_ContinuousVPSDE_I_BPROJ_beta_0_0.01_beta_T_0.5_beta_schedule_cosine', help='model id')
    parser.add_argument('--diffuser_id', type=str, default='ContinuousSBM_ContinuousVPSDE_I_BPROJ_bottleneck_16_firstc_6_phi_beta_cosine', help='diffuser id')
    parser.add_argument('--beta_0', type=float, default=0.01, help='beta_0')
    parser.add_argument('--beta_T', type=float, default=0.5, help='beta_T')
    parser.add_argument('--beta_schedule', type=str, default="cosine", help='beta_schedule')
    parser.add_argument('--ckpt_path', type=str, default='/mnt/home/dheurtel/ceph/02_checkpoints', help='ckpt_path')
    parser.add_argument('--epochs', type=int, default=3000, help='epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='lr')
    parser.add_argument('--optimizer', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--scheduler', type=str, default=None, help='scheduler')
    parser.add_argument('--in_channels', type=int, default=1, help='in_channels')
    parser.add_argument('--dim_param', type=int, default=2, help='dim_param')
    parser.add_argument('--in_size', type=int, default=256, help='in_size')
    parser.add_argument('--order', type=int, default=1, help='order')
    parser.add_argument('--num_blocks', type=int, default=3, help='num_blocks')
    parser.add_argument('--first_channels', type=int, default=10, help='first_channels')
    parser.add_argument('--time_embed_dim', type=int, default=100, help='time_embed_dim')
    parser.add_argument('--padding_mode', type=str, default="circular", help='padding_mode')
    parser.add_argument('--normalize', type=str, default="GN", help='normalize')
    parser.add_argument('--group_c', type=int, default=1, help='group_c')
    parser.add_argument('--skiprescale', type=bool, default=True, help='skiprescale')
    parser.add_argument('--discretization', type=str, default="continuous", help='discretization')
    parser.add_argument('--embedding_mode', type=str, default=None, help='embedding_mode')
    parser.add_argument('--n_steps', type=int, default=1000, help='n_steps')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    args = parser.parse_args()

    main(args)