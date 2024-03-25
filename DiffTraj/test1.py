''' get the loss and save the fig '''
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from types import SimpleNamespace

from utils.Traj_UNet import *
from utils.config import args
from utils.utils import * 
from torch.utils.data import DataLoader


def get_lossfig(path):
    loss_list = []
    # iterate the files in the path
    for i, file in enumerate(os.listdir(path)):
        if file.endswith('.npy'):
            loss_list.append(np.load(os.path.join(path, file)))
    print(len(loss_list))
    loss_list = np.array(loss_list)
    loss_list = np.concatenate(loss_list, axis=0)
    print(loss_list.shape)
    # plot the loss
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.show()

def get_distribution():
    # the mean and std of head information, using for rescaling
    hmean = np.array([227.42564669834402, 3.2168432031181324, 2380.0845723287666, 19.118453271201098, 106.05070685966737])
    hstd = np.array([227.42564669834402, 3.2168432031181324, 2380.0845723287666, 19.118453271201098, 106.05070685966737])
    mean = np.array([-8.61882354, 41.1565547])         # longitude, latitude
    std = np.array([0.02313287, 0.0094905])        # longitude, latitude
    # the original mean and std of trajectory length, using for rescaling the trajectory length
    len_mean = 209.33181  # Porto
    len_std = 107.47809894170952 # Porto
    return hmean, hstd, mean, std, len_mean, len_std

def generate_attr(batchsize=100):
    # duration, speed, dist, hop, start_timestamp, usr_id.
    hmean, hstd, _, _, _, _ = get_distribution()
    duration = np.random.normal(hmean[0], hstd[0], batchsize).clip(0, 5000).astype(int)
    speed = np.random.uniform(-1, 1, batchsize).astype(float)
    dist = np.random.uniform(-1, 1, batchsize).astype(float)
    hop = np.random.normal(hmean[3], hstd[3], batchsize).clip(0, 210).astype(int)
    stime = np.random.normal(hmean[4], hstd[4], batchsize).clip(0, 540).astype(int)
    id = np.random.randint(0, 900, batchsize)
    head = np.stack([duration, speed, dist, hop, stime, id], axis=1)
    return head

def traj_plot(trajs, path):
    # plot the generated trajectories
    plt.figure()
    for traj in trajs:
        plt.plot(traj[:, 0], traj[:, 1], 'b')
    # plt.xlim(-8.690, -8.555)
    # plt.ylim(41.139, 41.186)
    plt.savefig(os.path.join(path, 'test_trajs.png'))
    plt.show()

    # save the generated trajectories
    trajs = np.array(trajs)
    np.save(os.path.join(path, 'test_trajs.npy'), trajs)

def generate_traj(path):
    traj_path = os.path.join(path, 'trajs/')
    if not os.path.exists(traj_path):
        os.makedirs(path, exist_ok=True)

    print("load configs and models...")
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)   # SimpleNamespace类似于字典，但是可以通过属性访问
    config = SimpleNamespace(**temp)
    unet = Guide_UNet(config).cuda()
    unet.load_state_dict(torch.load(os.path.join(path, 'models/01-20-21-01-44/unet_200.pt')))
    print('load model successfully!')

    # generate the trajectories
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                           config.diffusion.beta_end, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 2e-4  # Explore this - might want it lower when training on the full dataset
    eta=0.0
    timesteps=100
    skip = n_steps // timesteps
    seq = range(0, n_steps, skip)

    # load head information for guide trajectory generation
    # random sample the head for trajs, num is the number of trajs to generate
    batchsize = 200
    head = generate_attr(batchsize)[:,:-1]
    head = torch.from_numpy(head).float().cuda()
    dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=4)

    x = torch.randn(batchsize, 2, config.data.traj_length).cuda()    # noise
    ims = [];    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        with torch.no_grad():
            pred_noise = unet(x, t, head)
            # print(pred_noise.shape)
            x = p_xt(x, pred_noise, t, next_t, beta, eta)
            if i % 10 == 0:
                ims.append(x.cpu().squeeze(0))
    trajs = ims[-1].cpu().numpy()
    trajs = trajs[:,:2,:]

    _, _, mean, std, len_mean, len_std = get_distribution()
    lengths = np.random.normal(len_mean, len_std, batchsize).clip(24, 1000).astype(int)
    Gen_traj = []

    # resample the trajectory length
    for j in range(batchsize):
        new_traj = resample_trajectory(trajs[j].T, lengths[j])    # trajs[j].T: (120, 2)
        new_traj = new_traj * std + mean
        # clip the trajectory on latitude and longitude
        # new_traj[:, 0] = np.clip(new_traj[:, 0], -8.689284, -8.5559414)
        # new_traj[:, 1] = np.clip(new_traj[:, 1], 41.1399093, 41.1858236)
        Gen_traj.append(new_traj)

    # save the generated trajectories and its figure
    traj_plot(Gen_traj, path)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    base_path = os.path.dirname(os.path.abspath(__file__))    # base path, /home/.../DiffTraj

    ''' if loss fig is needed, uncomment the following lines'''
    # loss_path = 'DiffTraj/Porto_steps=500_len=400_0.05_bs=32/results/'
    # print(base_path)
    # print(os.path.join(base_path, loss_path))
    # get_lossfig(os.path.join(base_path, loss_path))

    ''' get the generated trajectories and its figure '''
    task_path = 'DiffTraj/Porto_steps=500_len=400_0.05_bs=32/'
    generate_traj(os.path.join(base_path, task_path))
