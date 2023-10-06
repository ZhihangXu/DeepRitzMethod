import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from NNmodels import Poi_DeepRitzNet, Block
import numpy as np
import argparse
import os
# Try to solve the poisson equation:
'''  Solve the following PDE
-\Delta u(x) = 1, x\in \Omega,
u(x) = 0, x\in \partial \Omega  
\Omega = (-1,1) * (-1,1) \ [0,1) *{0}
'''

plt.switch_backend('agg')
print('ok')

# seeding
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# np.random.seed(0)

current_path = f'/project/mwang/zxu29/DeepRitzMethod/Results'
output_dir = current_path 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

block = Block(dim = 10)
print(block)


## problem settings
parser = argparse.ArgumentParser(description='Deep Ritz method to solve the Poisson equation')
parser.add_argument('--lr', type=float, default=1e-4,help ='learning rate')
# parser.add_argument('--weight', type=int, default=100, help ='weright of the loss_bd')
# parser.add_argument('--domain', type = list, default =[0,0.1,0,0.1], help='domain size')
# parser.add_argument('--epoches', type = int, default= 1000, help = '#epochs to train')
# parser.add_argument('--hidden_dim', type = int, default=20, help = 'hidden dimension of NN')
# parser.add_argument('--depth', type = int, default=5, help = 'depth of NN')
args = parser.parse_args()
# print("Epoches {} hidden_dim {} weight {} lr{:4f}".format(args.epoches, args.hidden_dim, args.weight, args.lr))

def get_interior_points(N=128,d=2):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    return torch.rand(N,d) * 2 - 1

def get_boundary_points(N=33):
    index = torch.rand(N, 1)
    index1 = torch.rand(N,1) * 2 - 1
    xb1 = torch.cat((index, torch.zeros_like(index)), dim=1)
    xb2 = torch.cat((index1, torch.ones_like(index1)), dim=1)
    xb3 = torch.cat((index1, torch.full_like(index1, -1)), dim=1)
    xb4 = torch.cat((torch.ones_like(index1), index1), dim=1)
    xb5 = torch.cat((torch.full_like(index1, -1), index1), dim=1)
    xb = torch.cat((xb1, xb2, xb3, xb4, xb5), dim=0)

    return xb

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def u_exact(x):
    r = np.sqrt(x[:,0]**2 + x[:,1]**2)
    theta = np.atan2(x[:,1], x[:,0])
    return np.sqrt(r) * np.sin(theta/2)

if __name__ == '__main__':
    epochs = 50000
    dim = 10
    depth = 4 

    print(torch.cuda.is_available())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = Poi_DeepRitzNet(dim,depth).to(device)
    # model.apply(weights_init)
    # criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    # x = torch.cat((xr, xb), dim=0)

    # if 2 < m:
    #     y = torch.zeros(x.shape[0], m - 2)
    #     x = torch.cat((x, y), dim=1)
    # # print(x.shape)

    
    # print(x.shape)
    for epoch in range(epochs+1):
        # print(xr.size())
        # print(xr)
        # generate the data set
        xr = get_interior_points()
        xb = get_boundary_points()
        xr.requires_grad_() 
        # xb.requires_grad()
        if 2 < dim:
            y = torch.zeros(xr.shape[0], dim - 2)
            y_b = torch.zeros(xb.shape[0], dim - 2)
            xr = torch.cat((xr, y), dim=1)
            xb = torch.cat((xb, y_b), dim=1)
        output_r = model(xr)
        output_b = model(xb)
        best_loss, best_epoch = 1000, 0
        beta = 500 
        grads = autograd.grad(outputs=output_r, inputs=xr,
                              grad_outputs=torch.ones_like(output_r),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        loss_r = 0.5 * torch.sum(torch.pow(grads, 2),dim=1) - output_r
        loss_r = torch.mean(loss_r)
        loss_b = torch.mean(torch.pow(output_b,2))
        loss = loss_r + beta * loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print('epoch:', epoch, 'loss:', loss.item(), 'loss_r:', (4 * loss_r).item(), 'loss_b:', (9 *500 * loss_b).item())
            if epoch > int(4 * epochs / 5):
                if torch.abs(loss) < best_loss:
                    best_loss = torch.abs(loss).item()
                    best_epoch = epoch
                    torch.save(model.state_dict(), 'new_best_deep_ritz1.mdl')
    print('best epoch:', best_epoch, 'best loss:', best_loss)

    # plot figure
    model.load_state_dict(torch.load('new_best_deep_ritz1.mdl'))
    print('load from ckpt!')
    with torch.no_grad():
        x1 = torch.linspace(-1, 1, 1001)
        x2 = torch.linspace(-1, 1, 1001)
        X, Y = torch.meshgrid(x1, x2)
        Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
        if 2 < dim:
            y = torch.zeros(Z.shape[0], dim - 2)
            Z = torch.cat((Z, y), dim=1)
        Z_torch = Z.to(device)
        pred = model(Z_torch)

    plt.figure()
    pred = pred.cpu().numpy()
    pred = pred.reshape(1001, 1001)
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',
                   extent=[-1, 1, -1, 1],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    fig = plt.gcf()
    fname = output_dir + f'/sol.png'
    fig.savefig(fname)

    plt.figure()
    u_ref = u_exact(Z)
    u_ref = u_ref.reshape(1001, 1001)
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',
                   extent=[-1, 1, -1, 1],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    fig = plt.gcf()
    fname = output_dir + f'/ref_sol.png'
    fig.savefig(fname)