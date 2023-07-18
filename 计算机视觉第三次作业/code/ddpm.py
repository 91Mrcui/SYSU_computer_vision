from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np





class residual_conv_block(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, res_flag: bool = False
    ) -> None:
        super().__init__()
        self.same_channels = input_channels == output_channels
        self.res_flag = res_flag
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, 1, 1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res_flag:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # 如果通道数相同，则将残差添加到输入上
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414  # 归一化
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Downsample, self).__init__()
        '''
        处理并下采样图像特征图
        '''
        layers = [residual_conv_block(input_channels, output_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Upsample, self).__init__()
        '''
        处理并上采样图像特征图
        '''
        layers = [
            nn.ConvTranspose2d(input_channels, output_channels, 2, 2),
            residual_conv_block(output_channels, output_channels),
            residual_conv_block(output_channels, output_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)  # 在通道维度上拼接张量
        x = self.model(x)
        return x


class Embeding_FC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embeding_FC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)



class U_Net(nn.Module):
    def __init__(self, input_channels, n_feat=256, n_classes=10):
        super(U_Net, self).__init__()

        self.input_channels = input_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = residual_conv_block(input_channels, n_feat, res_flag=True)  # 初始卷积层

        self.down1 = Downsample(n_feat, n_feat)  # 下采样模块1
        self.down2 = Downsample(n_feat, 2 * n_feat)  # 下采样模块2

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())  # 平均池化和GELU激活函数

        self.timeembed1 = Embeding_FC(1, 2 * n_feat)  # 时间嵌入层1
        self.timeembed2 = Embeding_FC(1, 1 * n_feat)  # 时间嵌入层2
        self.contextembed1 = Embeding_FC(n_classes, 2 * n_feat)  # 上下文嵌入层1
        self.contextembed2 = Embeding_FC(n_classes, 1 * n_feat)  # 上下文嵌入层2

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),  # 转置卷积进行上采样
            nn.GroupNorm(8, 2 * n_feat),  # 组归一化
            nn.ReLU(),
        )

        self.up1 = Upsample(4 * n_feat, n_feat)  # 上采样模块1
        self.up2 = Upsample(2 * n_feat, n_feat)  # 上采样模块2
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),  # 卷积层
            nn.GroupNorm(8, n_feat),  # 组归一化
            nn.ReLU(),
            nn.Conv2d(n_feat, self.input_channels, 3, 1, 1),  # 最终的卷积层
        )

    def forward(self, x, c, t, context_mask):
        # x是（带噪声的）图像，c是上下文标签，t是时间步骤，
        # context_mask表示哪些样本要屏蔽上下文

        x = self.init_conv(x)  # 初始卷积处理
        down1 = self.down1(x)  # 下采样1
        down2 = self.down2(down1)  # 下采样2
        hiddenvec = self.to_vec(down2)  # 将特征图转换为向量表示

        # 将上下文标签c转换为独热编码
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # 根据context_mask屏蔽上下文
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask))  # 需要翻转0和1
        c = c * context_mask

        # 嵌入上下文和时间步骤
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)  # 上下文嵌入层1
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)  # 时间嵌入层1
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)  # 上下文嵌入层2
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)  # 时间嵌入层2

        up1 = self.up0(hiddenvec)  # 上采样0
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # 上采样1，通过嵌入的乘法和加法融合上下文和时间信息
        up3 = self.up2(cemb2 * up2 + temb2, down1)  # 上采样2，通过嵌入的乘法和加法融合上下文和时间信息
        out = self.out(torch.cat((up3, x), 1))  # 将上采样的结果与初始输入拼接，并通过out生成最终的输出
        return out


#返回DDPM采样和训练过程的预先计算的调度表。
def get_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0 #beta1,beta2在(0, 1)
    # 每个时间步的 beta_t
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    # 每个时间步的 alpha_t
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    # 计算alphabar_t
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    # sqrtab
    sqrtab = torch.sqrt(alphabar_t)
    # oneover_sqrta
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    # sqrtmab
    sqrtmab = torch.sqrt(1 - alphabar_t)
    # mab_over_sqrtmab，即(1 - alpha_t) / sqrtmab
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_chance=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        # 使用get_schedules计算调度表
        for k, v in get_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.device = device
        self.drop_chance = drop_chance
        self.LOSS_MSE = nn.MSELoss()
    def forward(self, x, c):
        #训练，随机选择样本 t 和噪声
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        ) 
        # 以一定概率进行dropout
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_chance).to(self.device)
        # 返回加入噪声与预测的噪声之间的mse
        return self.LOSS_MSE(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0):
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)，采样初始噪声
        c_i = torch.arange(0, 10).to(device)  # 上下文标签仅循环遍历 mnist 标签
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))
        # 在测试时不进行上下文 dropout
        context_mask = torch.zeros_like(c_i).to(device)
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. 
        x_i_store = []  # 保留生成步骤以便绘图
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

# hardcoding these here
n_epoch = 20  # 训练轮数
batch_size = 256  # 批量大小
n_T = 400  # 500
device = "cuda:6"  # 设备（CUDA加速）
n_classes = 10  # 类别数
n_feat = 128  # 特征数，128个特征较好，256个特征更好但速度较慢
lrate = 1e-4  # 学习率
save_model = False  # 是否保存模型
save_dir = './data/output_fashion/'  # 保存路径
ws_test = [0.0, 0.5, 2.0]  # 生成指导的强度

def train(dataloader):

    ddpm = DDPM(nn_model=U_Net(input_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_chance=0.1)
    ddpm.to(device)  # 将模型移至指定设备上

    # 加载模型
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()])  # MNIST已经归一化到0到1之间

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)  # Adam优化器

    for ep in range(n_epoch):  # 按照指定轮数进行训练
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)  # 学习率线性衰减

        pbar = tqdm(dataloader)  # 进度条
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)  # 模型前向传播计算损失
            loss.backward()  # 反向传播
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # 对于eval，保存当前生成的样本的图像（最上面的行）
        # 然后是真实图像（最下面的行）
        
        ddpm.eval()  # 切换到评估模式
        with torch.no_grad():
            n_sample = 1*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)  # 生成样本

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try:
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]
                
                #print(x_gen.shape,x_real.shape)
                
                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                
                #print(x_all.shape)
                #print(grid.shape)
                
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")  # 保存生成图片
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep%5==0 or ep == int(n_epoch-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/n_classes)):
                            for col in range(n_classes):
                                axs[col].clear()
                                axs[col].set_xticks([])
                                axs[col].set_yticks([])
                                # plots.append(axs[col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))  # 保存生成的GIF动画
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")  # 保存模型
            print('saved model at ' + save_dir + f"model_{ep}.pth")




if __name__ == "__main__":
    
    # 加载MNIST数据集
    #tf = transforms.Compose([transforms.ToTensor()])  # MNIST已经归一化到0到1之间
    #dataset = MNIST("./data", train=True, download=True, transform=tf)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    
    # 加载Fashion MNIST数据集
    fashion_dataset = datasets.FashionMNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    dataloader = torch.utils.data.DataLoader(dataset=fashion_dataset, batch_size=64, shuffle=True)
    
    train(dataloader)
