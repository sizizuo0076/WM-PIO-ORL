import torch
import torch.nn.functional as F
import sys
import math
from tqdm import *
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.pred_model import get_subsequent_mask_with_batch_length


def gather(consts, t):
    c = consts.gather(-1, t)
    return c.reshape(-1,  1)

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model, pred_reward_model, n_steps, device):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ models
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model
        self.pred_reward_model = pred_reward_model
        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        # self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        # $T$
        self.n_steps = n_steps
        self.beta = self.cosine_beta_schedule(T=self.n_steps,s=0.008).to(device)



        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # $\sigma^2 = \beta$
        self.sigma2 = self.beta

        weights_init(self.eps_model, init_type='normal', init_gain=0.02)
        weights_init(self.pred_reward_model, init_type='normal', init_gain=0.02)

    # 改进：余弦调度（网页3推荐）
    # def cosine_beta_schedule(self, n_steps):
    #     steps = (torch.arange(n_steps) / n_steps) + 0.008
    #     return torch.cos(steps * math.pi / 2).clamp(0, 0.999)

    def cosine_beta_schedule(self, T, s=0.008):

        steps = T + 1
        x = torch.linspace(0, T, steps)


        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]


        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.999)

        return betas


    def q_xt_x0(self, x0, t):
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    #添加噪声
    def q_sample(self, x0, t, eps=None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps

    # 在时间步 t-1 下的去噪后的图片或数据点
    def p_sample(self, states, actions, xt, t):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """
        mask = get_subsequent_mask_with_batch_length(states.shape[1], device=states.device)
        eps_theta = self.eps_model(xt, t, actions, states, mask)

        alpha_bar = gather(self.alpha_bar, t)

        alpha = gather(self.alpha, t)

        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5

        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)

        var = gather(self.sigma2, t)


        eps = torch.randn(xt.shape, device=xt.device)

        if (t == 0).all():
            return mean
        return mean + torch.sqrt(var) * eps



    def train_loss(self,  trajectorys,actions, x0, rewards, noise = None):
        """
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        self.eps_model.train()
        self.pred_reward_model.train()

        # Get batch size
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)

        mask = get_subsequent_mask_with_batch_length(trajectorys.shape[1],device=x0.device)

        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        # x, time, action, state_condition, mask
        eps_theta = self.eps_model(xt, t,  actions, trajectorys,  mask)

        state_eps_loss = F.mse_loss(noise, eps_theta)

        reward_pred = self.pred_reward_model(trajectorys,actions)

        reward_loss = F.mse_loss(rewards, reward_pred)

        total_loss = state_eps_loss + reward_loss

        return total_loss,state_eps_loss,reward_loss

    def val_loss(self,  trajectorys,actions, x0, rewards, noise = None):
        """
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """

        self.eps_model.eval()
        self.pred_reward_model.eval()
        with torch.no_grad():
            # Get batch size
            batch_size = x0.shape[0]
            t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

            # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
            if noise is None:
                noise = torch.randn_like(x0)

            # Sample $x_t$ for $q(x_t|x_0)$
            xt = self.q_sample(x0, t, eps=noise)

            mask = get_subsequent_mask_with_batch_length(trajectorys.shape[1],device=x0.device)

            # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
            # x, time, action, state_condition, mask
            eps_theta = self.eps_model(xt, t,  actions, trajectorys,  mask)

            state_eps_loss = F.mse_loss(noise, eps_theta)

            reward_pred = self.pred_reward_model(trajectorys,actions)

            reward_loss = F.mse_loss(rewards, reward_pred)

            total_loss = state_eps_loss + reward_loss

        return total_loss,state_eps_loss,reward_loss


    def ddpm_sample(self,trajectorys,actions):

        batch_size,seq_len,state_dim = trajectorys.shape

        with torch.no_grad():

            x = torch.randn([batch_size, seq_len, state_dim], device=trajectorys.device)

            for t_ in range(self.n_steps):
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.p_sample(trajectorys, actions, x, x.new_full((batch_size,), t, dtype=torch.long))

            return x



class Trainer:

    def __init__(self, eps_model,data_loader):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eps_model = eps_model.to(self.device)

        self.n_samples = 16
        self.image_size = 32
        self.n_steps = 1000


        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )


        self.data_loader = data_loader


        self.optimizer = AdamW(
            eps_model.parameters(),
            lr=3e-4,
            betas=(0.95, 0.999),
            weight_decay=0.05
        )


        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=2000,
            eta_min=1e-5
        )



    def sample(self):
        """
        ### Sample images
        """
        # 禁用梯度计算
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$


            x = torch.randn([self.n_samples, 1, self.image_size, self.image_size], device=self.device)


            for t_ in range(self.n_steps):
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            return x

    def run(self,num_epochs):
        """
        ### Training loop
        """
        for epoch in tqdm(range(num_epochs),file = sys.stdout):

            total_loss = 0
            for data in self.data_loader:
                data = data.reshape(-1, 1, self.image_size, self.image_size)
                data = data.to(self.device)
                self.optimizer.zero_grad()
                loss = self.diffusion.loss(data)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()


            if epoch % 1 == 0:
                tqdm.write(f"Epoch {epoch+1}, Loss: {total_loss:.6f}")
            if epoch % 50 == 0:
                self.save(self.eps_model,f'eps_model_{epoch}.pth')

    def save(self,model,model_save_path):
        self.save(model.state_dict(),model_save_path)

