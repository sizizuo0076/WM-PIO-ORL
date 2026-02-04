import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.ddpm import DenoiseDiffusion


class WorldModel:
    def __init__(
        self,
        eps_model, pred_reward_model, agent,
        pred_batch_size, seq_len, state_dim, action_dim, n_steps,
        train_loader, val_loader, test_loader,

        k_unroll=1,
        enable_self_feeding=False,
        self_feed_p_start=1.0,
        self_feed_p_end=0.2,
        self_feed_decay_epochs=50,
        unroll_discount=1.0,
        unroll_sample_steps=None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eps_model = eps_model.to(self.device)
        self.pred_reward_model = pred_reward_model.to(self.device)
        self.agent = agent

        self.pred_batch_size = pred_batch_size
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_steps = n_steps

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.k_unroll = int(k_unroll)
        self.enable_self_feeding = bool(enable_self_feeding)
        self.self_feed_p_start = float(self_feed_p_start)
        self.self_feed_p_end = float(self_feed_p_end)
        self.self_feed_decay_epochs = int(self_feed_decay_epochs)
        self.unroll_discount = float(unroll_discount)
        self.unroll_sample_steps = unroll_sample_steps
        self._epoch_counter = 0

        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            pred_reward_model=self.pred_reward_model,
            n_steps=self.n_steps,
            device=self.device
        )

        self.optimizer = AdamW(
            list(self.eps_model.parameters()) + list(self.pred_reward_model.parameters()),
            lr=2e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=3000,
            eta_min=1e-6
        )

    def _scheduled_p(self):
        if self.self_feed_decay_epochs <= 0:
            return self.self_feed_p_end
        e = min(self._epoch_counter, self.self_feed_decay_epochs)
        frac = e / float(self.self_feed_decay_epochs)
        return self.self_feed_p_start + frac * (self.self_feed_p_end - self.self_feed_p_start)

    def _update_window(self, traj, next_state):
        return torch.cat([traj[:, 1:, :], next_state.unsqueeze(1)], dim=1)

    @torch.no_grad()
    def _sample_next_state(self, trajectorys, actions):
        """
        trajectorys: [B,L,S]
        actions:     [B,A]
        return:      [B,S]
        """
        if trajectorys.dim() == 2:
            trajectorys = trajectorys.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        B = trajectorys.shape[0]
        xt = torch.randn([B, self.state_dim], device=self.device)

        steps = self.n_steps if self.unroll_sample_steps is None else int(self.unroll_sample_steps)
        steps = max(1, min(steps, self.n_steps))

        if steps == self.n_steps:
            t_list = list(range(self.n_steps - 1, -1, -1))
        else:
            idx = torch.linspace(self.n_steps - 1, 0, steps, device=self.device)
            t_list = [int(x.item()) for x in idx]

        for t in t_list:
            xt = self.diffusion.p_sample(
                trajectorys,
                actions,
                xt,
                xt.new_full((B,), t, dtype=torch.long)
            )
        return xt

    def train_stage1(self):
        self.eps_model.train()
        self.pred_reward_model.train()

        total_loss = 0.0
        state_total_loss = 0.0
        reward_total_loss = 0.0

        p_gt = self._scheduled_p()

        for data in self.train_loader:
            trajectorys, actions, rewards, next_states = data
            self.optimizer.zero_grad()

            if actions.dim() == 2:
                loss, state_loss, reward_loss = self.diffusion.train_loss(
                    trajectorys, actions, next_states, rewards
                )

            else:
                B, K, _ = actions.shape
                K_use = min(K, self.k_unroll)

                traj_k = trajectorys  # [B,L,S]
                loss_sum = 0.0
                state_sum = 0.0
                reward_sum = 0.0
                weight_sum = 0.0

                for k in range(K_use):
                    act_k = actions[:, k, :]
                    ns_gt = next_states[:, k, :]
                    rew_gt = rewards[:, k, :]

                    loss_k, state_k, reward_k = self.diffusion.train_loss(
                        traj_k, act_k, ns_gt, rew_gt
                    )

                    wk = (self.unroll_discount ** k)
                    loss_sum += wk * loss_k
                    state_sum += wk * state_k
                    reward_sum += wk * reward_k
                    weight_sum += wk

                    if self.enable_self_feeding and p_gt < 1.0:
                        ns_pred = self._sample_next_state(traj_k, act_k)  # [B,S]
                        g = torch.bernoulli(torch.full((B, 1), p_gt, device=self.device))
                        ns_mix = g * ns_gt + (1.0 - g) * ns_pred.detach()
                        traj_k = self._update_window(traj_k, ns_mix)
                    else:
                        traj_k = self._update_window(traj_k, ns_gt)

                inv = 1.0 / max(weight_sum, 1e-8)
                loss = loss_sum * inv
                state_loss = state_sum * inv
                reward_loss = reward_sum * inv

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.eps_model.parameters()) + list(self.pred_reward_model.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

            total_loss += float(loss.item())
            state_total_loss += float(state_loss.item())
            reward_total_loss += float(reward_loss.item())

        self.scheduler.step()
        self._epoch_counter += 1

        n = len(self.train_loader)
        return total_loss / n, state_total_loss / n, reward_total_loss / n

    @torch.no_grad()
    def val_stage1(self):
        self.eps_model.eval()
        self.pred_reward_model.eval()

        total_loss = 0.0
        state_total_loss = 0.0
        reward_total_loss = 0.0

        for data in self.val_loader:
            trajectorys, actions, rewards, next_states = data

            if actions.dim() == 2:
                loss, state_loss, reward_loss = self.diffusion.val_loss(
                    trajectorys, actions, next_states, rewards
                )
            else:
                B, K, _ = actions.shape
                K_use = min(K, self.k_unroll)

                traj_k = trajectorys
                loss_sum = 0.0
                state_sum = 0.0
                reward_sum = 0.0
                weight_sum = 0.0

                for k in range(K_use):
                    act_k = actions[:, k, :]
                    ns_gt = next_states[:, k, :]
                    rew_gt = rewards[:, k, :]

                    loss_k, state_k, reward_k = self.diffusion.val_loss(
                        traj_k, act_k, ns_gt, rew_gt
                    )

                    wk = (self.unroll_discount ** k)
                    loss_sum += wk * loss_k
                    state_sum += wk * state_k
                    reward_sum += wk * reward_k
                    weight_sum += wk

                    traj_k = self._update_window(traj_k, ns_gt)

                inv = 1.0 / max(weight_sum, 1e-8)
                loss = loss_sum * inv
                state_loss = state_sum * inv
                reward_loss = reward_sum * inv

            total_loss += float(loss.item())
            state_total_loss += float(state_loss.item())
            reward_total_loss += float(reward_loss.item())

        n = len(self.val_loader)
        return total_loss / n, state_total_loss / n, reward_total_loss / n

    @torch.no_grad()
    def test_stage1(self):
        self.eps_model.eval()
        self.pred_reward_model.eval()

        total_loss = 0.0
        state_total_loss = 0.0
        reward_total_loss = 0.0

        for data in self.test_loader:
            trajectorys, actions, rewards, next_states = data

            if actions.dim() == 2:
                loss, state_loss, reward_loss = self.diffusion.val_loss(
                    trajectorys, actions, next_states, rewards
                )
            else:
                B, K, _ = actions.shape
                K_use = min(K, self.k_unroll)

                traj_k = trajectorys
                loss_sum = 0.0
                state_sum = 0.0
                reward_sum = 0.0
                weight_sum = 0.0

                for k in range(K_use):
                    act_k = actions[:, k, :]
                    ns_gt = next_states[:, k, :]
                    rew_gt = rewards[:, k, :]

                    loss_k, state_k, reward_k = self.diffusion.val_loss(
                        traj_k, act_k, ns_gt, rew_gt
                    )

                    wk = (self.unroll_discount ** k)
                    loss_sum += wk * loss_k
                    state_sum += wk * state_k
                    reward_sum += wk * reward_k
                    weight_sum += wk

                    traj_k = self._update_window(traj_k, ns_gt)

                inv = 1.0 / max(weight_sum, 1e-8)
                loss = loss_sum * inv
                state_loss = state_sum * inv
                reward_loss = reward_sum * inv

            total_loss += float(loss.item())
            state_total_loss += float(state_loss.item())
            reward_total_loss += float(reward_loss.item())

        n = len(self.test_loader)
        return total_loss / n, state_total_loss / n, reward_total_loss / n

    def gen_stage2(self, trajectorys,  agent):
        """
        ### Sample images with configurable loop count
        """
        trajectorys = trajectorys.to(self.device)


        with torch.no_grad():
            final_traj, final_actions, final_pred_reward, final_next_states = None, None, None, None

            for loop in range(self.seq_len):
                actions = agent.select_action(trajectorys)
                xt = torch.randn([self.pred_batch_size, self.state_dim], device=self.device)

                for t_ in range(self.n_steps):
                    t = self.n_steps - t_ - 1
                    next_states = self.diffusion.p_sample(
                        trajectorys, actions, xt,
                        xt.new_full((self.pred_batch_size,), t, dtype=torch.long)
                    )
                    xt = next_states

                if loop == self.seq_len - 1:
                    pred_reward = self.pred_reward_model(next_states, actions)
                    final_traj, final_actions, final_pred_reward, final_next_states = trajectorys, actions, pred_reward, next_states
                else:
                    trajectorys = torch.cat([trajectorys[:, 1:, :], next_states.unsqueeze(1)], dim=1)

            return final_traj, final_actions, final_pred_reward, final_next_states

    def save(self, model, model_save_path):
        self.save(model.state_dict(), model_save_path)