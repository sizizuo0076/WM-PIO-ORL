import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tcn import TemporalConvNet


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        num_channels = [256, 256, 256, 256]
        self.tcn = TemporalConvNet(state_dim, num_channels)
        self.fc1 = nn.Linear(self.seq_len, 1)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, trajectorys):  ##[B,L,D]
        trajectorys = trajectorys.permute(0, 2, 1)
        trajectorys = self.tcn(trajectorys)  # [B,256,L]
        actions = F.relu(self.fc1(trajectorys)).squeeze(2)
        actions = F.relu(self.fc2(actions))
        return torch.tanh(actions)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class T3D:
    def __init__(
            self,
            state_dim,
            action_dim,
            seq_len,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            device="cuda"
    ):
        self.actor = Actor(state_dim, action_dim, seq_len).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, states):
        with torch.no_grad():
            return self.actor(states)

    def train(self, trajectorys, actions, rewards, next_states):
        self.total_it += 1
        states = trajectorys[:, -1, :]
        next_trajectorys = torch.cat([trajectorys[:, :-1, :], next_states.unsqueeze(1)], dim=1)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_trajectorys) + noise
            ).clamp(-1.0, 1.0)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # # Delayed policy updates
        # # if self.total_it % self.policy_freq == 0:
        #
        # # Compute actor losse
        # actor_loss = -self.critic.Q1(states, self.actor(trajectorys)).mean()
        #
        # # Optimize the actor
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()
        #
        # # Update the frozen target models
        # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        #
        # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        #
        # return critic_loss.item(), actor_loss.item()
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(trajectorys)
            Q = self.critic.Q1(states, pi)
            lmbda = self.alpha / Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, actions)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)



