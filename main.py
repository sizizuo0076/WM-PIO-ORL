import os
from argparse import ArgumentParser

import torch
import pandas as pd
import tqdm

from data.data_loader import OfflineReplayBuffer, HybridReplayBuffer
from models.utils import set_seed
from models.worldmodel import WorldModel
from models.rl import T3D
from models.pred_model import TemporalTransformer, RewardPred


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_folder_name(args):
    return (
        f"seq_len_{args.cond_dim}_trans_feat_{args.trans_embed_dim}"
        f"_reward_hidden_{args.reward_hidden}_trans_num_layers_{args.trans_num_layers}"
        f"_k_{args.k_unroll}"
    )


def build_eps_model(args, device):
    return TemporalTransformer(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        device=device,
        cond_dim=args.cond_dim,
        embed_dim=args.trans_embed_dim,
        max_length=512,
        num_heads=args.trans_num_head,
        num_layers=args.trans_num_layers,
        dropout=0.2
    ).to(device)


def build_reward_model(args, device):
    return RewardPred(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.reward_hidden,
        out_dim=1,
        seq_len=args.cond_dim
    ).to(device)


def build_agent_stage1_or_stage2(args, device):
    return T3D(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        seq_len=args.cond_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.3,
        policy_freq=1,
        device=device
    )


def build_world_model(args, device, train_loader, val_loader, test_loader, eps_model, pred_reward_model, agent):
    return WorldModel(
        eps_model=eps_model,
        pred_reward_model=pred_reward_model,
        agent=agent,
        pred_batch_size=args.pred_batch_size,
        seq_len=args.cond_dim,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        n_steps=args.n_steps,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        k_unroll=args.k_unroll,
        enable_self_feeding=args.enable_self_feeding,
        self_feed_p_start=args.self_feed_p_start,
        self_feed_p_end=args.self_feed_p_end,
        self_feed_decay_epochs=args.self_feed_decay_epochs,
        unroll_discount=args.unroll_discount,
        unroll_sample_steps=args.unroll_sample_steps
    )



def build_hybrid_buffer_stage2(args):
    device = get_device()

    data = pd.read_excel(args.data_path)

    offline = OfflineReplayBuffer(
        data=data,
        time_steps=args.cond_dim,
        split_ratios=(1.0, 0.0, 0.0),
        k_unroll=1
    )
    train_loader, val_loader, test_loader = offline.get_dataloaders(
        batch_size=args.batch_size,
        shuffle=True,
        device=device
    )

    eps_model = build_eps_model(args, device)
    pred_reward_model = build_reward_model(args, device)

    agent = build_agent_stage1_or_stage2(args, device)

    world_model = build_world_model(
        args=args,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        eps_model=eps_model,
        pred_reward_model=pred_reward_model,
        agent=agent
    )

    folder_name = make_folder_name(args)

    ckpt_epoch = args.stage1_ckpt_epoch
    eps_ckpt = f'./ckpt/{folder_name}/eps_model_{ckpt_epoch}.pth'
    rew_ckpt = f'./ckpt/{folder_name}/pred_reward_model_{ckpt_epoch}.pth'

    eps_model.load_state_dict(torch.load(eps_ckpt, map_location=device))
    pred_reward_model.load_state_dict(torch.load(rew_ckpt, map_location=device))
    eps_model.eval()
    pred_reward_model.eval()

    agent.actor.eval()

    total_capacity = args.real_capacity + args.model_capacity
    hybrid_buffer = HybridReplayBuffer(
        capacity=total_capacity,
        state_shape=args.state_dim,
        action_shape=args.action_dim,
        seq_length=args.cond_dim
    )

    real_count = 0
    model_count = 0

    os.makedirs(f'./results/h5/{folder_name}', exist_ok=True)
    out_path = f'./results/h5/{folder_name}/hybrid_buffer.h5'

    print("#############################   Stage 2: Build Hybrid Buffer   #############################")
    print(f"real_capacity={args.real_capacity}, model_capacity={args.model_capacity}")

    with torch.no_grad():
        while real_count < args.real_capacity or model_count < args.model_capacity:
            for traj_b, act_b, rew_b, nxt_b in tqdm.tqdm(train_loader, leave=False):
                B = traj_b.shape[0]

                if real_count < args.real_capacity:
                    need = args.real_capacity - real_count
                    take = min(B, need)

                    hybrid_buffer.add_batch(
                        traj_b[:take].detach().cpu().numpy(),
                        act_b[:take].detach().cpu().numpy(),
                        rew_b[:take].detach().cpu().numpy(),
                        nxt_b[:take].detach().cpu().numpy()
                    )
                    real_count += take

                if model_count < args.model_capacity:
                    gen_block = min(128, B)

                    remain = args.model_capacity - model_count
                    gen_take = min(gen_block, remain)

                    traj_gen = traj_b[:gen_take]

                    a_gen = agent.select_action(traj_gen)
                    if not torch.is_tensor(a_gen):
                        a_gen = torch.tensor(a_gen, device=device, dtype=torch.float32)
                    a_gen = a_gen.clamp(-1.0, 1.0)

                    ns_gen = world_model._sample_next_state(traj_gen, a_gen)
                    r_gen = pred_reward_model(traj_gen, a_gen)

                    hybrid_buffer.add_batch(
                        traj_gen.detach().cpu().numpy(),
                        a_gen.detach().cpu().numpy(),
                        r_gen.detach().cpu().numpy(),
                        ns_gen.detach().cpu().numpy()
                    )
                    model_count += gen_take

                if real_count >= args.real_capacity and model_count >= args.model_capacity:
                    break

            if real_count >= args.real_capacity and model_count >= args.model_capacity:
                break

    hybrid_buffer.save_to_hdf5(out_path)

    print("#############################   Stage 2: Done   #############################")
    print(f"real={real_count}, model={model_count}")
    print(f"saved to: {out_path}")

    return out_path


def run_stage1(args):
    set_seed(args.seed)
    data = pd.read_excel(args.data_path)
    device = get_device()

    replay_buffer = OfflineReplayBuffer(
        data=data,
        time_steps=args.cond_dim,
        split_ratios=(0.8, 0.1, 0.1),
        k_unroll=args.k_unroll
    )
    train_loader, val_loader, test_loader = replay_buffer.get_dataloaders(
        batch_size=args.batch_size,
        shuffle=True,
        device=device
    )

    eps_model = build_eps_model(args, device)
    pred_reward_model = build_reward_model(args, device)
    agent = build_agent_stage1_or_stage2(args, device)

    world_model = build_world_model(
        args=args,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        eps_model=eps_model,
        pred_reward_model=pred_reward_model,
        agent=agent
    )

    train_loss_list = []
    val_loss_list = []

    folder_name = make_folder_name(args)
    os.makedirs("./ckpt/" + folder_name, exist_ok=True)
    os.makedirs("./results/excel/" + folder_name, exist_ok=True)

    print("#############################   Stage 1: Training begins   #############################")
    for epoch in range(args.epochs):
        train_total_loss, train_state_loss, train_reward_loss = world_model.train_stage1()
        train_loss_list.append([train_total_loss, train_state_loss, train_reward_loss])
        print(
            f"Epoch {epoch + 1} \n\t Train: Total Loss: {train_total_loss:.6f}| state_loss: "
            f"{train_state_loss:.6f}| reward_total_loss: {train_reward_loss:.6f}"
        )

        val_total_loss, val_state_loss, val_reward_loss = world_model.val_stage1()
        val_loss_list.append([val_total_loss, val_state_loss, val_reward_loss])
        print(
            f"\t Val: Total Loss: {val_total_loss:.6f}| state_loss: {val_state_loss:.6f}| "
            f"reward_loss: {val_reward_loss:.6f}"
        )

        if (epoch + 1) % 5 == 0:
            torch.save(eps_model.state_dict(), f'./ckpt/{folder_name}/eps_model_{epoch + 1}.pth')
            torch.save(pred_reward_model.state_dict(), f'./ckpt/{folder_name}/pred_reward_model_{epoch + 1}.pth')

            train_loss = pd.DataFrame(train_loss_list, columns=['total_loss', 'state_loss', 'reward_loss'])
            train_loss.to_excel(f'./results/excel/{folder_name}/train_loss_{folder_name}.xlsx')

            val_loss = pd.DataFrame(val_loss_list, columns=['total_loss', 'state_loss', 'reward_loss'])
            val_loss.to_excel(f'./results/excel/{folder_name}/val_loss_{folder_name}.xlsx')

    test_total_loss, test_state_loss, test_reward_loss = world_model.test_stage1()
    print(
        f"Test: World Loss: {test_total_loss:.6f}| state_total_loss: {test_state_loss:.6f}| "
        f"reward_total_loss: {test_reward_loss:.6f}"
    )

    print("##############################   Stage 1: Training ends   ##############################")


def run_stage2(args):
    print("\n" + "#" * 70)
    print("#####################   Stage 2: Build Hybrid Buffer   #####################")
    print("#" * 70 + "\n")

    build_hybrid_buffer_stage2(args)

    print("\n" + "#" * 70)
    print("#####################   Stage 2 Finished Successfully   #####################")
    print("#" * 70 + "\n")


def run_stage3(args):
    set_seed(args.seed)
    device = get_device()

    folder_name = make_folder_name(args)
    if args.h5_path is None or len(args.h5_path) == 0:
        h5_path = f'./results/h5/{folder_name}/hybrid_buffer.h5'
    else:
        h5_path = args.h5_path

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"[Stage3] Hybrid buffer not found: {h5_path}")

    print("\n" + "#" * 70)
    print("#####################   Stage 3: Offline RL Training (T3D)   #####################")
    print(f"[Stage3] Loading buffer from: {h5_path}")
    print("#" * 70 + "\n")

    hybrid_buffer = HybridReplayBuffer.load_from_hdf5(h5_path)
    N = int(hybrid_buffer.size)
    print(f"[Stage3] Buffer loaded: size={N}, capacity={hybrid_buffer.capacity}, seq_len={hybrid_buffer.seq_length}")

    agent = T3D(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        seq_len=args.cond_dim,
        discount=args.rl_discount,
        tau=args.rl_tau,
        policy_noise=args.rl_policy_noise,
        noise_clip=args.rl_noise_clip,
        policy_freq=args.rl_policy_freq,
        device=device
    )
    if not hasattr(agent, "alpha"):
        agent.alpha = float(args.rl_alpha)

    os.makedirs(f'./ckpt/rl/{folder_name}', exist_ok=True)

    if device.type != "cuda":
        print("[Stage3][Warn] device is CPU; scheme E provides no benefit. Consider scheme D instead.")

    traj_gpu = torch.from_numpy(hybrid_buffer.trajectorys[:N]).to(device=device, dtype=torch.float32)
    act_gpu  = torch.from_numpy(hybrid_buffer.actions[:N]).to(device=device, dtype=torch.float32)
    rew_gpu  = torch.from_numpy(hybrid_buffer.rewards[:N]).to(device=device, dtype=torch.float32)
    ns_gpu   = torch.from_numpy(hybrid_buffer.next_states[:N]).to(device=device, dtype=torch.float32)

    batch_size = int(args.rl_batch_size)

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    perm = torch.randperm(N, generator=g, device=device)
    pos = 0

    for it in range(1, args.rl_steps + 1):
        if pos + batch_size > N:
            perm = torch.randperm(N, generator=g, device=device)
            pos = 0

        idx = perm[pos:pos + batch_size]
        pos += batch_size

        trajectorys = traj_gpu[idx]
        actions     = act_gpu[idx]
        rewards     = rew_gpu[idx]
        next_states = ns_gpu[idx]

        agent.train(trajectorys, actions, rewards, next_states)

        if it % args.rl_log_interval == 0:
            print(f"[Stage3] step={it}/{args.rl_steps}")

        if it % args.rl_save_interval == 0:
            save_prefix = f'./ckpt/rl/{folder_name}/t3d_step_{it}'
            agent.save(save_prefix)
            print(f"[Stage3] saved: {save_prefix}_actor/_critic ...")

    final_prefix = f'./ckpt/rl/{folder_name}/t3d_final'
    agent.save(final_prefix)
    print(f"\n[Stage3] Finished. Final saved: {final_prefix}_actor/_critic ...\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a models with the specified configuration.")

    # -------- Basic / data --------
    parser.add_argument("--data_path", type=str, default="./data/processed_reward.xlsx", help="Path to offline dataset (Excel).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # -------- Stage1 training --------
    parser.add_argument("--epochs", type=int, default=200, help="Stage1 epochs (keep small for quick test).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for dataloaders in Stage1/Stage2.")
    parser.add_argument("--pred_batch_size", type=int, default=1024, help="Internal prediction batch size used in WorldModel.")

    # -------- Trajectory / dimensions --------
    parser.add_argument("--cond_dim", type=int, default=20, help="Trajectory length L (window size).")
    parser.add_argument("--state_dim", type=int, default=38, help="State dimension S.")
    parser.add_argument("--action_dim", type=int, default=8, help="Action dimension A.")

    # -------- Diffusion --------
    parser.add_argument("--n_steps", type=int, default=100, help="Diffusion steps (denoise steps).")

    # -------- TemporalTransformer (eps_model) --------
    parser.add_argument("--trans_embed_dim", type=int, default=256, help="Transformer embedding dim.")
    parser.add_argument("--trans_num_layers", type=int, default=4, help="Transformer number of layers.")
    parser.add_argument("--trans_num_head", type=int, default=8, help="Transformer attention heads.")

    # -------- Reward predictor --------
    parser.add_argument("--reward_hidden", type=int, default=256, help="Hidden dim for reward predictor.")

    # -------- Legacy / misc --------
    parser.add_argument("--capacity", type=int, default=100, help="Legacy placeholder (kept for compatibility).")

    # -------- Unroll / self-feeding (Stage1) --------
    parser.add_argument("--k_unroll", type=int, default=5, help="K-step unroll length for Stage1.")
    parser.add_argument("--enable_self_feeding", action="store_true", help="Enable scheduled self-feeding (optional).")
    parser.add_argument("--self_feed_p_start", type=float, default=1.0, help="Teacher forcing prob at start.")
    parser.add_argument("--self_feed_p_end", type=float, default=0.2, help="Teacher forcing prob at end.")
    parser.add_argument("--self_feed_decay_epochs", type=int, default=100, help="Decay epochs for self-feeding schedule.")
    parser.add_argument("--unroll_discount", type=float, default=1.0, help="Loss discount across unroll steps.")
    parser.add_argument("--unroll_sample_steps", type=int, default=20, help="Diffusion steps used during unroll sampling.")

    # -------- Stage selection --------
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="1=Stage1, 2=Stage2, 3=Stage3.")

    # -------- Stage2 buffer sizes --------
    parser.add_argument("--real_capacity", type=int, default=100000, help="Stage2: number of real transitions stored.")
    parser.add_argument("--model_capacity", type=int, default=200000, help="Stage2: number of generated transitions stored.")
    parser.add_argument("--stage1_ckpt_epoch", type=int, default=200, help="Stage2: which Stage1 checkpoint epoch to load.")

    # -------- Stage3 offline RL params --------
    parser.add_argument("--h5_path", type=str, default=None,
                        help="Stage3: path to hybrid_buffer.h5. If None, use default ./results/h5/{folder_name}/hybrid_buffer.h5.")
    parser.add_argument("--rl_steps", type=int, default=20000, help="Stage3: offline RL update steps (small for testing).")
    parser.add_argument("--rl_batch_size", type=int, default=512, help="Stage3: batch size sampled from hybrid buffer.")
    parser.add_argument("--rl_discount", type=float, default=0.99, help="Stage3: RL discount gamma.")
    parser.add_argument("--rl_tau", type=float, default=0.005, help="Stage3: target network soft update tau.")
    parser.add_argument("--rl_policy_noise", type=float, default=0.2, help="Stage3: TD3 target policy smoothing noise.")
    parser.add_argument("--rl_noise_clip", type=float, default=0.3, help="Stage3: clip range for target noise.")
    parser.add_argument("--rl_policy_freq", type=int, default=2, help="Stage3: actor update frequency.")
    parser.add_argument("--rl_alpha", type=float, default=2.5, help="Stage3: TD3+BC coefficient alpha (fallback).")
    parser.add_argument("--rl_log_interval", type=int, default=200, help="Stage3: print log every N steps.")
    parser.add_argument("--rl_save_interval", type=int, default=1000, help="Stage3: save checkpoint every N steps.")

    args = parser.parse_args()

    if args.stage == 1:
        run_stage1(args)
    elif args.stage == 2:
        run_stage2(args)
    else:
        run_stage3(args)
