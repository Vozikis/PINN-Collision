import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import pickle
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)
os.makedirs("collision_gifs2", exist_ok=True)


def finite_diff(y, t):
    # y: [N, 2], t: [N,1]
    dt = (t[1:] - t[:-1]).clamp_min(1e-8)  # [N-1,1]
    v = (y[1:] - y[:-1]) / dt              # [N-1,2]
    return v

def generate_observed_trajectory(m1_true, m2_true, v1_0=2.0, v2_0=0.0, T=2.0, N=400, R_dyn=0.2):
    base_radius = 0.08
    r1_true = base_radius * (m1_true ** (1/3))
    r2_true = base_radius * (m2_true ** (1/3))
    x1_0, x2_0 = -1.0, +1.0
    t = torch.linspace(0., T, N)
    rel_speed = (v1_0 - v2_0) + 1e-8
    tc_val = ((x2_0 - x1_0) - R_dyn) / rel_speed
    tc = t.new_tensor(tc_val)
    tc = torch.clamp(tc, min=t[1], max=t[-2])
    x1 = torch.empty_like(t); x2 = torch.empty_like(t)
    pre = t <= tc
    x1[pre] = x1_0 + v1_0 * t[pre]
    x2[pre] = x2_0 + v2_0 * t[pre]
    v1p = ((m1_true - m2_true)*v1_0 + 2*m2_true*v2_0) / (m1_true + m2_true)
    v2p = ((m2_true - m1_true)*v2_0 + 2*m1_true*v1_0) / (m1_true + m2_true)
    x1_tc = x1_0 + v1_0 * tc
    x2_tc = x2_0 + v2_0 * tc
    post = t > tc
    dt_post = t[post] - tc
    x1[post] = x1_tc + v1p * dt_post
    x2[post] = x2_tc + v2p * dt_post
    x_data = torch.stack([x1, x2], dim=1)  # [N,2]
    return t.unsqueeze(1), x_data, r1_true, r2_true, (x1_0, x2_0)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1, 64), nn.Tanh()]
        for _ in range(2):
            layers += [nn.Linear(64, 64), nn.Tanh()]
        layers += [nn.Linear(64, 2)]
        self.net = nn.Sequential(*layers)
    def forward(self, t): 
        return self.net(t)

def run_candidate(m1, m2, t, x_data, steps=5000, lr=1e-3, return_artifacts=False):
    t = t.to(device)
    x_data = x_data.to(device)

    net = MLP().to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    loss_total_list, loss_data_list, loss_phys_list = [], [], []

    for it in range(steps):
        opt.zero_grad()
        x_pred = net(t)  # [N,2]

        loss_data = ((x_pred - x_data) ** 2).mean()
        gap = x_pred[:, 1] - x_pred[:, 0]          # [N]
        tc_idx = int(torch.argmin(gap).item())    
        v_fd = finite_diff(x_pred, t)              # [N-1,2]
        pre_i  = max(0, min(v_fd.shape[0]-1, tc_idx-1))
        post_i = max(0, min(v_fd.shape[0]-1, tc_idx))
        v_minus = v_fd[pre_i]      # [2] velocities just before collision
        v_plus  = v_fd[post_i]     # [2] velocities just after collision
        mom_before = m1 * v_minus[0] + m2 * v_minus[1]
        mom_after  = m1 * v_plus[0]  + m2 * v_plus[1]
        mom_resid  = (mom_after - mom_before)
        en_before = 0.5 * m1 * (v_minus[0]**2) + 0.5 * m2 * (v_minus[1]**2)
        en_after  = 0.5 * m1 * (v_plus[0]**2)  + 0.5 * m2 * (v_plus[1]**2)
        en_resid  = (en_after - en_before)
        loss_gov = mom_resid.pow(2) + en_resid.pow(2)
        loss = loss_data + loss_gov
        loss.backward()
        opt.step()

        loss_total_list.append(loss.item())
        loss_data_list.append(loss_data.item())
        loss_phys_list.append(loss_gov.item())

    net.eval()
    with torch.no_grad():
        x_pred = net(t)
        mse = ((x_pred - x_data)**2).mean().item()
        gap = x_pred[:, 1] - x_pred[:, 0]
        tc_idx = int(torch.argmin(gap).item())
        v_fd = finite_diff(x_pred, t)
        pre_i  = max(0, min(v_fd.shape[0]-1, tc_idx-1))
        post_i = max(0, min(v_fd.shape[0]-1, tc_idx))
        v_minus = v_fd[pre_i]
        v_plus  = v_fd[post_i]
        mom_resid = (m1 * v_plus[0] + m2 * v_plus[1]) - (m1 * v_minus[0] + m2 * v_minus[1])
        en_resid  = (0.5*m1*v_plus[0]**2 + 0.5*m2*v_plus[1]**2) - (0.5*m1*v_minus[0]**2 + 0.5*m2*v_minus[1]**2)
        loss_gov = (mom_resid**2 + en_resid**2).item()
        total = mse + loss_gov

    artifacts = {
        "x_pred": x_pred.detach().cpu().numpy(),
        "x_true": x_data.detach().cpu().numpy(),
        "losses": {"total": loss_total_list, "data": loss_data_list, "gov": loss_phys_list},
        "t": t.detach().cpu().squeeze().numpy(),
        "rank_score": total,
        "mse": mse,
        "gov": loss_gov
    }
    return total, net, artifacts

def animate(x_vals, t_vals, r1, r2, filename, title):
    import numpy as np

    x_vis = np.array(x_vals, dtype=float).copy()
    R = float(r1 + r2)

    gap = x_vis[:, 1] - x_vis[:, 0]
    mask = gap < R
    if np.any(mask):
        mid  = 0.5 * (x_vis[mask, 0] + x_vis[mask, 1])
        need = 0.5 * (R - gap[mask])
        x_vis[mask, 0] = mid - need
        x_vis[mask, 1] = mid + need

    eps = 5e-4 * (x_vis.max() - x_vis.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(x_vis.min() - 0.2, x_vis.max() + 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel('x')
    time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
    ax.set_title(title)

    c1 = Circle((x_vis[0, 0], 0.0), r1, fc='C0', ec='k', zorder=2)
    c2 = Circle((x_vis[0, 1], 0.0), r2, fc='C1', ec='k', zorder=2)
    ax.add_patch(c1); ax.add_patch(c2)

    def update(i):
        x1i, x2i = x_vis[i, 0], x_vis[i, 1]
        gi = x2i - x1i
        if gi < R + eps:
            mid = 0.5 * (x1i + x2i)
            need = 0.5 * ((R + eps) - gi)
            x1i = mid - need
            x2i = mid + need
        c1.center = (x1i, 0.0)
        c2.center = (x2i, 0.0)
        time_text.set_text(f't = {t_vals[i]:.3f}s')
        return c1, c2, time_text

    ani = FuncAnimation(fig, update, frames=range(0, len(t_vals), 4), interval=10, blit=True)
    ani.save(filename, writer=PillowWriter(fps=60))
    plt.close()


@torch.no_grad()
def extract_vel_features(t, x, k=10):
    v_fd = finite_diff(x, t)           # [N-1,2]
    gap = (x[:,1] - x[:,0])            # [N]
    tc_idx = int(torch.argmin(gap).item())
    tc_idx = max(k, min(tc_idx, v_fd.shape[0]-k))
    v_minus = v_fd[tc_idx-k:tc_idx].mean(dim=0)   # [2]
    v_plus  = v_fd[tc_idx:tc_idx+k].mean(dim=0)   # [2]
    feat = torch.stack([v_minus[0], v_minus[1], v_plus[0], v_plus[1]], dim=0)
    return feat

class MassMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
        self.softplus = nn.Softplus()
    def forward(self, feat_n):
        logm = self.net(feat_n)           
        m = self.softplus(logm) + 1e-4    
        return m

def train_mass_mlp(num_sims=4000, epochs=200, lr=1e-3, 
                   T=2.0, N=400, v1_range=(0.5,5.0), v2_range=(-0.5,1.0),
                   m_range=(0.5,20.0), device=device, verbose=True):
    model = MassMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    feats, targets = [], []
    for _ in range(num_sims):
        m1 = float(np.exp(np.random.uniform(np.log(m_range[0]), np.log(m_range[1]))))
        m2 = float(np.exp(np.random.uniform(np.log(m_range[0]), np.log(m_range[1]))))
        v1_0 = float(np.random.uniform(*v1_range))
        v2_0 = float(np.random.uniform(*v2_range))

        t, x_data, _, _, _ = generate_observed_trajectory(m1, m2, v1_0=v1_0, v2_0=v2_0, T=T, N=N)
        t = t.to(device); x_data = x_data.to(device)
        feat = extract_vel_features(t, x_data)     # [4]
        feats.append(feat)
        targets.append(torch.tensor([m1, m2], device=device))

    X = torch.stack(feats, dim=0).to(device)       # [M,4]
    Y = torch.stack(targets, dim=0).to(device)     # [M,2]

    mu = X.mean(dim=0, keepdim=True)
    sigma = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sigma
    model.feat_mu = mu.detach()
    model.feat_sigma = sigma.detach()

    model.train()
    bs = 256
    for ep in range(epochs):
        idx = torch.randperm(Xn.shape[0], device=device)
        tot = 0.0
        for i in range(0, Xn.shape[0], bs):
            xb = Xn[idx[i:i+bs]]
            yb = Y[idx[i:i+bs]]
            pred = model(xb)
            loss = ((pred - yb)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        if verbose and (ep % 20 == 0 or ep == epochs-1):
            print(f"[MassMLP] epoch {ep+1:03d}/{epochs} | MSE={tot/Xn.shape[0]:.4e}")
    model.eval()
    return model

@torch.no_grad()
def mlp_predict_masses(model, t, x):
    """Predict (m1, m2) from positions x(t) using MassMLP."""
    dev = next(model.parameters()).device
    t = t.to(dev); x = x.to(dev)
    feat = extract_vel_features(t, x)                                 
    feat_n = (feat - model.feat_mu.squeeze(0)) / model.feat_sigma.squeeze(0)
    m = model(feat_n.unsqueeze(0)).squeeze(0)                          
    return float(m[0].item()), float(m[1].item())

def load_trajectories(base_directory, file_name, kernel_size=25):
    all_trajectories = []
    for trajectory_number in range(9):
        video_dir = os.path.join(base_directory, f"video_{trajectory_number}")
        file_path = os.path.join(video_dir, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            data_len = len(data)
            t = np.arange(data_len)
            x = np.array([item[0] if item is not None else np.nan for item in data])
            y = np.array([item[1] if item is not None else np.nan for item in data])
            
            valid_start_idx = np.where(~np.isnan(x))[0][0]
            x = x[valid_start_idx:]
            y = y[valid_start_idx:]
            t = t[valid_start_idx:]
            
            df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
            traj_x = df['x'].to_numpy() / 100.0
            traj_y = df['y'].to_numpy() / 100.0
            traj_t = df['t'].to_numpy()
            
            traj_t_shifted = np.linspace(0, 6, len(traj_x)) 
            t_data = torch.tensor(traj_t_shifted, dtype=torch.float32).reshape(-1, 1)
            x_data = torch.tensor(traj_x, dtype=torch.float32).reshape(-1, 1)
            y_data = torch.tensor(traj_y, dtype=torch.float32).reshape(-1, 1)

            all_trajectories.append((t_data, x_data, y_data))
    return all_trajectories


def make_common_time_grid(T=2.0, N=400):
    return np.linspace(0.0, T, N, dtype=np.float32)

def resample_series_to_grid(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t_src = t_src.flatten().astype(np.float32)
    y_src = y_src.flatten().astype(np.float32)
    return np.interp(t_grid, t_src, y_src)

def build_real_observation_from_pkls(traj_A, traj_B, T_match=2.0, N_match=400):
    tA, _, yA = traj_A
    tB, _, yB = traj_B

    tA_np = tA.squeeze(1).cpu().numpy().astype(np.float32)
    tB_np = tB.squeeze(1).cpu().numpy().astype(np.float32)
    yA_np = yA.squeeze(1).cpu().numpy().astype(np.float32)
    yB_np = yB.squeeze(1).cpu().numpy().astype(np.float32)

    t_min = max(tA_np.min(), tB_np.min())
    t_max = min(tA_np.max(), tB_np.max())
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        t_min = min(tA_np.min(), tB_np.min())
        t_max = max(tA_np.max(), tB_np.max())

    t_obs = np.linspace(t_min, t_max, N_match, dtype=np.float32)

    y1_obs = np.interp(t_obs, tA_np, yA_np)
    y2_obs = np.interp(t_obs, tB_np, yB_np)

    T_obs = max(t_max - t_min, 1e-6)
    t_match = (t_obs - t_min) * (T_match / T_obs)

    t_torch = torch.tensor(t_match, dtype=torch.float32).reshape(-1, 1)
    x_torch = torch.tensor(np.stack([y1_obs, y2_obs], axis=1), dtype=torch.float32) 
    return t_torch, x_torch

@torch.no_grad()
def mlp_predict_masses_on_real(mass_mlp: nn.Module, t_real: torch.Tensor, x_real: torch.Tensor):
    return mlp_predict_masses(mass_mlp, t_real, x_real)

def run_pinn_on_real_with_mlp(mass_mlp: nn.Module,
                              traj_A, traj_B,
                              steps=10000, T_match=2.0, N_match=400,
                              tag="real"):
    t_real, x_real = build_real_observation_from_pkls(traj_A, traj_B,
                                                      T_match=T_match, N_match=N_match)
    t_real = t_real.to(device)
    x_real = x_real.to(device)

    m1_hat, m2_hat = mlp_predict_masses_on_real(mass_mlp, t_real, x_real)
    print(f"[{tag}] MassMLP on REAL y(t): (m1≈{m1_hat:.3f}, m2≈{m2_hat:.3f})")

    score, net, art = run_candidate(m1_hat, m2_hat, t_real, x_real, steps=steps, return_artifacts=True)
    print(f"[{tag}] PINN total={score:.6e} | data={art['mse']:.6e} | gov={art['gov']:.6e}")

    base_radius = 0.08
    r1_cand = base_radius * (m1_hat ** (1/3))
    r2_cand = base_radius * (m2_hat ** (1/3))

    x_pred = art["x_pred"]   
    t_vals = art["t"]        
    tag_best = f"REAL_predMLP_{m1_hat:.3f}_{m2_hat:.3f}"

    os.makedirs("collision_gifs2", exist_ok=True)
    animate(x_pred, t_vals, r1_cand, r2_cand,
            f"collision_gifs2/real_pred_{tag_best}.gif",
            f"PINN on REAL (pred masses: {m1_hat:.2f}, {m2_hat:.2f})")
    animate(x_real.detach().cpu().numpy(), t_vals, r1_cand, r2_cand,
            f"collision_gifs2/real_trace_{tag_best}.gif",
            f"Observed REAL trace (y1,y2)")

    L = art["losses"]
    plt.figure()
    plt.plot(L["total"], label='Total (data + governing)')
    plt.plot(L["data"], label='Data MSE')
    plt.plot(L["gov"],  label='Governing residual')
    plt.yscale('log'); plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.legend(); plt.title(f"REAL Loss (predMLP {m1_hat:.2f},{m2_hat:.2f})"); plt.tight_layout()
    plt.savefig(f"collision_gifs2/real_loss_{tag_best}.png"); plt.close()

    print(f"\n=== REAL MASS RESULT for {tag} ===")
    print(f"Pred masses: (m1≈{m1_hat:.3f}, m2≈{m2_hat:.3f}) | TOTAL={score:.6e}")
    return {"m1_hat": m1_hat, "m2_hat": m2_hat, "score": score, "art": art}


if __name__ == "__main__":
    mass_mlp = train_mass_mlp(
        num_sims=4000, epochs=1000, lr=1e-3, 
        T=2.0, N=400,              
        v1_range=(0.5,5.0), v2_range=(-0.5,1.0),
        m_range=(0.5,20.0), device=device, verbose=True
    )

    base_dir = "real-world-cropped/collision"
    file_A = "centres3d_obj_1.pkl"  
    file_B = "centres3d_obj_2.pkl"   
    traj_A = load_trajectories(base_dir, file_A)  
    traj_B = load_trajectories(base_dir, file_B)

    n_traj = min(len(traj_A), len(traj_B))
    for idx in range(n_traj):
        print(f"\n=== REAL RUN: video_{idx} ===")
        _ = run_pinn_on_real_with_mlp(
            mass_mlp,
            traj_A[idx], traj_B[idx],
            steps=10000,      
            T_match=2.0,     
            N_match=400,
            tag=f"real_video_{idx}"
        )
