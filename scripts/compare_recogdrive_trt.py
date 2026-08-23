import sys
import numpy as np
import onnxruntime as ort

B, M, V, H, D = 1, 256, 1536, 8, 3

def load_inputs(path, M):
    a = np.fromfile(path, dtype=np.float32)
    off = 0
    vl = a[off:off+B*M*V].reshape(B, M, V); off += B*M*V
    his = a[off:off+B*12].reshape(B, 12); off += B*12
    status = a[off:off+B*8].reshape(B, 8); off += B*8
    return vl, his, status

def load_bin(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)

# ---- cosine DDIM schedule (matches recogdrive_diffusion_planner.py) ----
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps) / steps
    alphas_cumprod = np.cos((x + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

T, S = 100, 5
betas = cosine_beta_schedule(T)
alphas = 1 - betas
alphas_cumprod = np.cumprod(alphas)
alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])
step_ratio = T // S
ddim_t_schedule = np.arange(0, S) * step_ratio
ddim_alphas = alphas_cumprod[ddim_t_schedule]
ddim_alphas_prev = np.concatenate([[1.0], alphas_cumprod[ddim_t_schedule[:-1]]])
ddim_sqrt_one_minus_alphas = np.sqrt(1 - ddim_alphas)
ddim_t = ddim_t_schedule[::-1].copy()
alpha_t = ddim_alphas[::-1].copy()
sqrt_1m = ddim_sqrt_one_minus_alphas[::-1].copy()
alpha_prev = ddim_alphas_prev[::-1].copy()

# ---- engines (ONNX) ----
enc_sess = ort.InferenceSession("planner_onnx/planner_encode.onnx", providers=["CPUExecutionProvider"])
step_sess = ort.InferenceSession("planner_onnx/planner_step.onnx", providers=["CPUExecutionProvider"])

vl, his, status = load_inputs(sys.argv[1] if len(sys.argv) > 1 else "inputs.bin", M)
init_noise = load_bin(sys.argv[2] if len(sys.argv) > 2 else "init_noise.bin", (B, H, D))

enc_out = enc_sess.run(None, {"vl_features": vl.astype(np.float32),
                               "his_traj": his.astype(np.float32),
                               "status_feature": status.astype(np.float32)})
vl_embeds, history_embeds, ego_embeds = enc_out

cur = init_noise.astype(np.float64)
for i in range(S):
    t = np.full((B,), ddim_t[i], dtype=np.int64)
    pred = step_sess.run(None, {"current_actions": cur.astype(np.float32),
                                 "timesteps": t,
                                 "vl_embeds": vl_embeds.astype(np.float32),
                                 "history_embeds": history_embeds.astype(np.float32),
                                 "ego_embeds": ego_embeds.astype(np.float32)})[0]
    pred = pred.astype(np.float64)
    aT = float(alpha_t[i]); s1m = float(sqrt_1m[i]); aP = float(alpha_prev[i])
    x_recon = (cur - s1m * pred) / np.sqrt(aT)
    x_recon = np.clip(x_recon, -1, 1)
    cur = np.sqrt(aP) * x_recon
cur = np.clip(cur, -1, 1)
# denorm
p = cur.copy()
p[..., 0] = (p[..., 0] + 1) / 2 * 66.74 - 1.57
p[..., 1] = (p[..., 1] + 1) / 2 * 38.41 + 1.57
p[..., 2] = (p[..., 2] + 1) / 2 * 3.14
ref = p.reshape(-1).astype(np.float32)

cpp = load_bin("traj_cpp.bin", (B*H*D,))
cos = float(np.dot(ref, cpp) / (np.linalg.norm(ref) * np.linalg.norm(cpp) + 1e-12))
err = float(np.max(np.abs(ref - cpp)))
np.save("traj_ref.npy", ref)
print(f"steps={S}  cosine(ref,cpp)={cos:.8f}  max_err={err:.6e}")
