# ------------------------------------------------------------------------
# Run ONNX inference for the ReCogDrive DiT with onnxruntime (edgellm env)
# and verify it matches the PyTorch model.
# Run with: ~/edgellm/bin/python scripts/infer_dit_onnx.py
# ------------------------------------------------------------------------
import os
import sys

import numpy as np
import torch

# Neutralize @torch.compile so the PyTorch reference runs without dynamo.
torch.compile = lambda fn, *a, **k: fn  # type: ignore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from navsim.agents.recogdrive.recogdrive_dit import LightningDiT  # noqa: E402
import onnxruntime as ort

DIT_CFG = dict(
    num_heads=8,
    head_dim=48,
    num_layers=16,
    output_dim=512,
    dropout=0.0,
    attention_bias=True,
    norm_eps=1e-5,
    interleave_attention=True,
)

HIDDEN_SEQ_LEN = 8
ENC_SEQ_LEN = 256
INNER_DIM = DIT_CFG["num_heads"] * DIT_CFG["head_dim"]


def main():
    onnx_path = os.path.join(REPO_ROOT, "dit_onnx", "dit.onnx")
    assert os.path.exists(onnx_path), f"ONNX not found: {onnx_path}"

    # --- PyTorch reference (same weights as captured in the ONNX) ---
    model = LightningDiT(**DIT_CFG).eval()
    state_path = os.path.join(REPO_ROOT, "dit_onnx", "dit_state.pth")
    assert os.path.exists(state_path), f"state dict not found: {state_path}"
    model.load_state_dict(torch.load(state_path, map_location="cpu"))

    torch.manual_seed(0)
    hidden = torch.randn(2, HIDDEN_SEQ_LEN, INNER_DIM)
    enc = torch.randn(2, ENC_SEQ_LEN, INNER_DIM)
    cond = torch.randn(2, INNER_DIM)
    t = torch.tensor([123, 777], dtype=torch.long)

    with torch.no_grad():
        pt_out = model(hidden, enc, cond, t).float().numpy()

    # --- onnxruntime inference ---
    so = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "hidden_states": hidden.numpy(),
        "encoder_hidden_states": enc.numpy(),
        "conditioning_features": cond.numpy(),
        "timesteps": t.numpy(),
    }
    ort_out = sess.run(None, ort_inputs)[0]

    # --- compare ---
    max_err = float(np.max(np.abs(pt_out - ort_out)))
    cos = float(
        np.sum(pt_out * ort_out)
        / (np.linalg.norm(pt_out) * np.linalg.norm(ort_out) + 1e-12)
    )
    print(f"[infer] output shape (pytorch): {pt_out.shape}")
    print(f"[infer] output shape (onnxrt):  {ort_out.shape}")
    print(f"[infer] max abs error: {max_err:.6e}")
    print(f"[infer] cosine sim:    {cos:.8f}")

    assert pt_out.shape == ort_out.shape, "shape mismatch"
    assert max_err < 1e-3, f"outputs differ too much: max_err={max_err}"
    print("[infer] SUCCESS: ONNX inference matches PyTorch within tolerance.")


if __name__ == "__main__":
    main()
