# ------------------------------------------------------------------------
# Export the ReCogDrive DiT (LightningDiT) to ONNX.
# Run with: ~/edgellm/bin/python scripts/export_dit_onnx.py
# ------------------------------------------------------------------------
import os
import sys

import torch

# Neutralize @torch.compile decorators applied at import time so ONNX tracing
# (torch.jit.trace) does not try to wrap dynamo-optimized functions.
torch.compile = lambda fn, *a, **k: fn  # type: ignore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from navsim.agents.recogdrive.recogdrive_dit import LightningDiT  # noqa: E402

# "small" DiT config used by ReCogDrive (inner_dim = 8 * 48 = 384).
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

HIDDEN_SEQ_LEN = 8       # action_horizon
ENC_SEQ_LEN = 256        # VLM feature sequence length (dynamic)
INNER_DIM = DIT_CFG["num_heads"] * DIT_CFG["head_dim"]  # 1536


def main():
    out_dir = os.path.join(REPO_ROOT, "dit_onnx")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "dit.onnx")

    model = LightningDiT(**DIT_CFG)
    model.eval()

    dummy_hidden = torch.randn(1, HIDDEN_SEQ_LEN, INNER_DIM)
    dummy_enc = torch.randn(1, ENC_SEQ_LEN, INNER_DIM)
    dummy_cond = torch.randn(1, INNER_DIM)
    dummy_t = torch.tensor([500], dtype=torch.long)

    dynamic_axes = {
        "hidden_states": {0: "batch", 1: "seq_len"},
        "encoder_hidden_states": {0: "batch", 1: "enc_seq_len"},
        "conditioning_features": {0: "batch"},
        "timesteps": {0: "batch"},
        "output": {0: "batch", 1: "seq_len"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_hidden, dummy_enc, dummy_cond, dummy_t),
            onnx_path,
            input_names=[
                "hidden_states",
                "encoder_hidden_states",
                "conditioning_features",
                "timesteps",
            ],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
        )

    # Save the exact weights so the inference script can compare against the
    # same model that was baked into the ONNX graph.
    torch.save(model.state_dict(), os.path.join(out_dir, "dit_state.pth"))

    print(f"[export] ONNX written to {onnx_path}")


if __name__ == "__main__":
    main()
