# ------------------------------------------------------------------------
# Fused deployment export for the ReCogDrive diffusion planner (small DiT).
#
# The bare DiT is NOT deployable alone: its inputs are produced by the
# planner's encoders (+ the VLM). We fuse them into two deployable graphs so
# we do NOT end up with a pile of tiny ONNXes:
#
#   planner_encode.onnx  (per FRAME, once)
#       in : vl_features (B, M, 1536), his_traj (B, 12), status_feature (B, 8)
#       out: vl_embeds (B, M, 384), history_embeds (B, 8, 384), ego_embeds (B, 384)
#       = feature_encoder + his_traj_encoder + ego_status_encoder
#
#   planner_step.onnx    (per STEP, N times = the hot loop)
#       in : current_actions (B, 8, 3), t (B,) int64,
#            vl_embeds (B, M, 384), history_embeds (B, 8, 384), ego_embeds (B, 384)
#       out: pred (B, 8, 3)   (= action_decoder(DiT(...)))
#       = action_encoder + position_embedding + fusion_projector + DiT + action_decoder
#
# The VLM (InternVL) -> vl_features is deployed separately (tensorrt-edgellm).
#
# Run with: ~/edgellm/bin/python scripts/export_planner_onnx.py
# ------------------------------------------------------------------------
import os
import sys

import numpy as np
import torch
import torch.nn as nn

torch.compile = lambda fn, *a, **k: fn  # de-torchdynamo the @torch.compile decorators

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from timm.layers import Mlp  # noqa: E402
from navsim.agents.recogdrive.blocks.encoder import ActionEncoder  # noqa: E402
from navsim.agents.recogdrive.recogdrive_dit import LightningDiT  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "planner_onnx")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- dims (small DiT: num_heads=8, head_dim=48 -> inner_dim=384) ----
ACTION_DIM = 3
ACTION_HORIZON = 8
INNER_DIM = 384          # = num_heads * head_dim
OUTPUT_DIM = 512
VLM_HIDDEN = 1536        # vlm_size="small"
HIDDEN_SIZE = 1024
HIS_TRAJ_DIM = 12
STATUS_DIM = 8


class PlannerEncode(nn.Module):
    """Per-frame context encoder (runs once per driving frame)."""

    def __init__(self):
        super().__init__()
        self.feature_encoder = nn.Linear(VLM_HIDDEN, INNER_DIM)
        self.his_traj_encoder = Mlp(
            in_features=HIS_TRAJ_DIM, hidden_features=HIDDEN_SIZE,
            out_features=INNER_DIM, norm_layer=nn.LayerNorm,
        )
        self.ego_status_encoder = Mlp(
            in_features=STATUS_DIM, hidden_features=HIDDEN_SIZE,
            out_features=INNER_DIM, norm_layer=nn.LayerNorm,
        )

    def forward(self, vl_features, his_traj, status_feature):
        vl_embeds = self.feature_encoder(vl_features)
        history_embeds = self.his_traj_encoder(
        history_embeds = self.his_traj_encoder(
            his_traj.unsqueeze(1)
        ).repeat(1, ACTION_HORIZON, 1)
        ego_embeds = self.ego_status_encoder(status_feature)
        return vl_embeds, history_embeds, ego_embeds


class PlannerStep(nn.Module):
    """Single denoising step (the hot loop). Contains the DiT."""

    def __init__(self):
        super().__init__()
        self.action_encoder = ActionEncoder(ACTION_DIM, INNER_DIM)
        self.position_embedding = nn.Embedding(ACTION_HORIZON, INNER_DIM)
        self.fusion_projector = nn.Linear(INNER_DIM * 3, INNER_DIM)
        self.model = LightningDiT(
            num_heads=8, head_dim=48, num_layers=16,
            output_dim=OUTPUT_DIM, dropout=0.0, attention_bias=True,
            norm_eps=1e-5, interleave_attention=True,
        )
        self.action_decoder = Mlp(
            in_features=OUTPUT_DIM, hidden_features=HIDDEN_SIZE,
            out_features=ACTION_DIM, norm_layer=nn.LayerNorm,
        )

    def forward(self, current_actions, t, vl_embeds, history_embeds, ego_embeds):
        action_features = self.action_encoder(current_actions, t)
        action_features = action_features + self.position_embedding(
            torch.arange(ACTION_HORIZON, device=current_actions.device)
        )
        vl_embeds_mean = vl_embeds.mean(1, keepdim=True).repeat(1, ACTION_HORIZON, 1)
        fused_input = self.fusion_projector(
            torch.cat((history_embeds, vl_embeds_mean, action_features), dim=2)
        )
        model_output = self.model(fused_input, vl_embeds, ego_embeds, t)
        pred = self.action_decoder(model_output)
        return pred


def _export_step():
    torch.manual_seed(0)
    net = PlannerStep().eval()
    net.save_state = lambda: torch.save(net.state_dict(), os.path.join(OUT_DIR, "planner_step_state.pth"))

    B, M = 2, 64
    current_actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    t = torch.tensor([10, 200], dtype=torch.long)
    vl_embeds = torch.randn(B, M, INNER_DIM)
    history_embeds = torch.randn(B, ACTION_HORIZON, INNER_DIM)
    ego_embeds = torch.randn(B, INNER_DIM)

    onnx_path = os.path.join(OUT_DIR, "planner_step.onnx")
    torch.onnx.export(
        net, (current_actions, t, vl_embeds, history_embeds, ego_embeds),
        onnx_path,
        input_names=["current_actions", "timesteps", "vl_embeds",
                     "history_embeds", "ego_embeds"],
        output_names=["pred"],
        dynamic_axes={
            "current_actions": {0: "B"}, "timesteps": {0: "B"},
            "vl_embeds": {0: "B", 1: "M"}, "history_embeds": {0: "B"},
            "ego_embeds": {0: "B"}, "pred": {0: "B"},
        },
        opset_version=17, dynamo=False,
    )
    torch.save(net.state_dict(), os.path.join(OUT_DIR, "planner_step_state.pth"))
    print(f"[export] planner_step.onnx  ({os.path.getsize(onnx_path)/1e6:.1f} MB)")


def _export_encode():
    torch.manual_seed(1)
    net = PlannerEncode().eval()
    B, M = 2, 64
    vl_features = torch.randn(B, M, VLM_HIDDEN)
    his_traj = torch.randn(B, HIS_TRAJ_DIM)
    status_feature = torch.randn(B, STATUS_DIM)

    onnx_path = os.path.join(OUT_DIR, "planner_encode.onnx")
    torch.onnx.export(
        net, (vl_features, his_traj, status_feature),
        onnx_path,
        input_names=["vl_features", "his_traj", "status_feature"],
        output_names=["vl_embeds", "history_embeds", "ego_embeds"],
        dynamic_axes={
            "vl_features": {0: "B", 1: "M"}, "his_traj": {0: "B"},
            "status_feature": {0: "B"},
            "vl_embeds": {0: "B", 1: "M"}, "history_embeds": {0: "B"},
            "ego_embeds": {0: "B"},
        },
        opset_version=17, dynamo=False,
    )
    torch.save(net.state_dict(), os.path.join(OUT_DIR, "planner_encode_state.pth"))
    print(f"[export] planner_encode.onnx  ({os.path.getsize(onnx_path)/1e6:.1f} MB)")


def _verify_step():
    import onnxruntime as ort
    net = PlannerStep().eval()
    net.load_state_dict(torch.load(os.path.join(OUT_DIR, "planner_step_state.pth")))

    B, M = 3, 100
    ca = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    t = torch.tensor([1, 120, 500], dtype=torch.long)
    ve = torch.randn(B, M, INNER_DIM)
    he = torch.randn(B, ACTION_HORIZON, INNER_DIM)
    ee = torch.randn(B, INNER_DIM)

    with torch.no_grad():
        ref = net(ca, t, ve, he, ee).float().numpy()

    sess = ort.InferenceSession(os.path.join(OUT_DIR, "planner_step.onnx"),
                                providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "current_actions": ca.numpy(), "timesteps": t.numpy(),
        "vl_embeds": ve.numpy(), "history_embeds": he.numpy(),
        "ego_embeds": ee.numpy(),
    })[0]
    err = float(np.max(np.abs(out - ref)))
    cos = float(np.sum(out * ref) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
    print(f"[verify] planner_step  max_abs_err={err:.2e}  cosine={cos:.6f}")
    assert err < 1e-4 and cos > 0.9999


def _verify_encode():
    import onnxruntime as ort
    net = PlannerEncode().eval()
    net.load_state_dict(torch.load(os.path.join(OUT_DIR, "planner_encode_state.pth")))

    B, M = 3, 100
    vf = torch.randn(B, M, VLM_HIDDEN)
    ht = torch.randn(B, HIS_TRAJ_DIM)
    sf = torch.randn(B, STATUS_DIM)

    with torch.no_grad():
        r_vl, r_his, r_ego = net(vf, ht, sf)
    r_vl, r_his, r_ego = r_vl.float().numpy(), r_his.float().numpy(), r_ego.float().numpy()

    sess = ort.InferenceSession(os.path.join(OUT_DIR, "planner_encode.onnx"),
                                providers=["CPUExecutionProvider"])
    o_vl, o_his, o_ego = sess.run(None, {
        "vl_features": vf.numpy(), "his_traj": ht.numpy(), "status_feature": sf.numpy(),
    })
    for n, a, b in [("vl_embeds", o_vl, r_vl), ("history_embeds", o_his, r_his), ("ego_embeds", o_ego, r_ego)]:
        err = float(np.max(np.abs(a - b)))
        cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"[verify] planner_encode.{n}  max_abs_err={err:.2e}  cosine={cos:.6f}")
        assert err < 1e-4 and cos > 0.9999


if __name__ == "__main__":
    _export_encode()
    _export_step()
    _verify_encode()
    _verify_step()
    print("\nALL FUSED EXPORTS OK")
