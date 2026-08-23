# Planner 模型结构

RecogDrive Planner 由帧级编码（`planner_encode.onnx`）与步级去噪（`planner_step.onnx`）两个 ONNX 组成，
VLM 特征（`vl_features`）由 TensorRT-Edge-LLM 单独部署产生。

## 整体结构（帧级 encode + 步级 DiT 循环）

```mermaid
flowchart TB
    subgraph SRC["宿主预处理 · 每帧"]
        CAM["前视相机图像"]
        HT["ego 历史位姿\n最近 4 时刻 × (x, y, heading) = 12"]
        ST["状态向量 (8)\n导航命令 one-hot (3)\n+ 车速 (2) + 加速度 (3)"]
    end

    subgraph VLM["VLM 帧级编码 · InternVL（TensorRT-Edge-LLM）"]
        CAM -->|pixel_values| V["InternVL Backbone\noutput_hidden_states"]
        V --> VF["vl_features (B, M, 1536)"]
    end

    subgraph ENC["planner_encode.onnx · 帧级 1 次"]
        VF --> FE["feature_encoder\nLinear 1536 → 384"]
        HT --> HTE["his_traj_encoder\nMLP 12 → 1024 → 384"]
        ST --> ESE["ego_status_encoder\nMLP 8 → 1024 → 384"]
        FE --> VE["vl_embeds (B, M, 384)"]
        HTE -->|"repeat ×8 (action_horizon)"| HE["history_embeds (B, 8, 384)"]
        ESE --> EE["ego_embeds (B, 384)"]
    end

    subgraph STEP["planner_step.onnx · 步级 ×N 次（热循环）"]
        XT["current_actions (B, 8, 3)\n噪声 / 上一步去噪结果"]
        TS["timesteps (B,) int64\nDDIM 步索引"]
        XT --> AE["action_encoder\nLinear(3→384) + 正弦时间编码"]
        TS --> AE
        AE -->|"+ 位置编码\nnn.Embedding(8, 384)"| AEP
        AEP["action_features (B, 8, 384)"]
        VE -->|"mean(1) → repeat(1,8,1)\n全局场景摘要"| VM["vl_mean (B, 8, 384)"]
        VM --> FP["fusion_projector\nLinear 1152 → 384\ncat(history, vl_mean, action)"]
        HE --> FP
        AEP --> FP
        FP --> DIT["LightningDiT\n16 层 · 8 头 × 48 dim\nRoPE · interleave cross-attn"]
        VE -->|"encoder_hidden_states\n(cross-attn KV)"| DIT
        EE -->|"conditioning_features"| DIT
        TS -->|"TimestepEncoder"| DIT
        DIT --> FL["FinalLayer\nRMSNorm+modulate → Linear 384→512"]
        FL --> AD["action_decoder\nMLP 512 → 1024 → 3"]
        AD --> PRED["pred ε (B, 8, 3)"]
    end

    subgraph DDIM["DDIM 采样循环 · 宿主运行时"]
        PRED --> UP["x_{t-1} = √α_{t-1} · clip((x_t − √(1−α_t)·ε)/√α_t)"]
        UP -->|"反馈为下步输入"| XT
    end

    subgraph POST["后处理"]
        UP --> CLIP["clip + denorm\n(×66.74 − 1.57 等)"]
        CLIP --> OUT["8 个 waypoint\n(x, y, heading) 轨迹"]
    end
```

## DiT 单层 block 内部（16 层重复，奇数层带 cross-attention）

```mermaid
flowchart LR
    X["hidden_states (B, 8, 384)"] --> N1["norm1 RMSNorm"]
    C["conditioning\n= TimestepEncoder(t)\n+ ego_embeds"] --> ADLN["adaLN_modulation\nLinear 384 → 6×384 (SiLU)"]
    ADLN -->|"shift_attn, scale_attn, gate_attn"| M1["modulate"]
    ADLN -->|"shift_ffn, scale_ffn, gate_ffn"| M2["modulate"]
    N1 --> M1
    M1 --> ATT["Attention 8 头 × 48\nself-attn + RoPE"]
    VE["vl_embeds (B, M, 384)\n仅奇数层作为 KV"] -->|"cross-attn"| ATT
    ATT -->|"× gate_attn"| R1["+ 残差"]
    R1 --> N2["norm2 RMSNorm"]
    N2 --> M2
    M2 --> FFN["SwiGLUFFN\nLinear → 2×hidden → SiLU gate"]
    FFN -->|"× gate_ffn"| R2["+ 残差"]
    R2 --> OUT["输出 (B, 8, 384)"]
```

## 关键点

- **时间条件注入两处**：`action_encoder` 的正弦时间编码（融合前），和 DiT 内 `TimestepEncoder`（384 维，与 `ego_embeds` 相加组成 conditioning，驱动 6 路 adaLN 调制）。
- **`vl_embeds` 三条路**：`mean` 摘要进 `fusion_projector`（query 侧全局条件）、完整序列做 cross-attention KV（每 2 层一次）、无直接调制。
- **位置编码**：waypoint 索引 `Embedding(8, 384)` 加在 action 特征上（在融合之前），因此 DiT 内注释掉的 `pos_embed` 不再使用。
- **循环边界清晰**：图内只有单步去噪网络，DDIM 更新、timesteps 递进、clip/denorm 全部在宿主运行时——即"网络进 engine、循环在框架层"的部署结构。

## 相关文件

- 导出脚本：`scripts/export_planner_onnx.py`
- DiT 定义：`navsim/agents/recogdrive/recogdrive_dit.py`
- C++ 推理：`TensorRT-Edge-LLM/cpp/action/recogdriveActionRunner.cpp`
