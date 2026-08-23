# Load & run the fused planner_step FP16 engine with the `tensorrt` runtime
# inside the edgellm venv. Proves the planner graph is deployable here
# (tensorrt-edgellm LLM server is NOT needed for the planner; only the VLM uses it).
import os, sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import tensorrt as tr
import onnxruntime as ort

ENGINE = os.path.join(REPO_ROOT, "planner_onnx", "planner_step_fp16.engine")
ONNX = os.path.join(REPO_ROOT, "planner_onnx", "planner_step.onnx")

DT = {tr.DataType.FLOAT: torch.float32,
      tr.DataType.FLOAT16 if hasattr(tr.DataType, "FLOAT16") else tr.DataType.HALF: torch.float16,
      tr.DataType.INT64: torch.long}

def main():
    logger = tr.Logger(tr.Logger.WARNING)
    with open(ENGINE, "rb") as f:
        engine = tr.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    B, M = 1, 64
    sh = {"current_actions": (B, 8, 3), "timesteps": (B,),
          "vl_embeds": (B, M, 384), "history_embeds": (B, 8, 384), "ego_embeds": (B, 384)}
    for n, s in sh.items():
        ctx.set_input_shape(n, s)

    # fp16 random inputs
    inp = {
        "current_actions": np.random.randn(*sh["current_actions"]).astype(np.float16),
        "timesteps": np.array([150], np.int64),
        "vl_embeds": np.random.randn(*sh["vl_embeds"]).astype(np.float16),
        "history_embeds": np.random.randn(*sh["history_embeds"]).astype(np.float16),
        "ego_embeds": np.random.randn(*sh["ego_embeds"]).astype(np.float16),
    }
    dev = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        t = torch.empty(tuple(ctx.get_tensor_shape(name)), dtype=DT[engine.get_tensor_dtype(name)], device="cuda")
        dev[name] = t
        if engine.get_tensor_mode(name) == tr.TensorIOMode.INPUT:
            dev[name].copy_(torch.from_numpy(inp[name]))

    stream = torch.cuda.Stream()
    for n in inp:
        ctx.set_tensor_address(n, dev[n].data_ptr())
    ctx.set_tensor_address("pred", dev["pred"].data_ptr())

    for _ in range(10):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    trt_out = dev["pred"].cpu().float().numpy()

    # fp32 reference from onnxruntime
    ref = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"]).run(None, {
        "current_actions": inp["current_actions"].astype(np.float32),
        "timesteps": inp["timesteps"],
        "vl_embeds": inp["vl_embeds"].astype(np.float32),
        "history_embeds": inp["history_embeds"].astype(np.float32),
        "ego_embeds": inp["ego_embeds"].astype(np.float32),
    })[0]

    cos = float(np.sum(trt_out * ref) / (np.linalg.norm(trt_out) * np.linalg.norm(ref) + 1e-12))
    err = float(np.max(np.abs(trt_out - ref)))
    print(f"[edgellm+tensorrt] planner_step engine loaded & ran OK")
    print(f"[edgellm+tensorrt] output {trt_out.shape}, cosine vs onnxrt={cos:.6f}, max_abs_err={err:.2e}")
    assert cos > 0.99

if __name__ == "__main__":
    main()
