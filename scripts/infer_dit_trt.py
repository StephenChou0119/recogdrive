# ------------------------------------------------------------------------
# Run TensorRT (FP16) inference for the ReCogDrive small DiT and verify it
# matches the ONNX (onnxruntime) output. Uses the tensorrt python API.
# Run with: ~/edgellm/bin/python scripts/infer_dit_trt.py
# ------------------------------------------------------------------------
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import tensorrt as tr  # noqa: E402

TRT_ENGINE = os.path.join(REPO_ROOT, "dit_onnx", "dit_fp16.engine")
ONNX_PATH = os.path.join(REPO_ROOT, "dit_onnx", "dit.onnx")

def _tr_dtype(name):
    return getattr(tr.DataType, name)

DTYPE_MAP = {
    _tr_dtype("FLOAT"): torch.float32,
    _tr_dtype("FLOAT16" if hasattr(tr.DataType, "FLOAT16") else "HALF"): torch.float16,
    _tr_dtype("INT64"): torch.long,
    _tr_dtype("INT32"): torch.int32,
}


def main():
    assert os.path.exists(TRT_ENGINE), f"TRT engine not found: {TRT_ENGINE}"

    logger = tr.Logger(tr.Logger.WARNING)
    with open(TRT_ENGINE, "rb") as f:
        engine = tr.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Concrete shapes (must satisfy the engine profile used at build time).
    shapes = {
        "hidden_states": (1, 8, 384),
        "encoder_hidden_states": (1, 256, 384),
        "conditioning_features": (1, 384),
        "timesteps": (1,),
    }
    for name, sh in shapes.items():
        context.set_input_shape(name, sh)

    torch.manual_seed(0)
    np_dtype = {
        "hidden_states": np.float16,
        "encoder_hidden_states": np.float16,
        "conditioning_features": np.float16,
        "timesteps": np.int64,
    }
    host = {
        "hidden_states": np.random.randn(*shapes["hidden_states"]).astype(np.float16),
        "encoder_hidden_states": np.random.randn(*shapes["encoder_hidden_states"]).astype(np.float16),
        "conditioning_features": np.random.randn(*shapes["conditioning_features"]).astype(np.float16),
        "timesteps": np.array([123], dtype=np.int64),
    }

    # Allocate device buffers.
    dev = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        shape = tuple(context.get_tensor_shape(name))
        t = torch.empty(shape, dtype=DTYPE_MAP[dtype], device="cuda")
        dev[name] = t
        if engine.get_tensor_mode(name) == tr.TensorIOMode.INPUT:
            dev[name].copy_(torch.from_numpy(host[name]))

    stream = torch.cuda.Stream()
    context.set_tensor_address("hidden_states", dev["hidden_states"].data_ptr())
    context.set_tensor_address("encoder_hidden_states", dev["encoder_hidden_states"].data_ptr())
    context.set_tensor_address("conditioning_features", dev["conditioning_features"].data_ptr())
    context.set_tensor_address("timesteps", dev["timesteps"].data_ptr())
    out_name = "output"
    context.set_tensor_address(out_name, dev[out_name].data_ptr())

    # Warm-up.
    for _ in range(20):
        context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    # Correctness: compare against onnxruntime (fp32 reference).
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    host_fp32 = {
        "hidden_states": host["hidden_states"].astype(np.float32),
        "encoder_hidden_states": host["encoder_hidden_states"].astype(np.float32),
        "conditioning_features": host["conditioning_features"].astype(np.float32),
        "timesteps": host["timesteps"].astype(np.int64),
    }
    ort_out = sess.run(None, host_fp32)[0]  # fp32 reference
    context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    trt_out = dev[out_name].cpu().float().numpy()

    cos = float(
        np.sum(trt_out * ort_out)
        / (np.linalg.norm(trt_out) * np.linalg.norm(ort_out) + 1e-12)
    )
    max_err = float(np.max(np.abs(trt_out - ort_out)))
    print(f"[trt] output shape: {trt_out.shape}")
    print(f"[trt] cosine sim vs onnxrt: {cos:.6f}")
    print(f"[trt] max abs error (vs fp32 onnxrt): {max_err:.6e}")
    assert cos > 0.99, f"TRT output diverges from ONNX: cos={cos}"
    print("[trt] SUCCESS: TRT FP16 output matches ONNX within FP16 tolerance.")

    # Latency benchmark (CUDA events).
    n = 300
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(n):
        context.execute_async_v3(stream.cuda_stream)
    end.record(stream)
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / n
    print(f"[trt] latency (cuda events, {n} iters): mean={ms:.3f} ms "
          f"-> {1000 / ms:.1f} qps (batch={shapes['hidden_states'][0]})")


if __name__ == "__main__":
    main()
