cd /home/weiyu/recogdrive && export LD_LIBRARY_PATH=/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH && /usr/local/tensorrt/bin/trtexec \
  --onnx=dit_onnx/dit.onnx \
  --saveEngine=dit_onnx/dit_fp16.engine \
  --fp16 \
  --shapes=hidden_states:1x8x384,encoder_hidden_states:1x256x384,conditioning_features:1x384,timesteps:1 \
  --warmUp=200 --iterations=300 2>&1 | tail -40

# mean 2.72 ms、median 2.32 ms
