CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python robometer/evals/eval_server.py \
  model_path=Robometer-4B \
  server_url=127.0.0.1 \
  server_port=8001 \
  num_gpus=1 \
  max_workers=1