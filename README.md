# DAgger Robometer

This repository combines:

- `better_openpi/`: OpenPI policy inference, including the standalone live Robometer monitoring server.
- `robometer/`: Robometer progress and success estimation, including the customized visualization scripts.

## Model weights

Large `*.safetensors` files are intentionally excluded because they exceed GitHub's normal file-size limit.
Download or copy the required OpenPI and Robometer checkpoints separately. The Robometer model metadata and
configuration are retained under `robometer/Robometer-4B/`; place its downloaded safetensors shards back in
that directory when running locally.

For the combined inference workflow, see
[`better_openpi/docs/robometer_live_inference.md`](better_openpi/docs/robometer_live_inference.md).
