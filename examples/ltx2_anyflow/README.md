# LTX2 AnyFlow MVP

This example adds an experimental first-stage AnyFlow-style flow-map training entrypoint for LTX2 without modifying the original DiffSynth files.

It keeps the DiffSynth dataset/accelerate training stack and swaps in:

- `FlowMapSFTAudioVideoLoss`: samples a pair of endpoints `t >= r` and trains velocity conditioned on both.
- `model_fn_ltx2_anyflow`: keeps the original LTX2 DiT call path but injects `r` through a zero-initialized token adapter.
- `AnyFlowTimeAdapter`: a small zero-init MLP attached to `pipe.dit` at runtime, so base LTX2 behavior is unchanged before training.

Example shape of a future run:

```bash
conda activate far
cd /root/job/DiffSynth-Studio
accelerate launch examples/ltx2_anyflow/model_training/train.py \
  --task flowmap_sft \
  --dataset_base_path /path/to/dataset \
  --dataset_metadata_path /path/to/metadata.csv \
  --data_file_keys video \
  --model_id_with_origin_paths 'Lightricks/LTX-2:*.safetensors' \
  --trainable_models dit \
  --output_path ./models/ltx2_anyflow
```

Use the same UnifiedDataset metadata style as the original `examples/ltx2/model_training/train.py`.
