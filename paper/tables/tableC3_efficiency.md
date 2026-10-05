**Table 3. Parameter and time efficiency (TITAN RTX 24 GB, batch 8, AMP, 384×384).** † Median wall-clock between 5-epoch checkpoints, validation included. ‡ Trained after a container/dataloader upgrade — not directly comparable with the two controlled-comparison models. Inference time and VRAM to be reported.

| Method | Params (M) | Train (h/epoch)† | Inference (ms/slice) | Peak VRAM (GB) |
|---|---:|---:|---:|---:|
| bi-GRU (original) | 668 | 2.41 | [TBD] | [TBD] |
| SS2D (controlled) | 31 | 3.07 | [TBD] | [TBD] |
| SS2D (enhanced) | 34 | 2.84‡ | [TBD] | [TBD] |
