
```bash
python pretrain.py
```

```bash
python finetune.py --task craters \
  --checkpoint /path/to/pretrain/best.pt
```

Supported tasks:

| Task argument | Prediction target |
|---|---|
| `geology` | Geological class |
| `age` | Age class |
| `composition` | Patch-mean TiO2 and FeO |
| `crossmodal` | Patch means of withheld Thermal channels |
| `mare` | Mare segmentation |
| `craters` | Crater-region segmentation |

Evaluate a fine-tuned checkpoint on the selected split:

```bash
python evaluate.py --checkpoint /path/to/finetune/best.pt --split test
python evaluate.py --checkpoint /path/to/finetune/best.pt --split test --all-drops
```

## SomBench ICE

```bash
python ice.py finetune \
  --cache-dir /path/to/new_ice_cache \
  --checkpoint /path/to/pretrain/best.pt \
  --output-dir /path/to/new_ice_run

python ice.py evaluate \
  --checkpoint /path/to/new_ice_run/best.pt \
  --split test \
  --output-dir /path/to/new_ice_evaluation
```

```bash
python pretrain.py
python finetune.py
python evaluate.py
python ice.py

```
