DATA_DIR = ""
CACHE_DIR = ""
OUTPUT_ROOT = ""
PRETRAIN_CHECKPOINT = None  
EVALUATION_CHECKPOINT = None  

SETTINGS = {
    "seed": 42,
    "patch_size": 256,
    "model": {
        "variant": "nano", 
        "fusion": "gated", "decoder_dim": 64, "drop_path": 0.0,
        "gradient_checkpointing": False,
        "branch_mora": {"enabled": True, "rank": 4, "alpha": 0.01, "shared": True, "stages": 2},
    },
    "training": {
        "epochs": 100, "batch_size": 8, "grad_accum": 1,
        "lr": 1e-4, "min_lr": 1e-6, "warmup_epochs": 2,
        "weight_decay": 0.05, "clip_grad": 1.0,
        "encoder_lr_scale": 0.1,  
        "group_drop_prob": 0.5,  
        "patience": 20,
    },
    "runtime": {
        "device": "auto", "precision": "bf16", "num_workers": 4,
        "cpu_threads": 4, "log_every": 10, "deterministic": False,
    },
}
