"""User settings. Edit paths here once, then run the three entry scripts.

JSON files passed with --config contain the internal sections: data, model,
optim, downstream, runtime, name and seed. Command-line values take priority.
"""

DATA_ROOT = "/root/autodl-tmp/Moonstone"
OUTPUT_ROOT = "/root/autodl-tmp/try_runs/try_refactored"
PRETRAIN_CHECKPOINT = None   
EVALUATION_CHECKPOINT = None 

COMMON = {
    "seed": 42,
    "data": {
        "mmap_dir": DATA_ROOT + "/mmap",
        "h5_path": DATA_ROOT + "/benchmark/lunar_patches_v4.h5",
        "label_dir": DATA_ROOT + "/benchmark",
        "patch_size": 256,
        "ppd": 128,
        "split_protocol": "stored",
        "benchmark_backend": "mmap",
        "num_workers": 4,
    },
    "model": {
        "variant": "pico",
        "fusion": "gated",
        "decoder_dim": 64,
        "drop_path": 0.0,
        "gradient_checkpointing": False,
        "branch_mora": {"enabled": True, "rank": 4, "alpha": 0.01, "shared": True, "stages": 2,
                        "gram_context": "exchange"},
    },
    "runtime": {
        "device": "auto",
        "output_root": OUTPUT_ROOT,
        "cpu_threads": 4,
        "log_every": 10,
        "save_every": 10,
        "deterministic": False,
        "strict_validation": True,
        "mmap_read_hint": "default", 
        "loader_prefetch_factor": 1,
    },
}

PRETRAIN = {
    "name": "pretrain",
    "data": {"pretrain_pool": "train", "train_samples": 200_000, "val_samples": 1024},
    "masking": {"ratio": 0.75, "block_size": 16, "modality_drop_prob": 0.2, "consistency_weight": 0.5},
    "optim": {
        "epochs": 100, "batch_size": 32, "grad_accum": 4,
        "lr": 1.5e-4, "min_lr": 1e-6, "warmup_epochs": 2,
        "weight_decay": 0.05, "clip_grad": 1.0, "precision": "bf16",
    },
}

FINETUNE = {
    "name": "finetune",
    "downstream": {
        "task": "mare", "mode": "finetune", "checkpoint": PRETRAIN_CHECKPOINT,
        "encoder_lr_scale": 1.0, "label_smoothing": 0.1,
        "feature_dropout": 0.1, "patience": 10, "few_shot": 0,
        "drop_groups": [], "train_group_drop_prob": 0.0, "consistency_weight": 1.0,
    },
    "optim": {
        "epochs": 30, "lr": 1e-4, "min_lr": 1e-6, "warmup_epochs": 2,
        "weight_decay": 0.05, "clip_grad": 1.0, "precision": "bf16",
    },
}

TASKS = {
    "geology": {"optim": {"batch_size": 16, "grad_accum": 1}},
    "age": {"optim": {"batch_size": 16, "grad_accum": 1}},
    "composition": {"optim": {"batch_size": 16, "grad_accum": 1}},
    "crossmodal": {"optim": {"batch_size": 16, "grad_accum": 1}},
    "mare": {"optim": {"batch_size": 8, "grad_accum": 2}},
    "craters": {"optim": {"batch_size": 8, "grad_accum": 2}},
}

EVALUATE = {
    "checkpoint": EVALUATION_CHECKPOINT,
    "split": "val",             
    "all_drops": False,         
    "batch_size": 8,
    "num_workers": 4,
}
