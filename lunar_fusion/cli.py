import argparse
import copy
import json
import os
from pathlib import Path

import default as settings
from .schema import GROUPS, TASK_OUTPUTS


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="Optional JSON overrides; relative paths use project root")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--workers", type=int, help="DataLoader workers")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration without training")
    return parser


def training_parser(kind):
    parser = _parser("Lunar pretraining" if kind == "pretrain" else "Lunar six-task fine-tuning")
    parser.add_argument("--resume", type=Path, help="Resume the same run from its last.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--check", action="store_true", help="One forward/loss check on train and val; no updates")
    parser.add_argument("--consistency-weight", type=float,
                        dest="pretrain_consistency" if kind == "pretrain" else "downstream_consistency",
                        help="Paired full / one-group-removed views; 0 disables (2x forward batch when on)")
    if kind == "pretrain":
        parser.add_argument("--samples-per-epoch", type=int)
        parser.add_argument("--fusion", choices=("gated", "coverage", "mean"))
        parser.add_argument("--backbone", choices=("gram", "private", "original"),
                            help="gram: shared Gram branches; private: branches without Gram; original: early fusion")
        parser.add_argument("--branch-stages", type=int, choices=(1, 2),
                            help="Number of separate modality stages before fusion (new default: 2)")
        parser.add_argument("--gram-context", choices=("static", "observed", "exchange"),
                            help="exchange: Gram-affinity message passing between observed streams (new default); "
                                 "observed / static: earlier weight-modulating Grams")
    else:
        parser.add_argument("--task", choices=tuple(TASK_OUTPUTS))
        parser.add_argument("--checkpoint", type=Path, help="A complete try pretraining checkpoint")
        parser.add_argument("--mode", choices=("finetune", "linear", "scratch"))
        parser.add_argument("--encoder-lr-scale", type=float)
    return parser


def _json_options(args):
    if args.config is None:
        return {}
    with args.config.open(encoding="utf-8-sig") as handle:
        options = json.load(handle)
    if not isinstance(options, dict):
        raise ValueError("--config must contain a JSON object")
    return options


def _cli_overrides(args):
    # Named flags are easiest to read and take priority over --set.
    result = list(args.set)
    mappings = {
        "workers": "data.num_workers", "batch_size": "optim.batch_size",
        "device": "runtime.device", "output_root": "runtime.output_root",
        "epochs": "optim.epochs", "grad_accum": "optim.grad_accum",
        "lr": "optim.lr", "precision": "optim.precision", "seed": "seed",
        "samples_per_epoch": "data.train_samples", "fusion": "model.fusion",
        "branch_stages": "model.branch_mora.stages", "gram_context": "model.branch_mora.gram_context",
        "pretrain_consistency": "masking.consistency_weight",
        "downstream_consistency": "downstream.consistency_weight",
        "task": "downstream.task", "mode": "downstream.mode",
        "checkpoint": "downstream.checkpoint", "encoder_lr_scale": "downstream.encoder_lr_scale",
    }
    for argument, dotted in mappings.items():
        value = getattr(args, argument, None)
        if value is not None:
            result.append(dotted + "=" + json.dumps(str(value) if isinstance(value, Path) else value))
    backbone = getattr(args, "backbone", None)
    if backbone is not None:
        result += ["model.branch_mora.enabled=" + json.dumps(backbone != "original"),
                   "model.branch_mora.shared=" + json.dumps(backbone != "private")]
    return result


def resolve_training(kind, args):
    """Resolve a fresh run or resume, without opening datasets or writing files."""
    from .checkpoints import load_checkpoint, check_resume_compatibility
    from .config import DEFAULTS, _merge, apply_overrides, load_config, model_defaults

    options, overrides = _json_options(args), _cli_overrides(args)
    if args.resume:
        if getattr(args, "checkpoint", None):
            raise ValueError("--resume and --checkpoint have different meanings; choose one")
        saved = load_checkpoint(args.resume)
        if saved["kind"] != kind:
            raise ValueError("Resume checkpoint belongs to a different training stage")
        cfg = load_config(args.config, overrides, base=saved["config"])
        check_resume_compatibility(saved["config"], cfg)
        if cfg["runtime"]["output_root"] != saved["config"]["runtime"]["output_root"]:
            raise ValueError("Resume writes beside last.pt; --output-root cannot relocate an existing run")
        return cfg

    cfg = copy.deepcopy(DEFAULTS)
    _merge(cfg, settings.COMMON)
    _merge(cfg, settings.PRETRAIN if kind == "pretrain" else settings.FINETUNE)
    if kind == "downstream":
        # Resolve the task before selecting its batch/accumulation defaults.
        probe = copy.deepcopy(cfg)
        _merge(probe, options)
        apply_overrides(probe, overrides)
        task = probe["downstream"]["task"]
        if task not in settings.TASKS:
            raise ValueError(f"Unknown downstream task: {task}")
        _merge(cfg, settings.TASKS[task])
    _merge(cfg, options)
    apply_overrides(cfg, overrides)

    if kind == "downstream":
        task = cfg["downstream"]
        if task.get("init_checkpoint"):
            raise ValueError("Fresh six-task training uses a pretraining checkpoint; use --resume for continuation")
        if task["mode"] != "scratch":
            if not task["checkpoint"]:
                raise ValueError("Set PRETRAIN_CHECKPOINT in default.py or pass --checkpoint /path/to/pretrain_best.pt")
            from .config import PROJECT_DIR
            path = Path(os.path.expandvars(str(task["checkpoint"]))).expanduser()
            path = path if path.is_absolute() else PROJECT_DIR / path
            saved = load_checkpoint(path)
            if saved["kind"] != "pretrain":
                raise ValueError("Fine-tuning requires PRETRAIN weights, not another task's best.pt")
            source_model = model_defaults(saved["config"]["model"])
            # COMMON's model describes NEW pretraining. Loaded weights own their architecture.
            cfg["model"] = copy.deepcopy(source_model)
            _merge(cfg["model"], options.get("model", {}), "model.")
            apply_overrides(cfg, [item for item in overrides if item.startswith("model.")])
            if cfg["model"] != source_model:
                raise ValueError("Do not change a pretrained model's configuration; its architecture is read from the checkpoint")
        elif task["checkpoint"]:
            raise ValueError("scratch requires downstream.checkpoint=null (no pretraining weights)")
    cfg = load_config(base=cfg)
    stage = "pretrain" if kind == "pretrain" else "finetune/" + cfg["downstream"]["task"]
    cfg["runtime"]["output_root"] = str(Path(cfg["runtime"]["output_root"]) / stage)
    return cfg


def check_training(cfg, kind, resume=None):
    """Small real I/O/model check. No output directory, checkpoint or optimizer is created."""
    import torch
    from torch.utils.data import default_collate
    from .checkpoints import initialize_encoder, load_checkpoint
    from .data import BenchmarkDataset, RandomCropDataset
    from .engine import setup, forward_loss
    from .model import LunarMaskedModel, LunarTaskModel
    from .numerics import strict_numeric

    device = setup(cfg)
    task = cfg["downstream"]
    model = (LunarMaskedModel(cfg["model"]) if kind == "pretrain" else
             LunarTaskModel(cfg["model"], task["task"], task["mode"], task["feature_dropout"]))
    if resume:
        model.load_state_dict(load_checkpoint(resume)["model"], strict=True)
    elif kind == "downstream" and task["mode"] != "scratch":
        initialize_encoder(model.encoder, load_checkpoint(task["checkpoint"]))
    elif cfg["runtime"].get("init_weights"):
        from .backbone import load_official_weights
        load_official_weights(model.encoder.backbone, cfg["runtime"]["init_weights"])
    model.to(device).eval()
    from .engine import move_batch
    with torch.no_grad(), strict_numeric():
        for split in ("train", "val"):
            dataset = (RandomCropDataset(cfg["data"], split, cfg["seed"],
                                        mmap_read_hint=cfg["runtime"]["mmap_read_hint"])
                       if kind == "pretrain" else
                       BenchmarkDataset(cfg["data"], task["task"], split, task["drop_groups"]))
            try:
                batch = move_batch(default_collate([dataset[0]]), device)
                from .engine import consistency_weight
                pairs = (torch.Generator(device=device).manual_seed(cfg["seed"])
                         if split == "train" and consistency_weight(cfg, kind) > 0 else None)
                loss, prediction, _ = forward_loss(model, batch, cfg, kind, pair_generator=pairs)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite check loss")
                print(f"CHECK {split}: patches={len(dataset)}, output={tuple(prediction.shape)}, loss={float(loss):.6f}")
            finally:
                dataset.close() if hasattr(dataset, "close") else dataset.store.close()
    print("Check passed. No training updates or files were written.")


def training_main(kind, argv=None):
    parser = training_parser(kind)
    args = parser.parse_args(argv)
    try:
        cfg = resolve_training(kind, args)
        if args.dry_run:
            print(json.dumps(cfg, indent=2, ensure_ascii=False))
            return
        if args.check:
            check_training(cfg, kind, args.resume)
            return
        from .engine import train
        branch = cfg["model"]["branch_mora"]
        print(f"{kind}: batch={cfg['optim']['batch_size']}, accumulation={cfg['optim']['grad_accum']}, "
              f"workers={cfg['data']['num_workers']}, fusion={cfg['model']['fusion']}, "
              f"branch_mora={branch['enabled']}, branch_stages={branch['stages'] if branch['enabled'] else 0}, "
              f"gram={branch['enabled'] and branch['shared']}, "
              f"gram_context={branch['gram_context'] if branch['enabled'] and branch['shared'] else 'none'}",
              flush=True)
        directory = train(cfg, kind, args.resume)
        print(f"Best checkpoint: {directory / 'best.pt'}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


def evaluation_main(argv=None):
    parser = _parser("Evaluate original inputs or a complete seven-group robustness suite")
    parser.add_argument("--checkpoint", type=Path, default=settings.EVALUATE["checkpoint"])
    parser.add_argument("--split", choices=("val", "test"), default=settings.EVALUATE["split"])
    parser.add_argument("--all-drops", action=argparse.BooleanOptionalAction, default=settings.EVALUATE["all_drops"])
    parser.add_argument("--drop-group", choices=GROUPS, action="append", default=[])
    parser.add_argument("--max-batches", type=int, help="Smoke test only; report explicitly marks partial evaluation")
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        parser.error("Set EVALUATION_CHECKPOINT in default.py or pass --checkpoint /path/to/best.pt")
    if args.all_drops and args.drop_group:
        parser.error("Use --all-drops OR --drop-group, not both")
    try:
        from .evaluation import resolve_evaluation, evaluate_suite
        cfg, payload = resolve_evaluation(args, _json_options(args))
        if args.dry_run:
            print(json.dumps({"checkpoint": str(args.checkpoint), "split": args.split,
                              "all_drops": args.all_drops, "drop_groups": args.drop_group,
                              "config": cfg}, indent=2, ensure_ascii=False))
            return
        evaluate_suite(args.checkpoint, payload, cfg, args.split,
                       args.all_drops, args.drop_group, args.max_batches)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
