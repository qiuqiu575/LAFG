import argparse
import copy


def main(argv=None):
    import ice_settings as settings
    parser = argparse.ArgumentParser(
        description="TRY SomBench ICE: prepare -> train OR finetune -> evaluate. Original six tasks are unchanged.")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare official TIFF layers/splits and train-only normalization once")
    prep.add_argument("--data-dir", default=settings.DATA_DIR, help="Official Ice repository directory")
    prep.add_argument("--cache-dir", default=settings.CACHE_DIR, help="New output directory; never overwritten")
    prep.add_argument("--patch-size", type=int, default=settings.SETTINGS["patch_size"])
    for name in ("train", "finetune", "resume", "evaluate"):
        descriptions = {
            "train": "Supervised ICE training from scratch; this is NOT MAE pretraining",
            "finetune": "Transfer shared TRY backbone; train new input adapters and dense output",
            "resume": "Continue an interrupted ICE run from last.pt with its saved settings",
            "evaluate": "Evaluate original + six missing-group cases with a fixed checkpoint",
        }
        sub = commands.add_parser(name, help=descriptions[name], description=descriptions[name])
        sub.add_argument("--cache-dir", help="Prepared directory (default: ice_settings.py or saved checkpoint)")
        sub.add_argument("--device", help="auto, cpu, cuda, cuda:0, ...")
        sub.add_argument("--num-workers", type=int, help="DataLoader processes")
        sub.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
        if name != "train":
            sub.add_argument("--checkpoint", required=name == "resume", help="Source TRY weights / ICE last.pt / ICE best.pt")
        if name != "resume":
            sub.add_argument("--output-dir", help="New output directory, must not already exist")
            sub.add_argument("--batch-size", type=int)
        if name in ("train", "finetune"):
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--lr", type=float)
            sub.add_argument("--seed", type=int)
            sub.add_argument("--group-drop-prob", type=float, help="Probability of removing ONE input group per training sample")
        if name == "train":
            sub.add_argument("--variant", choices=("debug", "atto", "femto", "pico", "nano", "tiny"))
        if name == "evaluate":
            sub.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare
        prepare(args.data_dir, args.cache_dir, args.patch_size)
        return
    from .engine import evaluate, load_ice, train
    runtime = {key: getattr(args, key) for key in ("device", "precision", "num_workers") if getattr(args, key) is not None}
    if args.command == "evaluate":
        path = args.checkpoint or settings.EVALUATION_CHECKPOINT
        if not path:
            parser.error("evaluate requires --checkpoint ICE_BEST_PT or EVALUATION_CHECKPOINT in ice_settings.py")
        evaluate(path, args.cache_dir, args.split, args.output_dir, runtime, args.batch_size)
        return
    if args.command == "resume":
        cfg = copy.deepcopy(load_ice(args.checkpoint)["config"])
        cfg["runtime"].update(runtime)
        if args.cache_dir:
            cfg["cache_dir"] = args.cache_dir
        train(cfg, resume=args.checkpoint)
        return
    cfg = copy.deepcopy(settings.SETTINGS)
    cfg.update({"cache_dir": args.cache_dir or settings.CACHE_DIR,
                "output_root": settings.OUTPUT_ROOT, "output_dir": args.output_dir})
    cfg["runtime"].update(runtime)
    for key in ("epochs", "batch_size", "lr", "group_drop_prob"):
        value = getattr(args, key)
        if value is not None:
            cfg["training"][key] = value
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.command == "train":
        if args.variant:
            cfg["model"]["variant"] = args.variant
        train(cfg)
    else:
        path = args.checkpoint or settings.PRETRAIN_CHECKPOINT
        if not path:
            parser.error("finetune requires --checkpoint TRY_PRETRAIN_PT or PRETRAIN_CHECKPOINT in ice_settings.py")
        train(cfg, mode="finetune", checkpoint=path)
