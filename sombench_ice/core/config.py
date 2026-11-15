def model_defaults(options):
    return {**options, "branch_mora": {
        "enabled": False, "rank": 4, "alpha": 0.01, "shared": True, "stages": 1,
        "gram_context": "static", **options.get("branch_mora", {})}}
