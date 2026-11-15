SCHEMA_ID = "sombench_ice_sincos_9ch_v1"
CHANNELS = ("DICE", "TMAX", "SLOPE", "CUR", "ASP_SIN", "ASP_COS",
            "LPSR", "LPSR_DIS", "LPSR_DEN")
GROUP_INDICES = {"thermal": (0, 1), "terrain": (2, 3, 4, 5), "shadow": (6, 7, 8)}
GROUPS = tuple(GROUP_INDICES)
N_CHANNELS = len(CHANNELS)
TASK_OUTPUTS = {"ice": 1}
LAYERS = (("DICE", 1), ("TMAX", 1), ("SLOPE", 1), ("CUR", 1),
          ("ASP_SIN_COS", 2), ("LPSR", 1), ("LPSR_DIS", 1), ("LPSR_DEN", 1))
SCENARIOS = {
    "original": (),
    "drop_thermal": ("thermal",), "drop_terrain": ("terrain",),
    "drop_shadow": ("shadow",), "only_thermal": ("terrain", "shadow"),
    "only_terrain": ("thermal", "shadow"), "only_shadow": ("thermal", "terrain"),
}
