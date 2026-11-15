from collections import OrderedDict

GROUP_CHANNELS = OrderedDict([
    ("surface", ["wac_morphology", "elevation", "slope", "roughness"]),
    ("thermal", ["diviner_tbol_midnight", "diviner_temp_night",
                 "rock_abundance", "christiansen_feature"]),
    ("spectral_m3", ["m3_750", "m3_950", "m3_1000", "m3_1250",
                     "m3_1580", "m3_2000", "m3_2817", "m3_2857"]),
    ("gravity", ["grail_freeair", "grail_bouguer", "grail_uncertainty"]),
    ("radar", ["minirf_cpr", "minirf_s1"]),
    ("hapke", ["wac_hapke_415nm", "wac_hapke_566nm",
               "wac_hapke_604nm", "wac_hapke_689nm"]),
    ("composition", ["clementine_uvvis_750nm", "lpgrs_tio2", "lpgrs_feo"]),
])
CHANNELS = [name for group in GROUP_CHANNELS.values() for name in group]
GROUPS = tuple(GROUP_CHANNELS)
GROUP_INDICES = {g: tuple(CHANNELS.index(c) for c in cs)
                 for g, cs in GROUP_CHANNELS.items()}
N_CHANNELS = len(CHANNELS)
TASK_OUTPUTS = {"geology": 49, "age": 5, "composition": 2,
                "crossmodal": 4, "mare": 2, "craters": 2}
REGRESSION_CHANNELS = {"composition": ("lpgrs_tio2", "lpgrs_feo"),
                       "crossmodal": tuple(GROUP_CHANNELS["thermal"])}
