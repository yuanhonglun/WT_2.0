"""Physical and chemical constants for mass spectrometry."""

# ---------------------------------------------------------------------------
# Proton and electron masses
# ---------------------------------------------------------------------------
PROTON_MASS = 1.00727646677  # Da
ELECTRON_MASS = 0.00054857990924  # Da

# ---------------------------------------------------------------------------
# Common neutral losses (Da)
# ---------------------------------------------------------------------------
MASS_H2O = 18.01056468
MASS_CO2 = 43.98982924
MASS_NH3 = 17.02654910
MASS_CO = 27.99491462
MASS_HCOOH = 46.00547984  # formic acid
MASS_CH3COOH = 60.02112937  # acetic acid

# ---------------------------------------------------------------------------
# Isotope mass differences (Da)
# ---------------------------------------------------------------------------
C13_DELTA = 1.003355  # 13C - 12C
N15_DELTA = 0.997035  # 15N - 14N
S34_DELTA = 1.995796  # 34S - 32S
O18_DELTA = 2.004245  # 18O - 16O
CL37_DELTA = 1.997050  # 37Cl - 35Cl
BR81_DELTA = 1.997953  # 81Br - 79Br

# All classic isotope deltas with labels
ISOTOPE_DELTAS = {
    "C13": C13_DELTA,
    "N15": N15_DELTA,
    "S34": S34_DELTA,
    "O18": O18_DELTA,
    "Cl37": CL37_DELTA,
    "Br81": BR81_DELTA,
}

# Maximum isotope steps for C13 series
MAX_ISOTOPE_STEP = 4

# ---------------------------------------------------------------------------
# Adduct definitions: (name, delta_mass_from_neutral, charge, multiplier)
# delta_mass = mass_of_adduct_ion - neutral_mass * multiplier
# For [M+H]+:  adduct_mz = (M + PROTON_MASS) / 1
# For [2M+H]+: adduct_mz = (2*M + PROTON_MASS) / 1
# ---------------------------------------------------------------------------
ADDUCTS_POSITIVE = [
    {"name": "[M+H]+",       "mass": PROTON_MASS,              "charge": 1, "mult": 1},
    {"name": "[M+Na]+",      "mass": 22.989218,                "charge": 1, "mult": 1},
    {"name": "[M+K]+",       "mass": 38.963158,                "charge": 1, "mult": 1},
    {"name": "[M+NH4]+",     "mass": 18.033826,                "charge": 1, "mult": 1},
    {"name": "[M+H-H2O]+",   "mass": PROTON_MASS - MASS_H2O,  "charge": 1, "mult": 1},
    {"name": "[M+2H]2+",     "mass": 2 * PROTON_MASS,          "charge": 2, "mult": 1},
    {"name": "[2M+H]+",      "mass": PROTON_MASS,              "charge": 1, "mult": 2},
    {"name": "[2M+Na]+",     "mass": 22.989218,                "charge": 1, "mult": 2},
]

ADDUCTS_NEGATIVE = [
    {"name": "[M-H]-",       "mass": -PROTON_MASS,             "charge": 1, "mult": 1},
    {"name": "[M+Cl]-",      "mass": 34.969402,                "charge": 1, "mult": 1},
    {"name": "[M+FA-H]-",    "mass": MASS_HCOOH - PROTON_MASS, "charge": 1, "mult": 1},
    {"name": "[M+HAc-H]-",   "mass": MASS_CH3COOH - PROTON_MASS, "charge": 1, "mult": 1},
    {"name": "[M-H2O-H]-",   "mass": -PROTON_MASS - MASS_H2O, "charge": 1, "mult": 1},
    {"name": "[M-2H]2-",     "mass": -2 * PROTON_MASS,         "charge": 2, "mult": 1},
    {"name": "[2M-H]-",      "mass": -PROTON_MASS,             "charge": 1, "mult": 2},
]


def mz_from_neutral(neutral_mass: float, adduct: dict) -> float:
    """Calculate m/z from neutral mass and adduct definition."""
    return (neutral_mass * adduct["mult"] + adduct["mass"]) / adduct["charge"]


def neutral_from_mz(mz: float, adduct: dict) -> float:
    """Calculate neutral mass from observed m/z and adduct definition."""
    return (mz * adduct["charge"] - adduct["mass"]) / adduct["mult"]
