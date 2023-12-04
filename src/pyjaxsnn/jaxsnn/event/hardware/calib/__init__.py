# pylint: disable=line-too-long
from typing import NamedTuple


class WaferConfig(NamedTuple):
    file: str
    name: str
    weight_scaling: float


folder = "jaxsnn/event/hardware/calib/"


W_69_F0 = WaferConfig(
    folder
    + "calibration_W69F0_leak80_th150_reset80_taus-6us_taum-6us_trefrac-2.0us_isyningm-500_sdbias-1000.pbin",
    "W69 F0",
    48.0,
)
W_66_F3 = WaferConfig(
    folder
    + "calibration_W66F3_leak80_th150_reset80_taus-6us_taum-6us_trefrac-2.0us_isyningm-500_sdbias-1000.pbin",
    "W66 F3",
    48.0,
)

W_66_F3_TAU_MEM_FACTOR_2 = WaferConfig(
    folder
    + "calibration_W66F3_leak80_th150_reset80_taus-6us_taum-12us_trefrac-2.0us_isyningm-500_sdbias-1000.pbin",
    "W66 F3",
    48.0,
)
W_69_F0_TAU_MEM_FACTOR_2 = WaferConfig(
    folder
    + "calibration_W69F0_leak80_th150_reset80_taus-6us_taum-12us_trefrac-2.0us_isyningm-500_sdbias-1000.pbin",
    "W69 F0",
    48.0,
)

W_69_F0_LONG_REFRAC = WaferConfig(
    folder
    + "calibration_W69F0_leak80_th150_reset80_taus-6us_taum-12us_trefrac-30.0us_isyningm-500_sdbias-1000.pbin",
    "W69 F0 long refrac",
    56.0,
)
W_63_F3_LONG_REFRAC = WaferConfig(
    folder
    + "calibration_W63F3_leak80_th150_reset80_taus-6us_taum-12us_trefrac-30.0us_isyningm-500_sdbias-1000.pbin",
    "W63 F3 long refrac",
    48.0,
)

W_69_F0_LONG_REFRAC_130_THRESHOLD = WaferConfig(
    folder
    + "calibration_W69F0_leak80_th130_reset80_taus-6us_taum-12us_trefrac-30.0us_isyningm-500_sdbias-1000.pbin",
    "W69 F0 long refrac 130 treshold",
    32.0,
)

W_63_F3_LONG_REFRAC_130_THRESHOLD = WaferConfig(
    folder
    + "calibration_W63F3_leak80_th130_reset80_taus-6us_taum-12us_trefrac-30.0us_isyningm-500_sdbias-1000.pbin",
    "W63 F3 long refrac 130 treshold",
    30.0,
)
