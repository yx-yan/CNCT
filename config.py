# Pipeline configuration for projection.py and fdk.py

# --- Paths ---
DATA_DIR = "/projects/_hdd/CTdata/AbdomenCT-1K-ImagePart1"
OUTPUT_DIR = "output"
MAX_CASES = 5             # max number of cases to process; None = all cases

# --- Acquisition ---
N_ANGLES = 1000          # number of projection angles (full 360°)

# --- Geometry scaling ---
DSO_SCALE = 5            # DSO = max_radius * DSO_SCALE  (source outside object)
DSD_SCALE = 1.5          # DSD = DSO * DSD_SCALE

# --- Detector ---
DETECTOR_COL_MARGIN = 1.5   # dDetector[1] *= this to prevent truncation artifacts

# --- Forward projection ---
ACCURACY = 0.5           # tigre.Ax accuracy parameter (lower = faster, less accurate)

# --- HU → linear attenuation conversion ---
MU_WATER = 0.02          # mm⁻¹, linear attenuation of water at ~70 keV

# --- Visualisation ---
PROJ_SAVE_EVERY = 10     # save every Nth projection angle as PNG
IMAGE_DPI = 150          # DPI for all saved PNGs
