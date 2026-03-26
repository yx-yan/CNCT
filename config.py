# Pipeline configuration for projection.py and fdk.py

# --- Paths ---
DATA_DIR = "/projects/_hdd/CTdata/AbdomenCT-1K-ImagePart1"
OUTPUT_DIR = "output"
MAX_CASES = 20             # max number of cases to process; None = all cases

# --- Acquisition ---
N_ANGLES = 60          # number of projection angles (full 360°)

# --- Geometry scaling ---
DSO_SCALE = 5            # DSO = max_radius * DSO_SCALE  (source outside object)
DSD_SCALE = 1.5          # DSD = DSO * DSD_SCALE

# --- Detector ---
DETECTOR_COL_MARGIN = 1.5   # dDetector[1] *= this to prevent truncation artifacts

# --- Forward projection ---
ACCURACY = 0.5           # tigre.Ax ray-integration step size in voxels (lower = finer steps, more accurate but slower)

# --- FDK reconstruction ---
FDK_FILTER = "shepp_logan"  # filter for FDK backprojection: "ram_lak" | "shepp_logan" | "cosine" | "hamming" | "hann"

# --- HU → linear attenuation conversion ---
MU_WATER = 0.02          # mm⁻¹, linear attenuation of water at ~70 keV

# --- Output ---
SAVE_PNG = False          # save projection and reconstruction slice PNGs
SAVE_NII = False         # save recon_fdk.nii.gz (large; disable to save disk space)

# --- Visualisation ---
PROJ_SAVE_EVERY = 20     # save every Nth projection angle as PNG
IMAGE_DPI = 150          # DPI for all saved PNGs
