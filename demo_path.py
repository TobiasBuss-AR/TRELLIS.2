import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import argparse
import datetime
import logging
import shutil
import sys
import time
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# --- Constants ---
WORKDIR    = "/data/workdir"
S3_PREFIX  = "/s3"          # local mount point of the S3 bucket

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="TRELLIS.2 image-to-3D demo")
parser.add_argument("image_path", help="Path to input image — local or under /s3/...")
args = parser.parse_args()

given_path  = os.path.abspath(args.image_path)
use_s3      = given_path.startswith(S3_PREFIX + os.sep) or given_path == S3_PREFIX
timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Working result dir is always on fast local storage
result_dir  = os.path.join(WORKDIR, f"result_{timestamp}")
os.makedirs(result_dir, exist_ok=True)

# --- Logger setup (console + file, written live) ---
log_path = os.path.join(result_dir, "run.log")
logger = logging.getLogger("trellis2_demo")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

_fh = logging.FileHandler(log_path, encoding="utf-8")
_fh.setFormatter(fmt)
logger.addHandler(_fh)

_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(fmt)
logger.addHandler(_ch)

run_start = time.time()

# --- Copy input from S3 mount to workdir if needed ---
if use_s3:
    if not os.path.isfile(given_path):
        logger.error(f"Input image not found on S3 mount: {given_path}")
        sys.exit(1)
    local_input = os.path.join(WORKDIR, os.path.basename(given_path))
    logger.info(f"S3 input detected — copying to workdir...")
    logger.info(f"  {given_path}  →  {local_input}")
    shutil.copy2(given_path, local_input)
    input_path = local_input
    # mirror path: result will be copied back here when done
    s3_result_dir = os.path.join(os.path.dirname(given_path), f"result_{timestamp}")
else:
    if not os.path.isfile(given_path):
        logger.error(f"Input image not found: {given_path}")
        sys.exit(1)
    input_path    = given_path
    s3_result_dir = None

logger.info(f"Input image  : {input_path}")
logger.info(f"Result folder: {result_dir}")
logger.info(f"Log file     : {log_path}")

# 1. Setup Environment Map
logger.info("Loading environment map...")
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
logger.info("Loading pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Load Image & Run
logger.info("Running pipeline...")
t0 = time.time()
image = Image.open(input_path)
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)  # nvdiffrast limit
logger.info(f"Pipeline finished in {time.time() - t0:.1f}s")

# 4. Render Video
logger.info("Rendering video...")
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
mp4_path = os.path.join(result_dir, "sample.mp4")
imageio.mimsave(mp4_path, video, fps=15)
logger.info(f"Video saved : {mp4_path}")

# 5. Export to GLB
logger.info("Exporting GLB...")
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb_path = os.path.join(result_dir, "sample.glb")
glb.export(glb_path, extension_webp=True)
logger.info(f"GLB saved   : {glb_path}")

total = time.time() - run_start
logger.info(f"Total runtime: {total:.1f}s")

# --- Copy results back to S3 mount ---
if use_s3:
    logger.info(f"Copying results back to S3 mount: {s3_result_dir}")

    def retry(fn, description, retries=5, delay=5):
        """Call fn(), retry on exception with increasing wait."""
        for attempt in range(1, retries + 1):
            try:
                fn()
                return
            except Exception as e:
                if attempt == retries:
                    logger.error(f"FAILED after {retries} attempts — {description}: {e}")
                    raise
                wait = delay * attempt
                logger.warning(f"Attempt {attempt}/{retries} failed ({description}): {e} — retrying in {wait}s...")
                time.sleep(wait)

    retry(lambda: os.makedirs(s3_result_dir, exist_ok=True),
          f"makedirs {s3_result_dir}")

    for fname in os.listdir(result_dir):
        src = os.path.join(result_dir, fname)
        dst = os.path.join(s3_result_dir, fname)
        logger.info(f"  uploading {fname}...")
        retry(lambda s=src, d=dst: shutil.copy2(s, d),
              f"copy {fname}")

    logger.info(f"All results copied to: {s3_result_dir}")

logger.info("Done.")