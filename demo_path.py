import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import argparse
import datetime
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

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="TRELLIS.2 image-to-3D demo")
parser.add_argument("image_path", help="Path to the input image (e.g. assets/example_image/T.png)")
args = parser.parse_args()

input_path = os.path.abspath(args.image_path)
if not os.path.isfile(input_path):
    print(f"Error: input image not found: {input_path}", file=sys.stderr)
    sys.exit(1)

# --- Create timestamped result folder ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = os.path.join(os.path.dirname(input_path), f"result_{timestamp}")
os.makedirs(result_dir, exist_ok=True)

run_start = time.time()

def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    log_lines.append(line)

log_lines = []
log(f"Input image : {input_path}")
log(f"Result folder: {result_dir}")

# 1. Setup Environment Map
log("Loading environment map...")
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
log("Loading pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Load Image & Run
log("Running pipeline...")
t0 = time.time()
image = Image.open(input_path)
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)  # nvdiffrast limit
log(f"Pipeline finished in {time.time() - t0:.1f}s")

# 4. Render Video
log("Rendering video...")
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
mp4_path = os.path.join(result_dir, "sample.mp4")
imageio.mimsave(mp4_path, video, fps=15)
log(f"Video saved : {mp4_path}")

# 5. Export to GLB
log("Exporting GLB...")
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
log(f"GLB saved   : {glb_path}")

total = time.time() - run_start
log(f"Total runtime: {total:.1f}s")

# 6. Write log file
log_path = os.path.join(result_dir, "run.log")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"TRELLIS.2 demo run\n")
    f.write(f"Timestamp   : {timestamp}\n")
    f.write(f"Input image : {input_path}\n")
    f.write(f"Result dir  : {result_dir}\n")
    f.write(f"MP4         : {mp4_path}\n")
    f.write(f"GLB         : {glb_path}\n")
    f.write(f"Total time  : {total:.1f}s\n\n")
    f.write("--- Log ---\n")
    f.write("\n".join(log_lines) + "\n")
print(f"Log written : {log_path}")