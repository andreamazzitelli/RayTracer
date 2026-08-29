# In a Colab cell, after `!pip install numpy` and uploading/cloning your src/ tree,
# with sys.path including your src directory:

import cProfile
import pstats
import io
import time
import pandas as pd
import matplotlib.pyplot as plt

from examples.render_scene import build_scene, build_camera
from raytracer.rendering.renderer import render

def profile_render(hsize: int, vsize: int) -> tuple[pd.DataFrame, float]:
    world = build_scene()
    camera = build_camera(hsize, vsize)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    render(camera, world)
    profiler.disable()
    elapsed = time.perf_counter() - start

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(15)

    # Parse pstats' own text table into a DataFrame for display/plotting
    rows = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        rows.append({
            "function": f"{func[0]}:{func[1]}({func[2]})",
            "ncalls": nc,
            "tottime": tt,
            "cumtime": ct,
        })
    df = pd.DataFrame(rows).sort_values("cumtime", ascending=False).head(15)
    return df, elapsed


configs = [(50, 25), (100, 50), (200, 100), (400, 200)]
summary = []
per_config_frames = {}

for hsize, vsize in configs:
    df, elapsed = profile_render(hsize, vsize)
    per_config_frames[(hsize, vsize)] = df
    summary.append({"resolution": f"{hsize}x{vsize}", "pixels": hsize * vsize, "seconds": elapsed})
    print(f"--- {hsize}x{vsize} ({elapsed:.3f}s) ---")
    display(df)  # Colab renders this as a nice table

summary_df = pd.DataFrame(summary)
display(summary_df)

# Visualize how render time scales with pixel count
plt.plot(summary_df["pixels"], summary_df["seconds"], marker="o")
plt.xlabel("Pixels")
plt.ylabel("Render time (s)")
plt.title("Render time vs. resolution")
plt.show()