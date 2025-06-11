#!/usr/bin/env python3


from PIL import Image
import os
import math
from collections import defaultdict

# --- Settings ---
plot_dir = "plots"  # where your PNGs are
output_pdf = "plots_by_feature.pdf"
plots_per_row = 2
image_size = (600, 400)  # Resize images to fit nicely on page

# --- Group plots by feature ---
grouped = defaultdict(list)
for filename in os.listdir(plot_dir):
    if filename.endswith(".png") and "plot_feat" in filename:
        # Extract plotting_feature from filename like 'plot_feat5_if_feat3_eq_1.png'
        try:
            feat = int(filename.split("plot_feat")[1].split("_")[0])
            grouped[feat].append(os.path.join(plot_dir, filename))
        except Exception:
            continue

# --- Create pages ---
pages = []
for feat, file_list in sorted(grouped.items()):
    images = [Image.open(f).convert("RGB").resize(image_size) for f in sorted(file_list)]
    rows = math.ceil(len(images) / plots_per_row)
    page_width = image_size[0] * plots_per_row
    page_height = image_size[1] * rows
    page = Image.new("RGB", (page_width, page_height), color="white")

    for idx, img in enumerate(images):
        x = (idx % plots_per_row) * image_size[0]
        y = (idx // plots_per_row) * image_size[1]
        page.paste(img, (x, y))

    pages.append(page)

# --- Save all pages into one PDF ---
if pages:
    pages[0].save(output_pdf, save_all=True, append_images=pages[1:])
else:
    print("no plots found")
