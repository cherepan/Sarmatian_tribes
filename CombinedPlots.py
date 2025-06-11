#!/usr/bin/env python3


from PIL import Image
import os

# Directory containing your PNGs
plot_dir = "plots"  # change if your plots are elsewhere

# Collect all .png files and sort them
image_files = sorted([f for f in os.listdir(plot_dir) if f.endswith(".png")])
if not image_files:
        exit()


        # Open images and convert to RGB
images = [Image.open(os.path.join(plot_dir, f)).convert("RGB") for f in image_files]

# Save as multi-page PDF
output_pdf = "combined_plots.pdf"
images[0].save(output_pdf, save_all=True, append_images=images[1:])

print(f"Combined {len(images)} plots into {output_pdf}")
