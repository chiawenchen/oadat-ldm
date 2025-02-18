from PIL import Image

# Open the two PNG images
img_top = Image.open("dm_scd_batch_classifier_guidance_s=25_f=500_b=500_seed=42_denoised.png")
img_bottom = Image.open("dm_scd_batch_classifier_guidance_2_s=25_f=500_b=500_seed=42_denoised.png")

# Get sizes (width and height) of each image
width_top, height_top = img_top.size
width_bottom, height_bottom = img_bottom.size

# Determine the width and height of the new combined image
# The width is the maximum of the two widths,
# and the height is the sum of the two heights.
combined_width = max(width_top, width_bottom)
combined_height = height_top + height_bottom

# Create a new blank image with the combined dimensions.
# Choose "RGB" mode if your images don't have transparency,
# or "RGBA" if they do.
combined_image = Image.new("RGB", (combined_width, combined_height))

# Paste the top image at position (0, 0)
combined_image.paste(img_top, (0, 0))

# Paste the bottom image just below the top image.
# The x-coordinate is 0, and the y-coordinate is the height of the top image.
combined_image.paste(img_bottom, (0, height_top))

# Save the combined image to a new file
combined_image.save("dm_scd_batch_classifier_guidance_f500_b500_seed42_scale25.png")
