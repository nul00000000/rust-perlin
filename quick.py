import os
from PIL import Image
import numpy as np

for filename in os.listdir("./images"):
	img = Image.open(os.path.join("./images", filename))

	img = img.convert("RGBA")
	data = np.array(img)

	r, g, b, a = np.rollaxis(data, axis=-1)
	mask = (r == 0) & (g == 0) & (b == 0)

	data[..., 3][mask] = 0

	img = Image.fromarray(data, "RGBA")

	img.save(os.path.join("./images", "trans." + filename), "PNG")