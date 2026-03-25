import os
from deblur.deblur import deblur, init_model

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLURRY_DIR = os.path.join(DIR, "data/test/images")
DEBLURRED_DIR = os.path.join(DIR, "data/test/deblurred_images")

os.makedirs(DEBLURRED_DIR, exist_ok=True)

def deblur_images():
	print("Deblurring Images ...")
	blurry_files = sorted([
		f for f in os.listdir(BLURRY_DIR)
		if f.lower().endswith((".png", ".jpg", ".jpeg"))
	])

	NAFNet = init_model()

	print("Model Created...")
	
	for idx, blurry_name in enumerate(blurry_files):
		blurry_path = os.path.join(BLURRY_DIR, blurry_name)
		deblurred_name = "deblurred_" + blurry_name.replace("blurry_", "")
		deblurred_path = os.path.join(DEBLURRED_DIR, deblurred_name)
		deblur(blurry_path, deblurred_path, NAFNet)
		print(f"Image {idx + 1} deblurred!")

	print("Images Deblurred!")

if __name__ == "__main__":
	deblur_images()
