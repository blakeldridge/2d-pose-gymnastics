import os

BG_DIR = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data/backgrounds/")

background_paths = [os.path.join(BG_DIR, f) for f in os.listdir(BG_DIR)]

for i in range(len(background_paths)):
    os.rename(background_paths[i], os.path.join(BG_DIR, f"background{i+1}.jpg"))