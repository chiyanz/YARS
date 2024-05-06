"""
BASE MODEL: FAILED ['LXT4hCf1lRyUeM4HDBaSvg.jpg'] - 1 of 50K is not bad :D
"""


# import packages
print("Loading modules...")
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import requests
import os
from os import listdir
import json
import gc

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device: %s" % device)

# config
mode="start" # start, continue
root_dir = "/scratch/jl12734/yars/"
image_dir = root_dir + "photos/"
chunk_size = 256

# load model and processor
print("Loading model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def gather_photos(label: str, no_captions: bool = False):
    """
    takes the label (category of images to fetch) and filters photos.json for images with that label
    filters for images that 
    returns: list of photo objects (photo_id, label, caption, etc.)
    """ 
    with open(os.path.join(root_dir, 'photos.json'), 'r') as file:
        all_photos = []
        for line in file:
            line = line.rstrip()
            try:
                photo_record = json.loads(line)
                # filers for only ones wiht captions by default
                has_captions = len(photo_record['caption']) > 0 or no_captions 
                if photo_record['label'] == label and has_captions:
                    all_photos.append(photo_record["photo_id"] + ".jpg")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} at line: {line}")
        return all_photos
all_photos = gather_photos('food')

def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

# if running for the first time
if mode == "start":
    print("Reading photos.json...")
    chunks = divide_chunks(all_photos, chunk_size)
    
    paths  = ["empty"]
    failed = []
    features = torch.zeros(1, 512)
    for chunk in chunks:
        images = []
        print("\rLoading new batch...", end="")
        for path in chunk:
            try:
                images.append(Image.open(image_dir + path))
                paths.append(path)
            except Exception as e:
                print(e)
                failed.append(path)
        print("\rPreprocessing new batch...", end="")
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device) 
        print("\rProcessing new batch...", end="")
        image_features = model.get_image_features(**inputs).detach().cpu().clone()
        features = torch.cat((features, image_features), dim=0)
        print("\rProgress:                    \t\t\t%d of %d, mem allocated: %fGB" % 
              (len(paths) + len(failed), len(all_photos), (torch.cuda.memory_reserved(0)/1024/1024/1024)), end="")

        del inputs
        del image_features
        gc.collect()
        torch.cuda.empty_cache()
        
        # write data to file
        path_out    = "out/paths.txt"
        feature_out = "out/features.pt" # this is a binary file, shape is (n_images, 768)
        
        with open(path_out, "w") as f:
            for path in paths:
                f.write(path + "\n")
        
        with open(feature_out, "wb") as f:
            torch.save(features, f)
    print()
    print("Failed %d images:" % len(failed))
    print(failed)