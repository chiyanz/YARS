import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle as pk
from transformers import CLIPProcessor, CLIPModel
from sklearn.neighbors import NearestNeighbors
import torch
import os
from PIL import Image
import json
import gc
import requests

#import PCA model
pca_reload = pk.load(open("pca.pkl",'rb'))

# import the kmeans model
kmeans_reload = pk.load(open("kmeans.pkl", 'rb'))

root_dir = '../'

# load in photots data
photos_data = {}
with open(os.path.join(root_dir, 'photos.json'), 'r') as file:
        for line in file:
            line = line.rstrip()
            try:
                photo_record = json.loads(line)
                photos_data[photo_record['photo_id']] = photo_record
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} at line: {line}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def predict_img(file_path):
    images = Image.open(requests.get(file_path, stream=True).raw)
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device) 
    image_features = model.get_image_features(**inputs).detach().cpu().clone()
    gc.collect()
    torch.cuda.empty_cache()
    
    transformed_embedding = pca_reload.transform(image_features.numpy())
    new_label = kmeans_reload.predict(transformed_embedding)
    
    # find k nearest neighbors
    k = 150
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    cluster_embeddings = torch.load(f"{new_label[0]}/embeddings.pt")
    with open(f"{new_label[0]}/images.txt", 'r') as file:
        cluster_images = file.readlines()
    
    nbrs.fit(cluster_embeddings)
    distances, indices = nbrs.kneighbors(transformed_embedding)
        
    neighbor_captions = []
        
    for idx in indices[0]:
        image_id = cluster_images[idx].split('/')[-1][:-5]
        neighbor_captions.append(photos_data[image_id]["caption"])
    
    from collections import Counter
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.corpus import wordnet as wn
    
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    def tokenize_and_count(captions):
        # Regular expression to find words
        word_pattern = re.compile(r'\b\w+\b')
        text = ' '.join(captions).lower()
        words = word_pattern.findall(text)
        frequency = Counter(words)
        return frequency
    
    k = 3 # top-k labels
    im = Image.open(requests.get(file_path, stream=True).raw)
    freq = tokenize_and_count(neighbor_captions)
    filtered_counts = {word: count for word, count in freq.items() if word not in stop_words}
    top_k_labels = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:k]
    total_count = sum(count for _, count in top_k_labels)
    top_k_confidence = [(item, (count / total_count) * 100) for item, count in top_k_labels]
    
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as necessary

    # Display the image
    ax.imshow(im)
    ax.axis('off') 
    k_labels = f'top-{k} labels using KNN: \n' + '\n'.join([f'{label}: {percentage:.2f}%' for label, percentage in top_k_confidence])
    text_x = 0.01  # X position in figure units (0 left, 1 right)
    text_y = 0.01  # Y position in figure units (0 bottom, 1 top)
    ax.text(text_x, text_y, k_labels, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', color='white')
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.show()

# interactive inputs
while True:
    file_path = input("Please enter the file path: ")
    # Check if the file exists
    predict_img(file_path)
    
    continue_prompt = input("Do you want to predict another image? (yes/no): ")
    if continue_prompt.lower() != 'yes':
        break