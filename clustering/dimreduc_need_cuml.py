import torch
import cuml
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def load_data(embeddings_path, paths_file):
    embeddings = torch.load(embeddings_path)
    with open(paths_file, 'r') as f:
        paths = f.readlines()
    # Remove the placeholder entry
    embeddings = embeddings[1:]
    paths = paths[1:]
    return embeddings.numpy(), paths

def apply_tsne_gpu(embeddings, output_csv_path='tsne_results.csv'):
    tsne = cuml.TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=200, method='barnes_hut')
    tsne_results = tsne.fit_transform(embeddings)
    
    tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
    
    tsne_df.to_csv(output_csv_path, index=False)
    
    return tsne_results

# Display images
def display_images(paths, base_dir, indices=[0, 1, 2, 3, 4]):
    plt.figure(figsize=(15, 3))
    for i, index in enumerate(indices):
        plt.subplot(1, 5, i + 1)
        image_path = base_dir + paths[index].strip()
        with Image.open(image_path) as img:
            plt.imshow(img)
        plt.axis('off')
    plt.show()


def main():
    base_dir = '/scratch/qz2190/yars_data/base_out/'
    embeddings, paths = load_data(base_dir + 'features.pt', base_dir + 'paths.txt')
    tsne_results = apply_tsne_gpu(embeddings, 'tsne_results.csv')
    display_images(paths, base_dir)

if __name__ == '__main__':
    main()

    
