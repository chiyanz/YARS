import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def compute_squared_euclidean_distance(X):
    """ Computes the squared Euclidean distance matrix of a matrix X """
    sum_X = torch.sum(X ** 2, 1)
    D = -2.0 * torch.mm(X, X.t()) + sum_X + sum_X.view(-1, 1)
    return D

def compute_joint_probabilities(samples, perplexity=30.0, tol=1e-5):
    """ Compute the joint probabilities p_ij from the high-dimensional space """
    n_samples = samples.shape[0]
    distances = compute_squared_euclidean_distance(samples)
    P = torch.zeros((n_samples, n_samples), device=samples.device)
    beta = torch.ones(n_samples, device=samples.device)
    logU = torch.log(torch.tensor(perplexity, device=samples.device))

    for i in range(n_samples):
        betamin = torch.tensor(-float('inf'), device=samples.device)
        betamax = torch.tensor(float('inf'), device=samples.device)
        Di = distances[i, torch.arange(n_samples) != i]
        
        Hdiff = torch.tensor(float('inf'), device=samples.device)
        iter = 0
        while torch.abs(Hdiff) > tol and iter < 50:
            Pi = torch.exp(-Di * beta[i])
            sum_Pi = torch.sum(Pi)
            H = torch.sum(Pi * torch.log(Pi / sum_Pi))
            Hdiff = H - logU
            if Hdiff > 0:
                betamin = beta[i].clone()
                beta[i] *= 2.0
            else:
                betamax = beta[i].clone()
                beta[i] /= 2.0

            if betamax != float('inf'):
                beta[i] = (betamin + betamax) / 2.0

            iter += 1
        
        P[i, torch.arange(n_samples) != i] = Pi / sum_Pi

    return (P + P.t()) * 0.5

def custom_tsne(X, n_components=2, perplexity=30.0, n_iter=300, learning_rate=200.0, device=torch.device('cuda')):
    """ Perform t-SNE using PyTorch """
    X = X.to(device)
    P = compute_joint_probabilities(X, perplexity).detach()
    Y = torch.randn(X.shape[0], n_components, device=device, requires_grad=True)

    optimizer = optim.SGD([Y], lr=learning_rate, momentum=0.9)
    for epoch in range(n_iter):
        optimizer.zero_grad()
        D = compute_squared_euclidean_distance(Y)
        Q = 1 / (1 + D)
        Q.fill_diagonal_(0)
        Q /= torch.sum(Q)

        PQ_diff = P - Q
        grad = torch.matmul(PQ_diff, Y)
        Y.grad = grad * 4  # Gradient of t-SNE cost function
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Current loss: {torch.sum(P * torch.log(P / Q)).item()}')

    return Y.detach().cpu()

def load_data(embeddings_path, paths_file):
    embeddings = torch.load(embeddings_path)
    with open(paths_file, 'r') as f:
        paths = f.readlines()
    # Remove the placeholder entry
    embeddings = embeddings[1:]
    paths = paths[1:]
    return embeddings.numpy(), paths

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
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    tsne_results = custom_tsne(embeddings)
    display_images(paths, base_dir)

if __name__ == '__main__':
    main()