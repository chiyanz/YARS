# Finetuning code from: https://medium.com/aimonks/a-guide-to-fine-tuning-clip-models-with-custom-data-6c7c0d1416fb
# Multiprocessing code from: https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
# 1 GPU ~2s/it

"""
# Notes:
BATCHSIZE = 2048 - EFFECTIVE BATCH SIZE: 8192
TRAINING EPOCHS = 20
First, I chnaged the code to work with our data.
Then, I implemented multiprocessing since it was going very slowly.
Then, I inspected loss graphs to change hyperparameters for better convergence.
 - I ran a couple configurations for ~400 iterations each (100 iterations on main process) and compared the loss graphs because training takes a long time
Investigation: impact of drop_last - training loss dips greatly at the beginning of epoch (0 is used for padding) - this actually fixed it
Investigation: transfer learning - freeze earlier layers and only udpate FC layers
- allows much faster batch size
Different loss logs:
- original + v1-v5: various hyperparmeter tweaks with finetuning (not much difference)
- droplast: changing drop_last=True on the sampler to fix training loss inconsistencies
- transfer: changing to transfer learning by freezing all but the last FC layer (there are two - one for text and image)
- tried randomized and keeping the last layer, but keeping was much better

At this point, the Coca embeddings were ready, so I split the data into train/test so that I could compare
Observations: contrastive loss depends on batch_size
"""

# import modules
import clip
import json
import os
import pandas as pd
import tempfile
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from PIL import Image
from sys import argv
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# config
root_dir = "/scratch/jl12734/yars/"
image_dir = root_dir + "photos/"
batch_size = 2048
test_batch_size = 2048
out_path = "out/finetune_model.pt"     # model parameters, loss per epoch
out_path_loss = "out/finetune_loss_transfer_coca.pt" # loss per batch
test_set_file = "test_images.csv"

if len(argv) > 1:
    batch_size = int(argv[1])
warnings.filterwarnings("ignore", category=FutureWarning) 
test_set = {path: caption for path, caption in pd.read_csv(test_set_file, lineterminator='\n').values}

# setup multiprocessing
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def gather_pairs(label: str, no_captions: bool = False):
    """
    takes the label (category of images to fetch) and filters photos.json for images with that label
    filters for images that 
    returns: list of photo objects (photo_id, label, caption, etc.)
    """ 
    with open(os.path.join(root_dir, 'photos.json'), 'r') as file:
        all_photos = []
        all_captions = []
        cc = pd.read_csv("captions.txt", delimiter=":", names=["id", "caption"])
        cc.id = cc.id.str.split("/").str[-1].str.split(".").str[0]
        cc = {id: caption for id, caption in cc.values}
        for line in file:
            line = line.rstrip()
            try:
                photo_record = json.loads(line)
                photo_id = photo_record["photo_id"]
                photo_path = photo_id + ".jpg"
                caption = photo_record["caption"]
                # filers for only ones wiht captions by default
                if (photo_record['label'] == label and photo_path not in test_set and photo_id != "LXT4hCf1lRyUeM4HDBaSvg"):
                    if len(caption) == 0:
                        caption = cc.get(photo_id.strip(), "")
                    if len(caption) > 0:
                        all_captions.append(caption)
                        all_photos.append(photo_path)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} at line: {line}")
        return all_photos, all_captions
    
class TrainDataset():
    def __init__(self):
        list_image_path, list_txt = gather_pairs("food")
        self.image_path = list_image_path
        self.title = list_txt

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(image_dir, self.image_path[idx]))
        title = self.title[idx]
        return image, title

class TestDataset():
    def __init__(self):
        self.image_path = list(test_set.keys())
        self.title = list(test_set.values())
        
    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(image_dir, self.image_path[idx]))
        title = self.title[idx]
        return image, title

def prepare(rank, world_size, batch_size=batch_size, pin_memory=False, num_workers=0):
    dataset = TrainDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, 
                            drop_last=True, collate_fn=lambda x: zip(*x), sampler=sampler)
    
    return dataloader

num_epochs = 20
history = []
losses = {
    "by_it": [],
    "by_epoch": []
}
def main(rank, world_size):
    # setup the process groups
    setup(rank, world_size)
    
    # prepare dataloaders
    dataloader = prepare(rank, world_size)
    losses["its_per_epoch"] = len(dataloader) * 4
    if rank == 0:
        test_loader = DataLoader(TestDataset(), batch_size=test_batch_size, pin_memory=False, num_workers=0, collate_fn=lambda x: zip(*x), shuffle=True)
    
    # load model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(rank)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # freeze all but final layer
    frozen_layers = ["visual_projection.weight", "text_projection.weight"]
    for name, param in model.named_parameters():
        if name not in frozen_layers:
            param.requires_grad = False # freeze layers

    # reset FC weights
    # for child in model.children():
    #     if isinstance(child, nn.Linear):
    #         child.reset_parameters()
            
    # wrap the model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    # train loop
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #v1
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,eps=1e-6,weight_decay=0.2) # v2
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),weight_decay=0.2) # v3
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.8,0.9),weight_decay=0.2) # v4
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),weight_decay=0.5) # v5
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # higher lr for transfer
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,betas=(0.8,0.9),weight_decay=0.2) #
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)   

        pbar = tqdm(dataloader, total=len(dataloader))
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            images, titles = batch
            inputs = processor(
                text=titles, 
                images=images, 
                return_tensors="pt",
                padding=True, 
                truncation=True
            ).to(rank)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Compute loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=rank)
            total_loss = (loss_img(outputs.logits_per_image,ground_truth) + loss_txt(outputs.logits_per_text,ground_truth))/2

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Save loss
            if rank == 0:
                losses["by_it"].append(total_loss.item())
                torch.save(losses, out_path_loss)

            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}," +  
                    f" Reserved Memory: {(torch.cuda.memory_reserved(rank)/1024/1024/1024):.4f}GB, Rank: {rank}")
        
        if rank == 0:
            # calculate validation loss
            print("Calculating validation loss...", end="")
            val_loss = 0
            n_samples = 0
            _, batch = next(enumerate(test_loader))
            images, titles = batch
            inputs = processor(
                text=titles, 
                images=images, 
                return_tensors="pt",
                padding=True, 
                truncation=True
            ).to(rank)
            local_model = model.module.to(0)
            outputs = local_model(**inputs)
            ground_truth = torch.arange(len(images),dtype=torch.long,device=rank)
            val_loss = (loss_img(outputs.logits_per_image,ground_truth) + loss_txt(outputs.logits_per_text,ground_truth))/2
            val_loss = val_loss.item()
            
            print("\rEpoch %d validation loss: %f" % (epoch, val_loss))
            
            loss_entry = {
                "val": val_loss,
                "train": total_loss.item()
            }
            losses["by_epoch"].append(loss_entry)
            torch.save(losses, out_path_loss)

        if rank == 1:
            # checkpoint model
            tempdir = root_dir + "checkpoints/%d" % time.time()
            os.mkdir(tempdir)
            model_path = tempdir + "/model.checkpoint"
            torch.save(model.module.state_dict(), model_path)
            history.append({
                'epoch': epoch,
                'model_state_path': model_path,
                'train_loss': total_loss.item(),
            })
            torch.save(history, out_path)
            print("Saved model checkpoint")
            
        dist.barrier()    
        
    cleanup()

if __name__ == '__main__':
    world_size = 4
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )
        
        
    