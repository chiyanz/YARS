class CaptionedImageDataset(Dataset):
    def __init__(self, img_text_pairs, processor):
        self.img_text_pairs = img_text_pairs
        self.processor = processor
        self.root_path = './photos/'

    def __len__(self):
        return len(self.img_text_pairs)

    def __getitem__(self, idx):
        img_id, text = self.img_text_pairs[idx]
        image = Image.open(os.path.join(self.root_path, img_id)).convert("RGB")

        # Process image and text
        # text_input = self.processor(text=text, padding=True)
        # image_input = self.processor(images=image)
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        return inputs