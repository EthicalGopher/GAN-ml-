import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from models.sr_model import SRCNN

class SRDataset(Dataset):
    """
    Dataset loader for Super-Resolution training.
    Extracts the B10 thermal band (left half of the combined Pix2Pix tile),
    downsamples it to simulate 200m resolution, and upsamples it via Bicubic
    as the input. The original B10 tile acts as the ground truth.
    """
    def __init__(self, folder_path, tile_size=256, downscale_factor=2):
        self.file_paths = glob.glob(os.path.join(folder_path, "*.png"))
        self.tile_size = tile_size
        self.downscale_factor = downscale_factor
        
        # Transformations
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        combined_img = Image.open(img_path).convert("RGB")
        
        # Extract B10 thermal band (left half of combined image)
        w, h = combined_img.size
        # The B10 patch is w/2 x h (usually 256x256)
        b10_gt = combined_img.crop((0, 0, w // 2, h))
        
        # Convert B10 target to grayscale (single-channel)
        b10_gt_gray = b10_gt.convert("L")
        
        # Simulate low-res B10 @200m by downsampling
        lr_size = (self.tile_size // self.downscale_factor, self.tile_size // self.downscale_factor)
        b10_lr = b10_gt_gray.resize(lr_size, Image.Resampling.BICUBIC)
        
        # Bicubic upscale back to target size for SRCNN input
        b10_input = b10_lr.resize((self.tile_size, self.tile_size), Image.Resampling.BICUBIC)
        
        # Convert to Tensors
        input_tensor = self.to_tensor(b10_input) # [1, H, W]
        target_tensor = self.to_tensor(b10_gt_gray) # [1, H, W]
        
        return input_tensor, target_tensor

def main():
    parser = argparse.ArgumentParser(description="Train SRCNN Super-Resolution Model on B10 thermal band.")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/guwahati_azara_processed", help="Path to preprocessed dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/sr_model", help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="gpu/cpu or auto")
    
    args = parser.parse_args()
    
    # Establish device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Training on device: {device}")
    
    # Paths
    train_path = os.path.join(args.dataset_dir, "train")
    val_path = os.path.join(args.dataset_dir, "val")
    
    if not os.path.exists(train_path):
        print(f"Error: Training folder {train_path} does not exist. Run preprocessing first.")
        return
        
    # Dataset and Loader
    train_dataset = SRDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    has_val = os.path.exists(val_path) and len(glob.glob(os.path.join(val_path, "*.png"))) > 0
    if has_val:
        val_dataset = SRDataset(val_path)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print(f"Loaded {len(train_dataset)} training tiles and {len(val_dataset)} validation tiles.")
    else:
        print(f"Loaded {len(train_dataset)} training tiles (no validation dataset found).")
        
    # Initialize SRCNN Model (1-channel input/output for greyscale B10)
    model = SRCNN(num_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\nStarting SRCNN Training...")
    print("-----------------------------------")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            
        train_loss = epoch_loss / len(train_loader.dataset)
        
        # Validation
        val_loss_str = ""
        if has_val:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            val_loss = val_loss / len(val_loader.dataset)
            # Calculate PSNR
            psnr = 10 * np.log10(1.0 / (val_loss + 1e-10))
            val_loss_str = f" | Val Loss: {val_loss:.6f} | PSNR: {psnr:.2f} dB"
            
        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.6f}{val_loss_str}")
        
        # Save checkpoints periodically and at end
        if epoch % 10 == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_dir, f"srcnn_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Saved checkpoint: {checkpoint_path}")
            
    print("-----------------------------------")
    print("SRCNN Model Training Finished successfully!")

if __name__ == "__main__":
    main()
