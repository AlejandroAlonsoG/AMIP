import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset_utils import VancouverDataset
from unet import UNet


from torchvision import transforms
from PIL import Image

from trainer import BaselineTrainer
import general_loader as gl
import transforms_loader as tl

# Visualization
def visualize_predictions(data_loader, model, num_samples=3):
    model.eval()
    current_samples = 0
    with torch.no_grad():
        for i, (images, masks) in enumerate(data_loader):
            if current_samples >= num_samples:
                break

            if torch.cuda.is_available():
                images = images.cuda()

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            for j in range(images.size(0)):
                if current_samples >= num_samples:
                    break
                
                img = images[j].permute(1, 2, 0).cpu().numpy()
                gt_mask = masks[j].cpu().numpy()
                pred_mask = predictions[j]

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title("Original Image")
                plt.imshow((img * 255).astype(np.uint8))
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.title("Ground Truth Mask")
                plt.imshow(gt_mask, cmap="tab20")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.title("Predicted Mask")
                plt.imshow(pred_mask, cmap="tab20")
                plt.axis("off")

                plt.show()

                current_samples+=1

def main(cfg):

    ## Load the model ##

    classes = gl.get_classes()
    class_mapping = [cls[1] for cls in classes]
    num_classes = len(classes)

    model = UNet(num_classes=num_classes, class_mapping=class_mapping)
    model.load_state_dict(torch.load("./best_model.pth"))
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    ## Load the dataset ##

    image_transform, segmentation_transform = tl.get_transforms(general_transforms=cfg.dataset.use_general_transforms, input_size=cfg.dataset.original_size)

    test_dataset = VancouverDataset(
        root_dir=cfg.dataset.root,
        split='test',
        groups=None,
        transform=image_transform,
        target_transform=segmentation_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)

    criterion = gl.load_loss(cfg.training.loss)
    optimizer = gl.load_optimizer(cfg.training.optimizer, model.parameters(), lr=cfg.training.learning_rate)

    trainer = BaselineTrainer(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        num_classes=len(classes),
        use_cuda=torch.cuda.is_available()
    )

    ## Evaluate ##

    trainer.evaluate(test_loader)

if __name__ == '__main__':

    cfg = gl.load_config("./configs/conf_loss_1.yaml")

    main(cfg)