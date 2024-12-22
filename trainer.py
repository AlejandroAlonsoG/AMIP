
from typing import Callable, List
import torch
import torch.utils.data as data
from sklearn.metrics import confusion_matrix
import numpy as np

class BaselineTrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 use_cuda=True):
        self.loss = loss
        self.use_cuda = use_cuda
        self.optimizer = optimizer

        if use_cuda:
            self.model = model.to(device="cuda:0")
        else:
            self.model = model

    def fit(self, train_data_loader: data.DataLoader, epoch: int):
        avg_loss = 0.
        self.model.train()
        for e in range(epoch):
            print(f"\nStart epoch {e + 1}/{epoch}")
            epoch_loss = 0
            for i, (images, masks) in enumerate(train_data_loader):
                self.optimizer.zero_grad()

                if self.use_cuda:
                    images = images.cuda()
                    masks = masks.cuda()

                # Forward pass
                outputs = self.model(images)
                # compressed_outputs = self.model.compress_unet_output(outputs)
                # compressed_outputs = compressed_outputs.float()
                # masks = masks.squeeze(1)
                loss = self.loss(outputs, masks)

                # Backward pass
                loss.backward()

                # Optimize
                self.optimizer.step()

                epoch_loss += loss.item()

                print(f"\rBatch {i + 1}/{len(train_data_loader)}: loss = {loss.item():.4f}", end='')
            
            avg_loss = epoch_loss / len(train_data_loader)
            print(f"\nEpoch {e + 1} finished. Average loss: {avg_loss:.4f}")
        
        return avg_loss

    def evaluate(self, test_data_loader: data.DataLoader, num_classes):
        self.model.eval()
        total_loss = 0

        confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

        with torch.no_grad():
            for images, masks in test_data_loader:
                if self.use_cuda:
                    images = images.cuda()
                    masks = masks.cuda()

                # Forward pass
                outputs = self.model(images)
                loss = self.loss(outputs, masks)
                total_loss += loss.item()

                # Get predictions
                preds = torch.argmax(outputs, dim=1)

                # Update confusion matrix
                for pred, mask in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                    confusion_mat += confusion_matrix(
                        mask.flatten(), pred.flatten(), labels=list(range(num_classes))
                    )

        # Get intersection over union
        intersection = np.diag(confusion_mat) # TP
        union = (confusion_mat.sum(axis=0) + confusion_mat.sum(axis=1) - intersection)  # Union = TP + FP + FN
        iou_per_class = intersection / (union + 1e-6)

        # Get mean IoU
        mean_iou = np.mean(iou_per_class)

        # Get class-weighted IoU
        class_pixel_frequencies = confusion_mat.sum(axis=1) / confusion_mat.sum()
        weighted_iou = np.sum(class_pixel_frequencies * iou_per_class)

        # Get average loss
        avg_loss = total_loss / len(test_data_loader)

        print(f"\nEvaluation completed. Average loss: {avg_loss:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Class-Weighted IoU: {weighted_iou:.4f}")

        return avg_loss, mean_iou, weighted_iou
