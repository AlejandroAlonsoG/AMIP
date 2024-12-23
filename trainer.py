import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Callable, List
from datetime import datetime

class BaselineTrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 num_classes: int,
                 use_cuda=True):
        self.loss = loss
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.num_classes = num_classes

        if use_cuda:
            self.model = model.to(device="cuda:0")
        else:
            self.model = model

        self.metrics = {"train_loss": [], "train_iou": []}

    def fit(self, train_data_loader, epoch: int):
        for e in range(epoch):
            print(f"\nStart epoch {e + 1}/{epoch}")
            self.model.train()

            epoch_loss = 0.0
            confusion_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

            for i, (images, masks) in enumerate(train_data_loader):
                self.optimizer.zero_grad()

                if self.use_cuda:
                    images = images.cuda()
                    masks = masks.cuda()

                # Forward pass
                outputs = self.model(images)
                loss = self.loss(outputs, masks)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Calculate predictions and update confusion matrix
                preds = torch.argmax(outputs, dim=1)
                for pred, mask in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                    confusion_mat += confusion_matrix(
                        mask.flatten(), pred.flatten(), labels=list(range(self.num_classes))
                    )

            # Calculate IoU for the epoch
            intersection = np.diag(confusion_mat)
            union = (confusion_mat.sum(axis=0) + confusion_mat.sum(axis=1) - intersection)
            iou_per_class = intersection / (union + 1e-6)
            mean_iou = np.mean(iou_per_class)

            avg_loss = epoch_loss / len(train_data_loader)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"\n{timestamp} - Epoch {e + 1} finished. Avg Loss: {avg_loss:.4f}, Mean IoU: {mean_iou:.4f}")

            # Log metrics
            self.metrics["train_loss"].append(avg_loss)
            self.metrics["train_iou"].append(mean_iou)

        return self.metrics

    def evaluate(self, test_data_loader):
        self.model.eval()
        total_loss = 0.0
        confusion_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        with torch.no_grad():
            for images, masks in test_data_loader:
                if self.use_cuda:
                    images = images.cuda()
                    masks = masks.cuda()

                outputs = self.model(images)
                loss = self.loss(outputs, masks)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                for pred, mask in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                    confusion_mat += confusion_matrix(
                        mask.flatten(), pred.flatten(), labels=list(range(self.num_classes))
                    )

        intersection = np.diag(confusion_mat)
        union = (confusion_mat.sum(axis=0) + confusion_mat.sum(axis=1) - intersection)
        iou_per_class = intersection / (union + 1e-6)
        mean_iou = np.mean(iou_per_class)
        avg_loss = total_loss / len(test_data_loader)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"{timestamp} - Evaluation complete. Avg Loss: {avg_loss:.4f}, Mean IoU: {mean_iou:.4f}")
        return avg_loss, mean_iou
