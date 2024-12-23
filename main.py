
from torch.utils.data import DataLoader
import torch
import sys
from datetime import datetime
import pprint

from dataset_utils import VancouverDataset
from unet import UNet
from trainer import BaselineTrainer
import general_loader as gl
import transforms_loader as tl

# This is just for being able to log into a file at the same time that the output gets printed on the terminal
class TeeStream:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main(cfg):

    ## Set up logging ##

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = TeeStream(f"./logs/{timestamp}.log")
    sys.stderr = sys.stdout

    pprint.pprint(cfg)

    ## Load datasets ##

    image_transform, segmentation_transform = tl.get_transforms(general_transforms=cfg.dataset.use_general_transforms, input_size=cfg.dataset.original_size)

    train_dataset = VancouverDataset(
        root_dir=cfg.dataset.root,
        split='train',
        groups=None,
        transform=image_transform,
        target_transform=segmentation_transform,
        sunny_augmentation_prob=cfg.dataset.sunny_augmentation_prob,
        rainy_augmentation_prob=cfg.dataset.rainy_augmentation_prob
    )

    test_dataset = VancouverDataset(
        root_dir=cfg.dataset.root,
        split='test',
        groups=None,
        transform=image_transform,
        target_transform=segmentation_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")

    ## Define what we use for the training ##
    classes = gl.get_classes()
    class_mapping = [cls[1] for cls in classes]

    model = UNet(num_classes=len(classes), class_mapping=class_mapping, input_channels=3)

    criterion = gl.load_loss(cfg.training.loss) # TODO try focal loss
    optimizer = gl.load_optimizer(cfg.training.optimizer, model.parameters(), lr=cfg.training.learning_rate)

    ## Train ##
    trainer = BaselineTrainer(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        num_classes=len(classes),
        use_cuda=torch.cuda.is_available()
    )

    trainer.fit(train_data_loader=train_loader, epoch=cfg.training.num_epochs)

    ## Evaluate ##

    trainer.evaluate(test_loader)

    ## Save model ##

    torch.save(model.state_dict(), cfg.directories.save_model_path)

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python3 main.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        cfg = gl.load_config(config_path)
        main(cfg)
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)