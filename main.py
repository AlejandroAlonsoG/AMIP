
from torch.utils.data import DataLoader
import torch

from dataset_utils import VancouverDataset, generate_splits
from unet import UNet
from trainer import BaselineTrainer
import general_loader as gl

def main(cfg):

    ## Load datasets ##

    image_transform, segmentation_transform = gl.get_transforms()

    train_dataset = VancouverDataset(
        root_dir=cfg.directories.dataset_root,
        split='train',
        groups=None,
        transform=image_transform,
        target_transform=segmentation_transform
    )

    test_dataset = VancouverDataset(
        root_dir=cfg.directories.dataset_root,
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
        use_cuda=torch.cuda.is_available()
    )

    trainer.fit(train_data_loader=train_loader, epoch=cfg.training.num_epochs)

    ## Evaluate ##

    trainer.evaluate(test_loader, num_classes=len(classes))

    ## Save model ##

    torch.save(model.state_dict(), cfg.directories.save_model_path)

if __name__ == '__main__':

    cfg = gl.load_config("./config.yaml")

    main(cfg)