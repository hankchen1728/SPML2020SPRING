import os
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils
import models
import data_loader


__all_models__ = [
    "VGG", "ResNet", "DenseNet", "ResNeXt", "DPN", "EfficientNet"
    ]


def config_net(net_name="VGG"):
    assert net_name in __all_models__, "Unimplemented architecture"
    if net_name == "VGG":
        return models.VGG("VGG19")
    elif net_name == "ResNet":
        return models.ResNet18()
    elif net_name == "ResNeXt":
        return models.ResNeXt29_2x64d()
    elif net_name == "DenseNet":
        return models.DenseNet121()
    elif net_name == "DPN":
        return models.DPN92()
    elif net_name == "EfficientNet":
        return models.EfficientNetB0()


def _save_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set network
    net = config_net(args.net_name)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Set loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # optimizer = optim.SGD(
    #     net.parameters(), lr=args.lr,
    #     momentum=0.9, weight_decay=5e-4)

    # Data loader
    print("Constructing data loader ...")
    data_dir = args.data_dir
    train_loader, valid_loader = data_loader.get_train_valid_loader(
        data_dir=data_dir,
        batch_size=128,
        augment=True,
        random_seed=20,
        valid_size=0.1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Training processes
    epochs = args.epochs
    history = {"acc": [0.0] * epochs,
               "loss": [0.0] * epochs,
               "val_acc": [0.0] * epochs,
               "val_loss": [0.0] * epochs
               }

    ckpt_dir = os.path.join(args.ckpt_dir, args.net_name)
    _save_makedirs(ckpt_dir)
    best_val_acc = 0.0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss = 0
        correct = 0
        total = 0
        # Training
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (train_loss/(batch_idx+1),
                   100.*correct/total, correct, total)
                )

        # Validation
        val_loss = 0
        val_correct = 0
        val_total = 0
        net.eval()
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_loss /= train_loader.__len__()
        val_acc = 100. * val_correct / val_total
        val_loss /= valid_loader.__len__()
        print("[%03d/%03d] %2.2f sec(s)"
              " Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f" %
              (epoch + 1, epochs,
               time.time() - epoch_start_time,
               train_acc, train_loss, val_acc, val_loss
               ))

        history["acc"][epoch] = train_acc
        history["loss"][epoch] = train_loss
        history["val_acc"][epoch] = val_acc
        history["val_loss"][epoch] = val_loss

        # Save model checkpoints
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {
                'net': net.state_dict(),
                'acc': train_acc,
                'val_acc': val_acc,
                'epoch': epoch,
            }
            torch.save(
                state,
                os.path.join(ckpt_dir, "model_{:03}.pth".format(epoch+1))
                )

    # Save training history
    with open(os.path.join(ckpt_dir, "history.json"), 'w') as opf:
        json.dump(history, opf)

    # end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument(
            "--visible_gpus",
            type=int,
            nargs='+',
            default=[0],
            help="CUDA visible gpus")

    parser.add_argument(
            "--ckpt_dir",
            type=str,
            default="./checkpoint")

    parser.add_argument(
            "--data_dir",
            type=str,
            default="../data")

    parser.add_argument(
            "--lr",
            type=float,
            default=1e-2,
            help="learning rate")

    parser.add_argument(
            "--net_name",
            type=str,
            default="VGG")

    parser.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="training epochs")

    args = parser.parse_args()

    main(args)
