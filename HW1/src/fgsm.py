import os
import argparse
from PIL import Image

import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import cifar10_class
from train import config_net
from train import _save_makedirs


class EvalDataset(Dataset):
    """Dataset for loading evaluation data"""
    def __init__(self, eval_dir):
        self.eval_dir = eval_dir
        self.class_name = cifar10_class.classes
        self.images = np.zeros((100, 32, 32, 3), dtype=np.uint8)
        self.labels = torch.from_numpy(np.repeat(np.arange(10), 10)).long()

        # Read images
        for i_class in range(len(self.class_name)):
            name_class = self.class_name[i_class]
            for i_img in range(10):
                img = Image.open(
                    os.path.join(self.eval_dir,
                                 name_class,
                                 "%s%d.png" % (name_class, i_img+1))
                )
                self.images[i_class * 10 + i_img] = img

        # Image preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
            )
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def __getitem__(self, idx):
        img = self.transforms(self.images[idx])
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return 100


class FGSM_Attacker:
    def __init__(self, net, img_loader):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = net
        self.model.eval()

        self.img_loader = img_loader
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        self.criterion = nn.CrossEntropyLoss()

    def fgsm_attack(self, image, epsilon, data_grad):
        # Get sign of gradient
        # sign_data_grad = data_grad.sign()
        sign_data_grad = torch.sign(data_grad)
        # Add perturbation to image
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image

    def attack(self, epsilon, max_iter=1, decode=True):
        assert max_iter >= 1
        adv_images = np.zeros((len(self.img_loader), 32, 32, 3),
                              dtype=np.uint8)

        adv_correct = 0
        org_correct = 0
        total = 0
        for img_idx, (data, target) in tqdm.tqdm(enumerate(self.img_loader)):
            total += 1
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True

            # Get the original predicted class
            output = self.model(data)
            pred = np.argmax(output.data.cpu().numpy())
            if pred == target.data.cpu().numpy()[0]:
                org_correct += 1

            for _ in range(max_iter):
                # Compute gradients
                loss = self.criterion(
                    output,
                    Variable(torch.Tensor([float(pred)]).to(self.device).long()))

                self.model.zero_grad()
                loss.backward()
                img_grad = data.grad.data

                # Generate pertured image
                perturbed_data = self.fgsm_attack(data, epsilon, img_grad)

                # Predict the pertured image
                output = self.model(perturbed_data)
                pred = np.argmax(output.data.cpu().numpy())

            # Check prediction after attack
            if pred == target.data.cpu().numpy()[0]:
                adv_correct += 1

            adv_img = perturbed_data.data.cpu().numpy()[0]
            adv_img = adv_img.transpose(1, 2, 0)
            if decode:
                adv_img = (adv_img * self.std) + self.mean
                adv_img = adv_img * 255.0
                # adv_img = adv_img[..., ::-1]  # RGB to BGR
                adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)

            adv_images[img_idx] = adv_img

        print("Accuracy before attack: ", 100. * org_correct / total)
        print("Accuracy after attack: ", 100. * adv_correct / total)
        return adv_images


def load_weights(net, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['net'], False)
    return net


def main(args):
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained model
    net = config_net(args.net_name)
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net = load_weights(net, args.ckpt_path)

    # Construct dataloader
    eval_dataset = EvalDataset(args.data_dir)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False)

    # Call attacker
    attacker = FGSM_Attacker(net, eval_loader)
    adv_imgs = attacker.attack(epsilon=0.1, decode=True)

    # Save adv images
    for i_class in range(len(cifar10_class.classes)):
        name_class = cifar10_class.classes[i_class]
        class_dir = os.path.join(args.save_dir, name_class)
        _save_makedirs(class_dir)
        for i_img in range(10):
            Im = Image.fromarray(adv_imgs[i_class * 10 + i_img])
            Im.save(os.path.join(class_dir,
                                 "%s%d.png" % (name_class, i_img+1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../cifar-10_eval/")

    parser.add_argument(
        "--save_dir",
        type=str,
        default="../adv_imgs")

    parser.add_argument(
        "--net_name",
        type=str,
        default="VGG")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoint/VGG/model_094.pth")

    args = parser.parse_args()

    main(args)
