import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
import pretrainedmodels
from glob import glob
import os

# --- FeatureExtractor ---
class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


# --- ModelOutputs ---
class ModelOutputs():
    def __init__(self, model, feature_module, target_layers):
        self.model = model.module
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "bn4" in name.lower():
                x = module(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
            else:
                x = module(x)
        return target_activations, x


# --- Preprocessing function ---
def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    img = img[:, :, ::-1].copy()
    img = (img - means) / stds
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    return img.requires_grad_(True)


# --- Visualization function ---
def show_cam_on_image(img, mask, save_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_name + "_cam.jpg", np.uint8(255 * cam))


# --- GradCam ---
class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1].cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


# --- GuidedBackpropReLU  ---
class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = input * positive_mask
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        return grad_output * positive_mask_1 * positive_mask_2


# --- GuidedBackpropReLUModel ---
class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.replace_relu()

    def replace_relu(self):
        def recursive_apply(module):
            for idx, sub_module in module._modules.items():
                if isinstance(sub_module, nn.ReLU):
                    module._modules[idx] = GuidedBackpropReLU.apply
                else:
                    recursive_apply(sub_module)
        recursive_apply(self.model)

    def forward(self, input):
        return self.model.module(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)
        output = input.grad.cpu().data.numpy()[0]
        return output


def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


# --- Main program ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-folder', type=str, default='../data/onfh_3cls/training_data/', help='Folder with images')
    parser.add_argument('--save-folder', type=str, default='../cam/', help='Folder to save CAM results')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    model_checkpoint = torch.load("../models/epoch_195_.pth.tar")
    model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
    num_fc = model.last_linear.in_features
    model.last_linear = nn.Linear(num_fc, 3)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(model_checkpoint['state_dict'])

    img_list = glob(os.path.join(args.image_folder, '*.png'))
    grad_cam = GradCam(model=model, feature_module=model.module.conv4, target_layer_names=["pointwise"], use_cuda=args.use_cuda)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)

    for image_path in img_list:
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        img = cv2.imread(image_path, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)

        pred = model.module(input.cuda())
        pred_label = torch.argmax(pred).item()
        print(f"{img_name} prediction: {pred_label}")

        mask = grad_cam(input)
        show_cam_on_image(img, mask, os.path.join(args.save_folder, img_name + f"_pred{pred_label}"))

        gb = gb_model(input)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        cv2.imwrite(os.path.join(args.save_folder, img_name + f"_pred{pred_label}_gb.jpg"), gb)
        cv2.imwrite(os.path.join(args.save_folder, img_name + f"_pred{pred_label}_cam_gb.jpg"), cam_gb)