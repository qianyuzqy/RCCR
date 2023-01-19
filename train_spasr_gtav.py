import argparse
import os
import sys
import random
import timeit
import datetime

import numpy as np
import pickle
import scipy.misc
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d
from utils.timer import Timer
from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette
# from utils.random_crop import RandomCrop, RandomCrop_img

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateUDA import evaluate
import os
import os.path as osp
import time
import gc

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    return parser.parse_args()



def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_ema_model(model):
    #ema_model = getattr(models, config['arch']['type'])(self.train_loader.dataset.num_classes, **config['arch']['args']).to(self.device)
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    if len(gpus)>1:
        #return torch.nn.DataParallel(ema_model, device_ids=gpus)
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def strongTransform_inverse(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix_inverse(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def strongTransform_cutmix(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    return data, target

def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters

def get_spatial_matrix(path="./model/prior_array.mat"):
    if not os.path.exists(path):
        raise FileExistsError("please put the spatial prior in ..model/")
    sprior = sio.loadmat(path)
    sprior = sprior["prior_array"]
    # foreground_map = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
    # background_map = [0, 1, 2, 3, 4, 8, 9, 10]
    # sprior = sprior[foreground_map]
    # sprior = sprior[background_map]
    tensor_sprior = torch.tensor(sprior, dtype=torch.float64, device='cuda').float().cuda()
    return tensor_sprior

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('./visualiseImages/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('./visualiseImages', str(epoch)+ id + '.png'))

def _save_checkpoint(iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model

def main():
    print(config)

    best_mIoU = 0
    best_mIoU_iter = 0 
    _t = {'iter time' : Timer()}

    if consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss =  torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()
        else:
            unlabeled_loss =  MSELoss2d().cuda()
    elif consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label), device_ids=gpus).cuda()
        else:
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda()

    cudnn.enabled = True

    # create network
    model = Res_Deeplab(num_classes=num_classes)

    # load pretrained parameters
    #saved_state_dict = torch.load(args.restore_from)
        # load pretrained parameters
    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)
    #code from AdaptSegNet
    new_params = model.state_dict().copy()
    args.num_classes = 19
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if not args.num_classes == 19 or not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            # print i_parts
    model.load_state_dict(new_params)
    """
    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)
    """
    # init ema-model
    if train_unlabeled:
        ema_model = create_ema_model(model)
        ema_model.train()
        ema_model = ema_model.cuda()
        #ema_model  = ema_model.to("cuda:0")
    else:
        ema_model = None

    if len(gpus)>1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)

    model.train()
    model.cuda()

    cudnn.benchmark = True

    #--------------------------------------------------------------------------------------
    # Data loader for target domain
    if dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        if random_crop:
            data_aug = Compose([RandomCrop_city(input_size)])
        else:
            data_aug = None

        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size, img_mean = IMG_MEAN)
        # train_dataset = data_loader(data_path, is_transform=True, augmentations=None, img_size=input_size, img_mean = IMG_MEAN)
    
    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    if labeled_samples is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    else:
        partial_size = labeled_samples
        print('Training on number of samples:', partial_size)
        np.random.seed(random_seed)
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        trainloader_remain_iter = iter(trainloader_remain)

    #--------------------------------------------------------------------------------------
    # Data loader for source domain
    # New loader for Domain transfer
    if True:
        data_loader = get_loader('gta')
        data_path = get_data_path('gta')
        if random_crop:
            data_aug = Compose([RandomCrop_gta(input_size)])
        else:
            data_aug = None

        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN)

    trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    trainloader_iter = iter(trainloader)
    print('gta size:',len(trainloader))
    #--------------------------------------------------------------------------------------
    #Load new data for domain_transfer

    # optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer = optim.SGD(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        if len(gpus) > 1:
            optimizer = optim.Adam(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)

    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    start_iteration = 0

    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)

    accumulated_loss_l = []
    accumulated_loss_consistency = []
    accumulated_loss_contrastive = []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)


    #--------------------------------------------------------------------------------------   
    # start training
    _t['iter time'].tic()
    epochs_since_start = 0

    #params for contrastive loss:
    feature_bank = []
    label_bank = []
    step_count = 0
    step_save = 2
    temp = 0.1
    pos_thresh_value = 0.75
    weight_contrastive = 0.1
    iter_start_contrastive = 300000

    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_l_value = 0
        loss_consistency_value = 0
        loss_contrastive_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

        # training loss for labeled data only
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(trainloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        #if random_flip:
        #    weak_parameters={"flip":random.randint(0,1)}
        #else:
        weak_parameters={"flip": 0}


        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.cuda().long()

        #images, labels = weakTransform(weak_parameters, data = images, target = labels)

        output = model(images)
        pred = interp(output['seg'])
        L_l = loss_calc(pred, labels) # Cross entropy loss for labeled data
        #L_l = torch.Tensor([0.0]).cuda()

        if train_unlabeled:
            #get target images
            try:
                batch_remain = next(trainloader_remain_iter)
                if batch_remain[0].shape[0] != batch_size:
                    batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)
            images_remain, _, _, _, _ = batch_remain
            images_remain = images_remain.cuda()

            # classmix for consistency loss: select half classes region from the source image and paste it to the target image.
            output_target = ema_model(images_remain)
            output_target_seg = output_target['seg']  # [batch_size,19,65,65]
            output_target_seg_upsample = interp(output_target_seg)  # [batch_size,19,512,512]
            target_prob_upsample = torch.softmax(output_target_seg_upsample.detach(), dim=1)# [batch_size,512,512]
            target_max_probs, targets_label_upsample = torch.max(target_prob_upsample, dim=1)# [batch_size,512,512]

            for image_i in range(batch_size):
                classes = torch.unique(labels[image_i])
                # classes=classes[classes!=ignore_label]
                nclasses = classes.shape[0]
                # if nclasses > 0:
                classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)).long()]).cuda()
                if image_i == 0:
                    MixMask0 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
                else:
                    MixMask1 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()

            augmentation1_parameters = {"Mix": MixMask0}
            if random_flip:
                augmentation1_parameters["flip"] = random.randint(0, 1)
            else:
                augmentation1_parameters["flip"] = 0
            if color_jitter:
                augmentation1_parameters["ColorJitter"] = random.uniform(0, 1)
            else:
                augmentation1_parameters["ColorJitter"] = 0
            if gaussian_blur:
                augmentation1_parameters["GaussianBlur"] = random.uniform(0, 1)
            else:
                augmentation1_parameters["GaussianBlur"] = 0

            mix1_images_0, _ = strongTransform(augmentation1_parameters, data = torch.cat((images[0].unsqueeze(0),images_remain[0].unsqueeze(0))))
            augmentation1_parameters["Mix"] = MixMask1
            mix1_images_1, _ = strongTransform(augmentation1_parameters, data = torch.cat((images[1].unsqueeze(0),images_remain[1].unsqueeze(0))))
            mix1_images = torch.cat((mix1_images_0, mix1_images_1))
            output_mix1 = model(mix1_images)
            output_mix1_seg = output_mix1['seg']  # [batch_size,19,65,65]
            output_mix1_seg_upsample = interp(output_mix1_seg)  # [batch_size,19,512,512]

            augmentation1_parameters["Mix"] = MixMask0
            _, mix1_label_0 = strongTransform(augmentation1_parameters, target = torch.cat((labels[0].unsqueeze(0),targets_label_upsample[0].unsqueeze(0))))
            augmentation1_parameters["Mix"] = MixMask1
            _, mix1_label_1 = strongTransform(augmentation1_parameters, target = torch.cat((labels[1].unsqueeze(0),targets_label_upsample[1].unsqueeze(0))))
            mix1_label = torch.cat((mix1_label_0, mix1_label_1)).long()

            unlabeled_weight = torch.sum(target_max_probs.ge(0.968).long() == 1).item() / np.size(np.array(mix1_label.cpu()))
            pixelWiseWeight = unlabeled_weight * torch.ones(target_max_probs.shape).cuda()

            onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
            augmentation1_parameters["Mix"] = MixMask0
            _, pixelWiseWeight0 = strongTransform(augmentation1_parameters, target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0))))
            augmentation1_parameters["Mix"] = MixMask1
            _, pixelWiseWeight1 = strongTransform(augmentation1_parameters, target = torch.cat((onesWeights[1].unsqueeze(0),pixelWiseWeight[1].unsqueeze(0))))
            pixelWiseWeight = torch.cat((pixelWiseWeight0,pixelWiseWeight1)).cuda()

            L_consistency = consistency_weight * unlabeled_loss(output_mix1_seg_upsample, mix1_label, pixelWiseWeight)

            if i_iter >= iter_start_contrastive:
                # cutmix for contrastive loss: cut a square region from the target image and paste it to the source image.
                w = images_remain.shape[2]
                h = images_remain.shape[3]
                cut_ratio = np.random.beta(1, 1)
                cut_side_ratio = np.sqrt(cut_ratio)
                cut_w = np.int(w * cut_side_ratio)
                cut_h = np.int(h * cut_side_ratio)
                cut_h = (cut_h // 8) * 8
                cut_w = (cut_w // 8) * 8
                if cut_w < 8: cut_w += 8
                if cut_h < 8: cut_h += 8
                cx = np.random.randint(w // 8 - cut_w // 8)
                cy = np.random.randint(h // 8 - cut_h // 8)
                bl = cx * 8
                bu = cy * 8
                br = cx * 8 + cut_w
                bb = cy * 8 + cut_h
                mix2_images = images.clone()
                mix2_images[:, :, bu:bb, bl:br] = images_remain[:, :, bu:bb, bl:br]
                # other data augmentation for mix images
                augmentation2_parameters = {}
                if color_jitter:
                    augmentation2_parameters["ColorJitter"] = random.uniform(0, 1)
                else:
                    augmentation2_parameters["ColorJitter"] = 0
                if gaussian_blur:
                    augmentation2_parameters["GaussianBlur"] = random.uniform(0, 1)
                else:
                    augmentation2_parameters["GaussianBlur"] = 0
                mix2_images, _ = strongTransform_cutmix(augmentation2_parameters, data=mix2_images)
                if random_flip:
                    flip2 = random.randint(0, 1)
                else:
                    flip2 = 0
                if flip2 == 1:
                    mix2_images = torch.flip(mix2_images, (3,))

                # send the target images and the mix images to the net
                output_mix2 = model(mix2_images)
                output_mix2_embedding = output_mix2['embedding']     # [batchsize,128,65,65]
                output_target_embedding = output_target['embedding'] # [batchsize,128,65,65]
                if flip2 == 1:
                    output_mix2_embedding = torch.flip(output_mix2_embedding, (3,))

                target_logits = F.softmax(output_target_seg, 1).max(1)[0].detach()  # [batch_size, 65, 65]
                target_label = output_target_seg.max(1)[1].detach()                 # [batch_size,65, 65]

                # compute the label of the mix2 images, shape: [2,65,65]
                # downsampling
                labels = labels.unsqueeze(1).float().clone()
                labels = torch.nn.functional.interpolate(labels, (output_mix2_embedding.shape[2], output_mix2_embedding.shape[3]), mode='nearest')
                labels = labels.squeeze(1).long()
                #mix
                mix2_label = labels.clone()
                mix2_label[:, bu//8:bb//8, bl//8:br//8] = target_label[:, bu//8:bb//8, bl//8:br//8]

                # get overlap part
                mix2_feature_overlap_list = []
                target_feature_overlap_list = []
                target_label_overlap_list = []
                target_logits_overlap_list = []
                for idx in range(output_mix2_embedding.size(0)):
                    output_mix2_idx = output_mix2_embedding[idx]
                    output_target_idx = output_target_embedding[idx]
                    target_label_idx = target_label[idx]
                    target_logits_idx = target_logits[idx]
                    mix2_feature_overlap_list.append(output_mix2_idx[:, bu // 8:bb // 8, bl // 8:br // 8].permute(1, 2, 0).contiguous().view(-1, output_mix2_embedding.size(1)))
                    target_feature_overlap_list.append(output_target_idx[:, bu // 8:bb // 8, bl // 8:br // 8].permute(1, 2, 0).contiguous().view(-1,output_mix2_embedding.size(1)))
                    target_label_overlap_list.append(target_label_idx[bu // 8:bb // 8, bl // 8:br // 8].contiguous().view(-1))
                    target_logits_overlap_list.append(target_logits_idx[bu // 8:bb // 8, bl // 8:br // 8].contiguous().view(-1))
                mix2_feature_overlap = torch.cat(mix2_feature_overlap_list, 0)  # [n, c]
                target_feature_overlap = torch.cat(target_feature_overlap_list, 0)  # [n, c]
                target_label_overlap = torch.cat(target_label_overlap_list, 0)  # [n,]
                target_logits_overlap = torch.cat(target_logits_overlap_list, 0)  # [n,]
                assert mix2_feature_overlap.size(0) == target_feature_overlap.size(0)
                assert target_label_overlap.size(0) == target_logits_overlap.size(0)

                #get negtive samples
                b, c, h, w = output_mix2_embedding.size()
                selected_num = 4000
                mix2_feature_flatten = output_mix2_embedding.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
                target_feature_flatten = output_target_embedding.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
                selected_idx1 = np.random.choice(range(b * h * w), selected_num, replace=False)
                selected_idx2 = np.random.choice(range(b * h * w), selected_num, replace=False)
                mix2_feature_flatten_selected = mix2_feature_flatten[selected_idx1]
                target_feature_flatten_selected = target_feature_flatten[selected_idx2]
                all_flatten_selected = torch.cat([mix2_feature_flatten_selected, target_feature_flatten_selected], 0)
                all_feature = all_flatten_selected  # single gpu

                mix_label_flatten_selected = mix2_label.view(-1)[selected_idx1]
                target_label_flatten_selected = target_label.view(-1)[selected_idx2]
                all_label_flatten_selected = torch.cat([mix_label_flatten_selected, target_label_flatten_selected],0)
                all_label = all_label_flatten_selected  # single gpu

                feature_bank.append(all_feature)
                label_bank.append(all_label)

                if step_count > step_save:
                    feature_bank = feature_bank[1:]
                    label_bank = label_bank[1:]
                else:
                    step_count += 1

                all_feature = torch.cat(feature_bank, 0)
                all_feature = all_feature.detach()
                all_label = torch.cat(label_bank, 0)

                eps = 1e-8
                pos = (mix2_feature_overlap * target_feature_overlap.detach()).sum(-1, keepdim=True) / temp  # [n, 1]

                # compute RCCR loss
                b = 2000
                def run(pos, mix_feature_overlap, all_feature_idx,all_label_idx,target_label_overlap, neg_max):
                    # print("gpu: {}, i_1: {}".format(gpu, i))
                    mask_idx = (all_label_idx.unsqueeze(0) != target_label_overlap.unsqueeze(-1)).float()  # [n, b]
                    neg_idx = (mix_feature_overlap @ all_feature_idx.T) / temp  # [n, b]
                    logits_neg_idx = (torch.exp(neg_idx - neg_max) * mask_idx).sum(-1)  # [n, ]
                    return logits_neg_idx

                def run_0(pos, mix_feature_overlap, all_feature_idx,all_label_idx,target_label_overlap):
                    # print("gpu: {}, i_1_0: {}".format(gpu, i))
                    mask_idx = (all_label_idx.unsqueeze(0) != target_label_overlap.unsqueeze(-1)).float()  # [n, b]
                    neg_idx = (mix_feature_overlap @ all_feature_idx.T) / temp  # [n, b]
                    neg_idx = torch.cat([pos, neg_idx], 1)  # [n, 1+b]
                    mask_idx = torch.cat([torch.ones(mask_idx.size(0), 1).float().cuda(), mask_idx], 1)  # [n, 1+b]
                    neg_max = torch.max(neg_idx, 1, keepdim=True)[0]  # [n, 1]
                    logits_neg_idx = (torch.exp(neg_idx - neg_max) * mask_idx).sum(-1)  # [n, ]
                    return logits_neg_idx, neg_max

                N = all_feature.size(0)
                logits_down = torch.zeros(pos.size(0)).float().cuda()
                # neg_max1 = torch.zeros_like(pos1)
                for i in range((N - 1) // b + 1):
                    # for i in range(1):
                    # print("gpu: {}, i: {}".format(gpu, i))
                    all_label_idx = all_label[i * b:(i + 1) * b]
                    # output_ul_all = torch.zeros_like(output_ul_all)
                    all_feature_idx = all_feature[i * b:(i + 1) * b]
                    if i == 0:
                        logits_neg_idx, neg_max = torch.utils.checkpoint.checkpoint(run_0, pos, mix2_feature_overlap, all_feature_idx,all_label_idx,target_label_overlap)
                    else:
                        logits_neg_idx = torch.utils.checkpoint.checkpoint(run, pos, mix2_feature_overlap, all_feature_idx,all_label_idx,target_label_overlap,neg_max)
                    logits_down += logits_neg_idx

                pos_mask = ((target_logits_overlap > pos_thresh_value)).float()
                # neg_max1 = torch.zeros_like(pos1)
                # logits1_down = torch.zeros_like(pos1)
                logits = torch.exp(pos - neg_max).squeeze(-1) / (logits_down + eps)
                L_contrastive = -torch.log(logits + eps)
                L_contrastive = (L_contrastive * pos_mask).sum() / (pos_mask.sum() + 1e-12)

                # dynamic weight_contrastive
                weight_contrastive = 0.01*math.exp((-5)*(1-(i_iter/num_iterations)**0.5))
                L_contrastive = L_contrastive*weight_contrastive

                loss = L_l + L_consistency + L_contrastive
            else:
                loss = L_l+L_consistency
        else:
            loss = L_l


        if len(gpus) > 1:
            #print('before mean = ',loss)
            loss = loss.mean()
            #print('after mean = ',loss)
            loss_l_value += L_l.mean().item()
            if train_unlabeled:
                loss_consistency_value += L_consistency.mean().item()
                if i_iter >= iter_start_contrastive:
                    loss_contrastive_value += L_contrastive.mean().item()
        else:
            loss_l_value += L_l.item()
            if train_unlabeled:
                loss_consistency_value += L_consistency.item()
                if i_iter >= iter_start_contrastive:
                    loss_contrastive_value += L_contrastive.item()

        loss.backward()
        optimizer.step()

        #exit(1)
        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = 0.99
            #teacher model params = student model params before the unsup iter start
            #if i_iter <= iter_start_unsup:
            #    alpha_teacher = 0
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter)
        
        #print(checkpoint_dir)
        _t['iter time'].toc(average=False)
        print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_consistency = {3:.3f}, loss_contrastive = {4:.6f}, iter time = {5:.2f} s'.format(
            i_iter, num_iterations, loss_l_value, loss_consistency_value, loss_contrastive_value,  _t['iter time'].diff))

        if i_iter % save_checkpoint_every == 0 and i_iter!=0:
            if epochs_since_start * len(trainloader) < save_checkpoint_every:
                _save_checkpoint(i_iter, model, optimizer, config, ema_model, overwrite=False)
            else:
                _save_checkpoint(i_iter, model, optimizer, config, ema_model)
                # if i_iter > 200000:
                #     _save_checkpoint(i_iter, model, optimizer, config, ema_model, overwrite=False)
                # else:
                #     _save_checkpoint(i_iter, model, optimizer, config, ema_model)

        if config['utils']['tensorboard']:
            if 'tensorboard_writer' not in locals():
                tensorboard_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

            accumulated_loss_l.append(loss_l_value)
            if train_unlabeled:
                accumulated_loss_consistency.append(loss_consistency_value)
                if i_iter >= iter_start_contrastive:
                    accumulated_loss_contrastive.append(loss_contrastive_value)

            if i_iter % log_per_iter == 0 and i_iter != 0:

                tensorboard_writer.add_scalar('Training/Supervised loss', np.mean(accumulated_loss_l), i_iter)
                accumulated_loss_l = []

                if train_unlabeled:
                    tensorboard_writer.add_scalar('Training/consistency loss', np.mean(accumulated_loss_consistency), i_iter)
                    accumulated_loss_consistency = []
                    if i_iter >= iter_start_contrastive:
                        tensorboard_writer.add_scalar('Training/contrastive loss', np.mean(accumulated_loss_contrastive), i_iter)
                        accumulated_loss_contrastive = []

        if i_iter % val_per_iter == 0 and i_iter != 0:
            model.eval()
            if dataset == 'cityscapes':
                mIoU, _, _ = evaluate(model, dataset, ignore_label=255, input_size=(512,1024), save_dir=checkpoint_dir)
                mIoU = mIoU * 100
                print('Cur mIoU', mIoU)
                print('Cur mIoU iter', i_iter)
                print('Best mIoU', best_mIoU)
                print('Best mIoU iter', best_mIoU_iter)
                f_mIoU = open(osp.join(checkpoint_dir, 'mIoU.txt'), 'a')
                # f_mIoU.write('{0:.4f}\n'.format(mIoU))
                f_mIoU.write('{0:d} steps: {1:.4f}\n'.format(i_iter, mIoU))
                f_mIoU.close()            

            model.train()

            if mIoU > best_mIoU and save_best_model:
                best_mIoU = mIoU
                best_mIoU_iter = i_iter
                print('Best mIoU', best_mIoU)
                print('Best mIoU iter', best_mIoU_iter)
                _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

            if config['utils']['tensorboard']:
                tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
                # tensorboard_writer.add_scalar('Validation/Loss', eval_loss, i_iter)


    _save_checkpoint(num_iterations, model, optimizer, config, ema_model)

    model.eval()
    if dataset == 'cityscapes':
        mIoU, _, _ = evaluate(model, dataset, ignore_label=255, input_size=(512,1024), save_dir=checkpoint_dir)
        mIoU = mIoU * 100
        print('Cur mIoU', mIoU)
        print('Cur mIoU iter', i_iter)
        f_mIoU = open(osp.join(checkpoint_dir, 'mIoU.txt'), 'a')
        # f_mIoU.write('{0:.4f}\n'.format(mIoU))
        f_mIoU.write('{0:d} steps: {1:.4f}\n'.format(num_iterations, mIoU))
        f_mIoU.close()                    
    model.train()
    if mIoU > best_mIoU and save_best_model:
        best_mIoU = mIoU
        best_mIoU_iter = i_iter
        print('Best mIoU', best_mIoU)
        print('Best mIoU iter', best_mIoU_iter)
        _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

    if config['utils']['tensorboard']:
        tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
        # tensorboard_writer.add_scalar('Validation/Loss', val_loss, i_iter)


    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + 'seconds')

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()

    if False:#args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']


    if config['pretrained'] == 'coco':
        # restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
        # restore_from = './model/resnet101COCO-41f33a49.pth'
        #imageNet pretrained model
        restore_from = './model/DeepLab_init.pth'

    num_classes=19
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label'] 

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable)
    log_dir = checkpoint_dir

    warm_up_iter = config['utils']['warm_up_iter']
    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    if args.save_images:
        print('Saving unlabeled images')
        save_unlabeled_images = True
    else:
        save_unlabeled_images = False

    gpus = (0,1,2,3)[:args.gpus]

    main()
