import argparse


import numpy as np
import torch


from config import get_config
# from dataloaders import utils
# from dataloaders.dataset import (BaseDataSets, RandomGenerator,
#                                  TwoStreamBatchSampler)
from networks.unet import UNet
from networks.vision_transformer import SwinUnet as ViT_seg
# from utils import losses, metrics, ramps
# from val_2D import test_single_volume
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import h5py
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')

parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../XAI/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

args = parser.parse_args()
config = get_config(args)

def predict():

    def get_vaild_transform():
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485],
                std=[0.229],
                max_pixel_value=255.0
                ),
            ToTensorV2(),
        ])
        return transform

    num_classes = args.num_classes



    model1 = UNet(in_chns=1, class_num=num_classes).cuda()
    model2 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    
    # model2.load_from(config)


    model1.load_state_dict(torch.load("/home/student/SSL4MIS/model/ACDC/Cross_Teaching_Between_CNN_Transformer_7/unet/model1_iter_30000_dice_0.8636.pth"))
    model2.load_state_dict(torch.load("/home/student/SSL4MIS/model/ACDC/Cross_Teaching_Between_CNN_Transformer_7/unet/model2_iter_29400_dice_0.853.pth"))
    model1.eval()
    model2.eval()

    gt = []
    #

    predres1 = []
    predres2 = []

    with open(args.root_path + "/valACDC.list", "r") as f:
        sample_list = f.readlines()
        sample_list = [item.replace("\n", "") for item in sample_list]

    for i in range(len(sample_list)) :
        sample = sample_list[i]
        h5f = h5py.File(args.root_path + "/data/slices/{}.h5".format(sample), "r")

        img = h5f["image"][:]
        label = h5f["label"][:]
        print(img.shape)
        # plt.imsave(f'/home/student/Desktop/ACDC/img/{sample}.png', img)
        # plt.imsave(f'/home/student/Desktop/ACDC/annotation/{sample}.png', label)
        # plt.imshow(label, "gray")
        # plt.show()
        # hf1.create_dataset('img', data=img)
        # hf1.create_dataset('label', data=label)
        # hf2.create_dataset('img', data=img)
        # hf2.create_dataset('label', data=label)
        
        # img = vaild_transform(image=img)['image'].unsqueeze(0).cuda()
        plt.figure(num='noise',figsize=(8,8))
        img = cv2.resize(img, (224, 224))
        img_org = img.copy()
        
        img_MSA = img.copy()
        x, y = np.meshgrid(np.linspace(-1, 1, img.shape[0]), np.linspace(-1, 1, img_MSA.shape[1]))
        f = np.exp(-(x**2 + y**2) / (2*0.1**2))
        print("f",f.shape)
        b = f - 1

        # simulate the magnetic field inhomogeneity artifact
        img_with_mti = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_with_mti[i, j] = img[i, j] * np.exp(-b[i, j])
        

        x,y,w,h = 85,85,20,20 
        img_noise = img.copy()
        roi = img_noise[x:x+w, y:y+h]
        mean = 0
        variance = 0.1
        sigma = np.sqrt(variance)
        gaussian = np.random.normal(mean, sigma, roi.shape)
        noisy_image = np.clip(roi + gaussian, 0, 1.0)
        img_noise[x:x+w, y:y+h] = noisy_image
        img = img_with_mti.copy()
        # cv2.circle(img, (85, 85), 20, 0, -1)
        #plt.subplot(3,4,1, )
        #plt.title('mask image')
        #plt.imshow(img)
        #plt.axis('off') 
        #plt.subplot(3,4,5)
        #plt.title('origin image')
        #plt.imshow(img_org)
        #plt.axis('off') 
        label= cv2.resize(label, (224, 224))

        roi = label[x:x+w, y:y+h]
        mask_rate = [0,0,0,0]
        # 计算ROI中每个像素值的数量
        unique, counts = np.unique(roi, return_counts=True)
        # pixel_counts = dict(zip(unique, counts))
        unique_label, counts_label = np.unique(label, return_counts=True)
        # pixel_counts_label = dict(zip(unique_label, counts_label))
        
        try:
            mask_rate[0] = counts[0]/counts_label[0]
        except:
            mask_rate[0] = 0
        try:
            mask_rate[1] = counts[1]/counts_label[1]
        except:
            mask_rate[1] = 0
        try:
            mask_rate[2] = counts[2]/counts_label[2]
        except:
            mask_rate[2] = 0
        try:
            mask_rate[3] = counts[3]/counts_label[3]
        except:
            mask_rate[3] = 0

        print(mask_rate)
        

        #plt.subplot(3,4,9)
        #plt.title('noise image')
        #plt.axis('off')
        #plt.imshow(img_noise,)
        
        #plt.subplot(3,4,2)
        #plt.title('label image')
        #plt.axis('off') 
        #plt.imshow(label,)
        #plt.subplot(3,4,6)
        #plt.title('label image')
        #plt.axis('off') 
        #plt.imshow(label,)     
        #plt.subplot(3,4,10)
        #plt.title('label image')
        #plt.axis('off') 
        #plt.imshow(label,)

        
        mask = torch.tensor(label).unsqueeze(0).cuda()
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).cuda()
        img_org = torch.tensor(img_org).unsqueeze(0).unsqueeze(0).cuda()
        img_noise = torch.tensor(img_noise).unsqueeze(0).unsqueeze(0).cuda()

        print(mask.shape)
        with torch.no_grad():
            img= img.to(torch.float32)
            img_org = img_org.to(torch.float32)
            img_noise = img_noise.to(torch.float32)



            out1org = model1(img_org)
            out1 = model1(img)
            out1noice = model1(img_noise)            
            out2org = model2(img_org)
            out2 = model2(img)
            out2noice = model2(img_noise)
            
            probs1 = torch.softmax(out1, dim=1)
            _, preds1 = torch.max(probs1, dim=1)
            pred_img1 = preds1.cpu().detach().numpy().transpose(1, 2, 0)

          
            probs1org = torch.softmax(out1org, dim=1)
            _, preds1org = torch.max(probs1org, dim=1)
            pred_img1org = preds1org.cpu().detach().numpy().transpose(1, 2, 0)
            predres1.append(pred_img1)

            probs1noice = torch.softmax(out1noice, dim=1)
            _, preds1noice = torch.max(probs1noice, dim=1)
            pred_img1noice = preds1noice.cpu().detach().numpy().transpose(1, 2, 0)
            # pred_img1 = pred_img1.unsqueeze(0)
            
            w ,h = pred_img1.shape[0], pred_img1.shape[1]
            pred_img1 = pred_img1.reshape(w,h)

            w, h = pred_img1org.shape[0], pred_img1org.shape[1]
            pred_img1org = pred_img1org.reshape(w,h)

            w, h = pred_img1noice.shape[0], pred_img1noice.shape[1]
            pred_img1noice = pred_img1noice.reshape(w,h)


            #plt.subplot(3,4,3)
            #plt.title('unet mask image')
            #plt.axis('off') 
            #plt.imshow(pred_img1)
            #plt.subplot(3,4,7)
            #plt.title('unet origin image')
            #plt.axis('off') 
            #plt.imshow(pred_img1org)
            #plt.subplot(3,4,11)
            #plt.title('unet noise image')
            #plt.axis('off')
            #plt.imshow(pred_img1noice)
            # print(pred_img1.shape)
            ##plt.imsave(f'/home/student/Desktop/ACDC/predict1/{sample}.png', pred_img1)
            # hf1.create_dataset('predict', data=pred_img1)
            probs2 = torch.softmax(out2, dim=1)
            _, preds2 = torch.max(probs2, dim=1)
            pred_img2 = preds2.cpu().detach().numpy().transpose(1, 2, 0)

            probs2org = torch.softmax(out2org, dim=1)
            _, preds2org = torch.max(probs2org, dim=1)
            pred_img2org = preds2org.cpu().detach().numpy().transpose(1, 2, 0)

            probs2noice = torch.softmax(out2noice, dim=1)
            _, preds2noice = torch.max(probs2noice, dim=1)
            pred_img2noice = preds2noice.cpu().detach().numpy().transpose(1, 2, 0)
            # pred_img2 = pred_img2.unsqueeze(0)
            predres2.append(pred_img2)
            w ,h = pred_img2.shape[0], pred_img2.shape[1]
            pred_img2 = pred_img2.reshape(w,h)

            w, h = pred_img2org.shape[0], pred_img2org.shape[1]
            pred_img2org = pred_img2org.reshape(w,h)

            w, h = pred_img2noice.shape[0], pred_img2noice.shape[1]
            pred_img2noice = pred_img2noice.reshape(w,h)

            #plt.subplot(3,4,4)
            #plt.title('swin mask image')
            #plt.axis('off') 
            #plt.imshow(pred_img2)
            #plt.subplot(3,4,8)
            #plt.title('swin origin image')
            #plt.axis('off') 
            #plt.imshow(pred_img2org)
            #plt.subplot(3,4,12)
            #plt.title('swin noise image')
            #plt.axis('off')
            #plt.imshow(pred_img2noice)
            
            #plt.show()
            ##plt.imsave(f'/home/student/Desktop/ACDC/predict2/{sample}.png', pred_img2)

            # #plt.show()
            # hf2.create_dataset('predict', data=pred_img2)
            # hf1.close()
            # hf2.close()
            # predres2.append(pred_img2)
            # predres1.append(pred_img1)
            # gt.append(mask_img)
            mask_img = mask.cpu().detach().numpy().transpose(1, 2, 0)
            gt.append(mask_img)

    import metric
    iou1 = metric.mean_iou(predres1, gt, num_classes, None,)
    iou2 = metric.mean_iou(predres2, gt, num_classes, None,)

    print("intersect1 ", iou1)
    print("intersect2 ", iou2)



if __name__ == '__main__':


    predict()
