import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision import datasets


# for cartoon image
def edge_smooth(in_path, out_path):
    transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    cartoon_loader = torch.utils.data.DataLoader(datasets.ImageFolder(in_path, transform), batch_size=1)
    for i, img in enumerate(cartoon_loader):
        # use canny to abstract edges
        print("start smooth")
        img = img[0][0].numpy().transpose(1, 2, 0)
        img_int = ((img + 1) / 2 * 255).astype(np.uint8)
#         filename = "img_int.png"
#         path = os.path.join("./result", filename)
#         cv2.imwrite(path, img_int)
#         print(img.shape)
        gray = cv2.cvtColor(img_int,cv2.COLOR_BGR2GRAY)
        
        filename = "gray.png"
        path = os.path.join("./result", filename)

#         gray = ((gray + 1) / 2 * 255).astype(np.uint8)
        cv2.imwrite(path, gray)
#         print(gray)
        edges = cv2.Canny(gray,100,200)
#         filename = "edge.png"
#         path = os.path.join("./result", filename)
#         cv2.imwrite(path, edges)
        
#         print("edge: ", edges)
        
        # dilate
        kernel_big = np.ones((5,5), np.uint8)
        kernel_small = np.ones((3,3), np.uint8)
        edge_dilation = cv2.dilate(edges, kernel_big, iterations=1)
        edge_small = cv2.dilate(edges, kernel_small, iterations=1)
        
#         filename = "edge_dilation.png"
#         path = os.path.join("./result", filename)
#         cv2.imwrite(path, edge_dilation)
        edge_region = np.zeros(img.shape)
        edge_dilation = np.clip(edge_dilation, 0, 1)
        edge_region[:,:,0] = img_int[:,:,0] * edge_dilation
        edge_region[:,:,1] = img_int[:,:,1] * edge_dilation
        edge_region[:,:,2] = img_int[:,:,2] * edge_dilation
#         filename = "edge_region.png"
#         path = os.path.join("./result", filename)
#         cv2.imwrite(path, edge_region)
        
        # apply a Gaussian smoothing in the dilated edge regions
        guass_edge = cv2.GaussianBlur(edge_region, (3, 3), 0, 0);
        guass_edge_region = np.zeros(guass_edge.shape)
        edge_small = np.clip(edge_small, 0, 1)
        guass_edge_region[:,:,0] = guass_edge[:,:,0] * edge_small
        guass_edge_region[:,:,1] = guass_edge[:,:,1] * edge_small
        guass_edge_region[:,:,2] = guass_edge[:,:,2] * edge_small
        
#         filename = "guass_edge.png"
#         path = os.path.join("./result", filename)
#         cv2.imwrite(path, guass_edge)

        thre_img = np.ones_like(edge_small) - edge_small
        
        out_edge = np.zeros(img_int.shape)
        out_edge[:,:,0] = img_int[:,:,0] * thre_img
        out_edge[:,:,1] = img_int[:,:,1] * thre_img
        out_edge[:,:,2] = img_int[:,:,2] * thre_img
        
        out = out_edge + guass_edge_region
        filename = "smoothed_%s.png" % i
        if not os.path.isdir(os.path.join(out_path, "1/")):
            os.mkdir(os.path.join(out_path, "1/"))
                 
        path = os.path.join(out_path, "1", filename)
        
#         print(path)
        
        cv2.imwrite(path, out)
                 


        
def cal_gram(x):
    batch_size, channel, height, width = x.shape
    x_flat = x.reshape((batch_size, channel, -1))
    x_flat_t = torch.transpose(x_flat, 1, 2)
    gram = torch.bmm(x_flat, x_flat_t)
    return gram / (channel * height * width)
        
        
        
        