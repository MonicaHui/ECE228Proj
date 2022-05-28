import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision import datasets



def edge_smooth(in_path, out_path):
    transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    cartoon_loader = torch.utils.data.DataLoader(datasets.ImageFolder(in_path, transform), batch_size=1, shuffle = False)
    for i, src in enumerate(cartoon_loader):
        # use canny to abstract edges
        img = src[0][0].numpy().transpose(1, 2, 0)
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
        kernel_big = np.ones((7,7), np.uint8)
        kernel_small = np.ones((5,5), np.uint8)
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
#         print(src[0][0].shape)
#         print(out.shape)

#         print(out)
        out = out / 256
        img_ = (img + 1) / 2
        result = np.concatenate((img_, out), axis=1)
#         print(result.shape)
        
        filename = "smoothed_%s.png" % i
        if not os.path.isdir(os.path.join(out_path, "1/")):
            os.mkdir(os.path.join(out_path, "1/"))
                 
        path = os.path.join(out_path, "1", filename)
        
#         print(path)
        plt.imsave(path, result)
#         cv2.imwrite(path, result)
                 


        
def cal_gram(x):
    batch_size, channel, height, width = x.shape
    x_flat = x.reshape((batch_size, channel, -1))
    x_flat_t = torch.transpose(x_flat, 1, 2)
    gram = torch.bmm(x_flat, x_flat_t)
    return gram / (channel * height * width)
        
    
    
def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out
        
        
        