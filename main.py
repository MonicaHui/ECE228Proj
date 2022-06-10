import os
import time
import torch
import argparse


import torch.optim as optim
import numpy as np
import torch.nn as nn
from torchvision.models import vgg19
from torchvision import transforms
from torch.autograd import Variable
from torchvision import datasets

import matplotlib.pyplot as plt
import network, util

parser = argparse.ArgumentParser()

parser.add_argument('--test', action='store_true',
                    help='use this command to test generator')
parser.add_argument('--num_epoch', type=int, default=100,
                    help='num of training epoch')                    

parser.add_argument('--init_num_epoch', type=int, default=10,
                    help='num of initialization epoch')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--root_path', default="./result", help='save path')
parser.add_argument('--lr_G', type=float, default=0.0001,
                    help='learning rate of Generator')
parser.add_argument('--lr_D', type=float, default=0.0001,
                    help='learning rate of Discriminator')
parser.add_argument('--gamma_G', type=float, default=0.1, help='gamma_G')
parser.add_argument('--gamma_D', type=float, default=0.1, help='gamma_D')
parser.add_argument('--beta_1', type=float, default=0.5, help='beta_1')
parser.add_argument('--beta_2', type=float, default=0.99, help='beta_2')
parser.add_argument('--cont_lambda', type=float, default=10, help='cont_lambda')
parser.add_argument('--gray_lambda', type=float, default=20, help='gray_lambda')
parser.add_argument('--adv_lambda', type=float, default=5, help='adv_lambda for generator')
parser.add_argument('--add_attention', type=bool, default=True, help='use generator_att')
parser.add_argument('--add_losses', type=bool, default=True, help='add gray loss and color loss')
parser.add_argument('--load_model', type=bool, default=False, help='load previous model')
parser.add_argument('--is_pretrained', type=bool, default=False, help='is pretrained')
parser.add_argument('--is_smoothed', type=bool, default=False, help='is smoothed')
parser.add_argument('--start_epoch', type=int, default=0, help='start from which epoch')

opt = parser.parse_args()

if opt.add_attention:
    G = network.generator_att() # use our model
else:
    G = network.generator() # structure from CartoonGAN

D = network.discriminator()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


BCE = nn.BCELoss().to(device)
sig = nn.Sigmoid().to(device)
L1 = nn.L1Loss().to(device)
tahn = nn.Tanh().to(device)
huber = nn.HuberLoss().to(device)

G_optimizer = optim.Adam(G.parameters(), lr = opt.lr_G,
                         betas=(opt.beta_1, opt.beta_2))
D_optimizer = optim.Adam(D.parameters(), lr = opt.lr_D,
                         betas=(opt.beta_1, opt.beta_2))

G.to(device)
D.to(device)

vgg = vgg19(pretrained=False)
vgg.load_state_dict(torch.load('./vgg19.pth'))

# extract feature from VGG19 conv4_4
vgg = vgg.features[:26]
vgg.to(device)
vgg.eval()

# normolization
transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

src_path = "./data/real/train"
cart_path= "./data/cartoon/train"
cart_smooth_path = "./data/cartoon/edge_smoothed"
test_path = "./data/real/test"

if not opt.is_smoothed:
    if not os.path.isdir(cart_smooth_path):
        os.mkdir(cart_smooth_path)
    util.edge_smooth(cart_path, cart_smooth_path)


src_loader = torch.utils.data.DataLoader(datasets.ImageFolder(src_path, transform), batch_size=opt.batch_size, shuffle=True, drop_last=True)
cartoon_smooth_loader = torch.utils.data.DataLoader(datasets.ImageFolder(cart_smooth_path,transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])), batch_size=opt.batch_size, shuffle=True, drop_last=True)
test_loader=torch.utils.data.DataLoader(datasets.ImageFolder(test_path, transform), batch_size=opt.batch_size, shuffle=False, drop_last=True)

# use content loss to pre-train generator
def pretrain():
    print("start pretrain")
    train_hist = {}
    train_hist['Cont_losses'] = []

    for epoch in range(opt.start_epoch, opt.init_num_epoch):
        Cont_losses = []
        

        for i, img in enumerate(src_loader):
            src = img[0]
            src = src.to(device)

            # train generator
            G_optimizer.zero_grad()
            gen_cart = tahn(G(src))

            src_feature = vgg(src)
            gen_feature = vgg(gen_cart)
            Cont_loss = opt.cont_lambda * L1(src_feature, gen_feature)

            G_loss = Cont_loss
            
            G_loss.backward()
            G_optimizer.step()

            Cont_losses.append(Cont_loss.item())
            train_hist['Cont_losses'].append(Cont_loss.item())

            if i % 50 == 0:
                print("i: %s, Content_loss: %.3f" % (i, Cont_loss.item()))
                real = src[0].cpu().detach().numpy().transpose(1, 2, 0)
                cart = gen_cart[0].cpu().detach().numpy().transpose(1, 2, 0)
                result = np.concatenate((real, cart), axis=1)
                result = (result + 1) / 2
                filename = "pretrain_%s_%s.png" % (epoch, i)
                path = os.path.join("./result", filename)
                plt.imsave(path, result)

        average_cont_loss = np.mean(Cont_losses)

        print("epoch: %s, Content_loss: %.3f" % (epoch, average_cont_loss))

        if not os.path.isdir('models/'):
            os.mkdir('models/')

        save_path = os.path.join('models/', "pretrain-model" + str(epoch) + ".ckpt")
        torch.save({
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'G_optim_state': G_optimizer.state_dict(),
                'D_optim_state': D_optimizer.state_dict(),
            }, save_path)



# our model
def train_gray():
    print("start training -- add gray loss, color loss")
    train_hist = {}
    train_hist['G_losses'] = []
    train_hist['D_losses'] = []
    train_hist['Cont_losses'] = []
    train_hist['Gray_losses'] = []
    if opt.load_model:
        load_model(os.path.join('models/', "model.ckpt"))

    for epoch in range(opt.start_epoch, opt.num_epoch):
        
        start_time = time.time()
        
        G_losses = []
        D_losses = []
        Cont_losses = []
        Gray_losses = []
        for i, img in enumerate(zip(src_loader,cartoon_smooth_loader)):
            
            src, cartoon = img[0][0], img[1][0]
            cart = cartoon[:,:,:,0:256]
            cart_smooth = cartoon[:,:,:,256:]
            src = src.to(device)
            cart = cart.to(device)
            cart_smooth = cart_smooth.to(device)

            gen_cart = tahn(G(src))

            #train discriminator
            D_optimizer.zero_grad()
            
            D_adv_c = sig(D(cart))
            D_adv_e = sig(D(cart_smooth))
            D_adv_g = sig(D(gen_cart.detach()))

            D_adv_c_loss = BCE(D_adv_c, Variable(torch.ones(D_adv_c.size()).to(device)))
            D_adv_e_loss = BCE(D_adv_e, Variable(torch.zeros(D_adv_e.size()).to(device)))     
            D_adv_g_loss  = BCE(D_adv_g, Variable(torch.zeros(D_adv_g.size()).to(device)))      
            

            D_loss = D_adv_c_loss + D_adv_e_loss + D_adv_g_loss

            D_loss.backward() 
            D_optimizer.step()

            # train generator
            G_optimizer.zero_grad()

            G_adv = D(gen_cart)
            src_feature = vgg(src)
            gen_feature = vgg(gen_cart)
            Cont_loss = opt.cont_lambda * L1(src_feature, gen_feature)

            G_adv = sig(G_adv)
            G_adv_loss = opt.adv_lambda * BCE(G_adv, Variable(torch.ones(G_adv.size()).to(device)))
        
            # cal gray loss 
            gray = transforms.functional.rgb_to_grayscale(cart)
            gray = gray.to(device)
            
            gray_input = Variable(torch.zeros(src.size()).to(device))
            gray = np.squeeze(gray)

            gray_input[:,0,:,:] = gray
            gray_input[:,1,:,:] = gray
            gray_input[:,2,:,:] = gray
            gray_feature = vgg(gray_input)

            gray_gram = util.cal_gram(gray_feature)
            gen_cart_gram = util.cal_gram(gen_feature)
            
            Gray_loss = opt.gray_lambda * L1(gray_gram, gen_cart_gram)
            
            # cal color loss 
            yuv_gen = util.rgb_to_yuv(gen_cart)
            yuv_src = util.rgb_to_yuv(src)
            
            Color_loss = L1(yuv_src[:,0,:,:], yuv_gen[:,0,:,:])+ huber(yuv_src[:,1,:,:],yuv_gen[:,1,:,:]) + huber(yuv_src[:,2,:,:],yuv_gen[:,2,:,:])

            G_loss = Cont_loss + G_adv_loss + Gray_loss + Color_loss
            
            G_loss.backward()
            G_optimizer.step()

            # finish one iter

            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())
            Cont_losses.append(Cont_loss.item())
            Gray_losses.append(Gray_loss.item())
            
            train_hist['G_losses'].append(G_loss.item())
            train_hist['D_losses'].append(D_loss.item())
            train_hist['Cont_losses'].append(Cont_loss.item())
            train_hist['Gray_losses'].append(Gray_loss.item())
            

            if i % 50 == 0:
                print("i: %s, G_loss: %.3f, D_loss: %.3f, Content_loss: %.3f, Gray_loss: %.3f" % (i, G_loss.item(),D_loss.item(),Cont_loss.item(),Gray_loss.item()))
                real = src[0].cpu().detach().numpy().transpose(1, 2, 0)
                cart = gen_cart[0].cpu().detach().numpy().transpose(1, 2, 0)
                result = np.concatenate((real, cart), axis=1)
                result = (result + 1) / 2
                if not os.path.isdir('result/'):
                    os.mkdir('result/')
                filename = "train_%s_%s.png" % (epoch, i)
                path = os.path.join("./result", filename)
                plt.imsave(path, result)
        
        end_time = time.time()
        epoch_time = end_time - start_time

        average_G_loss = np.mean(G_losses)
        average_D_loss = np.mean(D_losses)
        average_cont_loss = np.mean(Cont_losses)
        average_gray_loss = np.mean(Gray_losses)


        print("epoch: %s, epoch time: %0.3f, G_loss: %.3f, D_loss: %.3f, Content_loss: %.3f, Gray_loss: %.3f" % (epoch,epoch_time,average_G_loss,average_D_loss,average_cont_loss, average_gray_loss))
        if not os.path.isdir('models/'):
                os.mkdir('models/')

        save_path = os.path.join('models/', "model-gray" + str(epoch) + ".ckpt")
        torch.save({
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'G_optim_state': G_optimizer.state_dict(),
                'D_optim_state': D_optimizer.state_dict(),
            }, save_path)

 # original CartoonGAN       
def train():

    print("start training")
    train_hist = {}
    train_hist['G_losses'] = []
    train_hist['D_losses'] = []
    train_hist['Cont_losses'] = []
    if opt.load_model:
        load_model(os.path.join('models/', "model.ckpt"))

    for epoch in range(opt.start_epoch, opt.num_epoch):
        
        start_time = time.time()
        
        G_losses = []
        D_losses = []
        Cont_losses = []
        for i, img in enumerate(zip(src_loader,cartoon_smooth_loader)):           
            src, cartoon = img[0][0], img[1][0]

            cart = cartoon[:,:,:,0:256]
            cart_smooth = cartoon[:,:,:,256:]
            
            src = src.to(device)
            cart = cart.to(device)
            cart_smooth = cart_smooth.to(device)

            gen_cart = tahn(G(src))

            #train discriminator
            D_optimizer.zero_grad()
            
            D_adv_c = sig(D(cart))
            D_adv_e = sig(D(cart_smooth))
            D_adv_g = sig(D(gen_cart.detach()))

            D_adv_c_loss = BCE(D_adv_c, Variable(torch.ones(D_adv_c.size()).to(device)))
            D_adv_e_loss = BCE(D_adv_e, Variable(torch.zeros(D_adv_e.size()).to(device)))     
            D_adv_g_loss  = BCE(D_adv_g, Variable(torch.zeros(D_adv_g.size()).to(device)))      
            

            D_loss = D_adv_c_loss + D_adv_e_loss + D_adv_g_loss

            D_loss.backward() 
            D_optimizer.step()

            # train generator
            G_optimizer.zero_grad()

            G_adv = D(gen_cart)
            src_feature = vgg(src)
            gen_feature = vgg(gen_cart)
            Cont_loss = opt.cont_lambda * L1(src_feature, gen_feature)

            G_adv = sig(G_adv)
            G_adv_loss = opt.adv_lambda * BCE(G_adv, Variable(torch.ones(G_adv.size()).to(device)))

            G_loss = Cont_loss + G_adv_loss
            
            G_loss.backward()
            G_optimizer.step()

            # finish one iter

            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())
            Cont_losses.append(Cont_loss.item())
            
            train_hist['G_losses'].append(G_loss.item())
            train_hist['D_losses'].append(D_loss.item())
            train_hist['Cont_losses'].append(Cont_loss.item())
            
            if i % 50 == 0:
                print("i: %s, G_loss: %.3f, D_loss: %.3f, Content_loss: %.3f" % (i, G_loss.item(),D_loss.item(),Cont_loss.item()))
                real = src[0].cpu().detach().numpy().transpose(1, 2, 0)
                cart = gen_cart[0].cpu().detach().numpy().transpose(1, 2, 0)
                result = np.concatenate((real, cart), axis=1)
                result = (result + 1) / 2
                if not os.path.isdir('result/'):
                    os.mkdir('result/')
                filename = "train_%s_%s.png" % (epoch, i)
                path = os.path.join("./result", filename)
                plt.imsave(path, result)
        
        end_time = time.time()
        epoch_time = end_time - start_time

        average_G_loss = np.mean(G_losses)
        average_D_loss = np.mean(D_losses)
        average_cont_loss = np.mean(Cont_losses)


        print("epoch: %s, epoch time: %0.3f, G_loss: %.3f, D_loss: %.3f, Content_loss: %.3f" % (epoch,epoch_time,average_G_loss,average_D_loss,average_cont_loss))
        if not os.path.isdir('models/'):
                os.mkdir('models/')

        save_path = os.path.join('models/', "model" + str(epoch) + ".ckpt")
        torch.save({
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'G_optim_state': G_optimizer.state_dict(),
                'D_optim_state': D_optimizer.state_dict(),
            }, save_path)


def test():
        assert opt.root_path,'Providing the path of trained models before the start of testing'
        print("start testing")
        if opt.load_model:
            load_model(os.path.join('models/',"model.ckpt"))

        #read test images and generate images
        for i, img in enumerate(zip(test_loader)):
            test_image=img[0][0]
            test_image=test_image.to(device)

            with torch.no_grad():
                gen_test=tahn(G(test_image))
            
            test_images=test_image[0].cpu().detach().numpy().transpose(1, 2, 0)
            gen_test_images = gen_test[0].cpu().detach().numpy().transpose(1, 2, 0)

            if not os.path.isdir('result/test'):
                os.mkdir('result/test')
            
            #print("start generating test images")
            #test different generators and compare with input images
            result_gen = np.concatenate((test_images, gen_test_images), axis=1)
            result_gen=(result_gen+1)/2
            filename_gen = "test_generator_%s.png" % (i)
            path_gen = os.path.join("./result/test", filename_gen)
            plt.imsave(path_gen, result_gen)


    
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G_state'])
    D.load_state_dict(checkpoint['D_state'])
    G_optimizer.load_state_dict(checkpoint['G_optim_state'])
    D_optimizer.load_state_dict(checkpoint['D_optim_state'])


        
def main():
    if not opt.is_pretrained:
        pretrain()
    if opt.add_losses:
        train_gray()
    else:
        train()
    test()

if __name__ == "__main__":
    main()
    
