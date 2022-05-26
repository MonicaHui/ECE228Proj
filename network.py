import torch
import torch.nn as nn
import torch.nn.functional as F


# kernel decomposition
class DecomposedConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1):
        super(DecomposedConv2d, self).__init__()
        
        self.decomposed_conv = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=dim_in, bias=True),
            nn.InstanceNorm2d(dim_in),
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out,
                      kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        return self.decomposed_conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        
        self.residual_conv = nn.Sequential(
            DecomposedConv2d(dim_in, dim_out, kernel_size=3, stride=1),
            nn.InstanceNorm2d(dim_out),
            nn.ReLU(inplace=True),
            DecomposedConv2d(dim_out, dim_out, kernel_size=3, stride=1),
            nn.InstanceNorm2d(dim_out)
        )

    def forward(self, x):
        return x + self.residual_conv(x)
    
class generator(nn.Module):
    def __init__(self, dim_in = 3, dim_out = 3, hidden_dim = 64):
        super(generator, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size = 7, stride = 1, padding = 3),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.down_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True)
        )

        # residual_blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4)
        )



        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, dim_out, kernel_size = 7, stride = 1, padding = 3),
        )

    def forward(self, x):
        in_down = self.in_conv(x)
        out_down = self.down_conv(in_down)
        out_res = self.residual_blocks(out_down)
        out_up = self.up_conv(out_res)
        output = self.out_conv(out_up)
        return output
    
    
    

class generator_att(nn.Module):
    def __init__(self, dim_in = 3, dim_out = 3, hidden_dim = 64):
        super(generator_att, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size = 7, stride = 1, padding = 3),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.down_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True)
        )

        # residual_blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4),
            ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4)
        )

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.up_conv_content = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, 2, 1, 1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3, 2, 1, 1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.deconv3_content = nn.Conv2d(hidden_dim, 27, 7, 1, 0)
        
        self.up_conv_attention = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, 2, 1, 1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3, 2, 1, 1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 10, 1, 1, 0),
        )
        self.deconv3_attention_mask = nn.Conv2d(hidden_dim, 10, 1, 1, 0)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, dim_out, kernel_size = 7, stride = 1, padding = 3),
        )
        self.tanh = nn.Tanh()
        

    def forward(self, x):
        in_down = self.in_conv(x)
        out_down = self.down_conv(in_down)
        out_res = self.residual_blocks(out_down)
        
        # print(out_res.shape)
        
        #introduce two different up conv to get the content_mask and attention_mask
        out_res_content = self.up_conv_content(out_res)
        out_res_content = F.pad(out_res_content, (3, 3, 3, 3), 'reflect')
        content = self.deconv3_content(out_res_content)
        image_mask = self.tanh(content)


        image_mask1 = image_mask[:, 0:3, :, :]
        print("image_mask1.size()")
        print(image_mask1.size()) # [1, 3, 256, 256]
        image_mask2 = image_mask[:, 3:6, :, :]
        image_mask3 = image_mask[:, 6:9, :, :]
        image_mask4 = image_mask[:, 9:12, :, :]
        image_mask5 = image_mask[:, 12:15, :, :]
        image_mask6 = image_mask[:, 15:18, :, :]
        image_mask7 = image_mask[:, 18:21, :, :]
        image_mask8 = image_mask[:, 21:24, :, :]
        image_mask9 = image_mask[:, 24:27, :, :]
        
        out_res_attention = self.up_conv_attention(out_res)
        softmax_ = torch.nn.Softmax(dim=1)
        attention_mask = softmax_(out_res_attention)
        
        attention_mask1_ = attention_mask[:, 0:1, :, :]
        attention_mask2_ = attention_mask[:, 1:2, :, :]
        attention_mask3_ = attention_mask[:, 2:3, :, :]
        attention_mask4_ = attention_mask[:, 3:4, :, :]
        attention_mask5_ = attention_mask[:, 4:5, :, :]
        attention_mask6_ = attention_mask[:, 5:6, :, :]
        attention_mask7_ = attention_mask[:, 6:7, :, :]
        attention_mask8_ = attention_mask[:, 7:8, :, :]
        attention_mask9_ = attention_mask[:, 8:9, :, :]
        attention_mask10_ = attention_mask[:, 9:10, :, :]
        
        attention_mask1 = attention_mask1_.repeat(1, 3, 1, 1)
        print("attention_mask1.size()")
        # print(attention_mask1.size())
        attention_mask2 = attention_mask2_.repeat(1, 3, 1, 1)
        attention_mask3 = attention_mask3_.repeat(1, 3, 1, 1)
        attention_mask4 = attention_mask4_.repeat(1, 3, 1, 1)
        attention_mask5 = attention_mask5_.repeat(1, 3, 1, 1)
        attention_mask6 = attention_mask6_.repeat(1, 3, 1, 1)
        attention_mask7 = attention_mask7_.repeat(1, 3, 1, 1)
        attention_mask8 = attention_mask8_.repeat(1, 3, 1, 1)
        attention_mask9 = attention_mask9_.repeat(1, 3, 1, 1)
        attention_mask10 = attention_mask10_.repeat(1, 3, 1, 1)
        
        output1 = image_mask1 * attention_mask1
        output2 = image_mask2 * attention_mask2
        output3 = image_mask3 * attention_mask3
        output4 = image_mask4 * attention_mask4
        output5 = image_mask5 * attention_mask5
        output6 = image_mask6 * attention_mask6
        output7 = image_mask7 * attention_mask7
        output8 = image_mask8 * attention_mask8
        output9 = image_mask9 * attention_mask9
        output10 = x * attention_mask10
        
        AllOutPut = output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10
        
        # return AllOutPut, output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, attention_mask1,attention_mask2,attention_mask3, attention_mask4, attention_mask5, attention_mask6, attention_mask7, attention_mask8,attention_mask9,attention_mask10, image_mask1, image_mask2,image_mask3,image_mask4,image_mask5,image_mask6,image_mask7,image_mask8,image_mask9
        return AllOutPut
        # out_up = self.up_conv(out_res)
        # output = self.out_conv(out_up)
        # return output



class discriminator(nn.Module):
    def __init__(self, dim_in = 3, dim_out = 1, hidden_dim = 32):
        super(discriminator, self).__init__()
        self.flat_conv = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size = 3, stride = 1, padding = 1), 
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size = 3, stride = 2, padding = 1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size = 3, stride = 1, padding = 1), 
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size = 3, stride = 2, padding = 1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size = 3, stride = 1, padding = 1), 
            nn.InstanceNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size = 3, stride = 1, padding = 1), 
            nn.InstanceNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )


        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 8, dim_out, kernel_size = 3, stride = 1, padding = 1),
        )

    def forward(self, x):
        x = self.flat_conv(x)
        x = self.down_conv(x)
        output = self.out_conv(x)
        return output

