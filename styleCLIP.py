import clip
import numpy as np
import argparse
from PIL import Image

import torch
from torch.nn import functional as F

from stylegan2.models import Generator
from utils import *

def conv_warper(layer, input, style, noise):
    # the conv should change
    conv = layer.conv
    batch, in_channel, height, width = input.shape
    
    style = style.view(batch, 1, in_channel, 1, 1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample: # up==2
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
    # image + self.weight * noise
    out = layer.noise(out, noise=noise)
    out = layer.activate(out)
    
    return out

def decoder(G, style_space, latent, noise):
    """
    G: stylegan generator
    style_space: S space of size 9088 (6048 for conv / 3040 for torgb)
    latent: W space vector of size 512
    noise: noise for each layer of styles (predefined from encoder step)
    """
    out = G.input(latent)
    out = conv_warper(G.conv1, out, style_space[0], noise[0])
    skip = G.to_rgb1(out, latent[:, 0])
    i = 2; j = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
        skip = to_rgb(out, latent[:, j + 2], skip) # style space manipulation not used in torgb
        i += 3; j += 2
    image = skip

    return image

def encoder(G, latent): 
    noise_constants = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    style_space = []
    style_names = []
    # rgb_style_space = []
    style_space.append(G.conv1.conv.modulation(latent[:, 0]))
    res=4
    style_names.append(f"b{res}/conv1")
    style_space.append(G.to_rgbs[0].conv.modulation(latent[:, 0]))
    style_names.append(f"b{res}/torgb")
    i = 1;j=3

    for conv1, conv2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], G.to_rgbs
    ):
        res=2**j
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_names.append(f"b{res}/conv1")
        style_space.append(conv2.conv.modulation(latent[:, i+1]))
        style_names.append(f"b{res}/conv2")
        style_space.append(to_rgb.conv.modulation(latent[:, i + 2]))
        style_names.append(f"b{res}/torgb")
        i += 2; j += 1
        
    return style_space, style_names, noise_constants

def visual(output, save=False, name="original"):
    output = (output + 1)/2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    output = output[0].detach().cpu().permute(1,2,0).numpy()
    output = (output*255).astype(np.uint8)
    img = Image.fromarray(output)
    if save:
        img.save(f"results/{name}.png")
    return output

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Configuration for styleCLIP')
    parser.add_argument('--neutral', type=str, default='A girl', help='Neutral text without additional information of the image')
    parser.add_argument('--target', type=str, help='Target text to manipulate the image generated')
    parser.add_argument('--alpha', type=float, default=5.0, help='Manipulation strength, Between -10 ~ 10')
    parser.add_argument('--beta', type=float, default=0.08, help='Manipulation threshold, Between 0.08 ~ 3')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Torch device on {device}")
    config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}

    generator = Generator(
            size= 1024,
            style_dim=config["latent"],
            n_mlp=config["n_mlp"],
            channel_multiplier=config["channel_multiplier"]
        )
    generator.load_state_dict(torch.load("model/stylegan2-ffhq-config-f.pt")['g_ema'])
    generator.eval()
    generator.to(device)

    # Load W space latent vector of size 1, 18, 512
    latent = torch.load("./model/latent.pt")
    # Load style space of S from the latent/ Extract noise constant from pretrained model
    style_space, style_names, noise_constants = encoder(generator, latent)
    image = decoder(generator, style_space, latent, noise_constants)
    tmp = visual(image, orig=True)

    # StyleCLIP
    model, _ = clip.load("ViT-B/32", device=device)
    fs3 = np.load('./npy/ffhq/fs3.npy') # 6048, 512
    np.set_printoptions(suppress=True)

    classnames=[args.target, args.neutral]
    dt=GetDt(classnames, model) # normalized deviation of t
   
    boundary_tmp2, c, dlatents = GetBoundary(fs3, dt, args.beta, style_space, style_names) # Move each channel by dStyle
    dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
    manipulated_s= MSCode(dlatents_loaded, boundary_tmp2, manipulate_layers=None, num_images=1, alpha=[args.alpha], device=device)
    image = decoder(generator, manipulated_s, latent, noise_constants)
    tmp = visual(image, orig=False)
    print(f"Generated Image {args.target}")