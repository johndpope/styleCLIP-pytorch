# generate fs3.npy for W PC version
import clip
import torch
import argparse
import numpy as np
import sys
import copy
import pickle
from PIL import Image
from types import SimpleNamespace
from stylegan2.models import Generator
from styleCLIP import encoder, decoder, MSCode, visual
# def GetImgF(out,model,preprocess,device)
# def GetFs(fs)

# Shift images to CLIP image feature 
def GetImgF(out, model, preprocess, device):
    imgs=out
    imgs1=imgs.reshape([-1]+list(imgs.shape[2:]))
    
    tmp=[]
    for i in range(len(imgs1)):
        
        img=Image.fromarray(imgs1[i])
        image = preprocess(img).unsqueeze(0).to(device)
        tmp.append(image)
    
    image=torch.cat(tmp)
    with torch.no_grad():
        image_features = model.encode_image(image) # 2*num_images, 512
    
    image_features1=image_features.cpu().numpy()
    image_features1=image_features1.reshape(list(imgs.shape[:2])+[512])
    
    return image_features1

def GetFs(fs):
    # Take mean of given features 
    # Returned output fs3 : (6048, 512)
    print(f"fs shape {fs.shape}")
    tmp=np.linalg.norm(fs,axis=-1)
    print(f"tmp shape {tmp.shape}")
    fs1=fs/tmp[:,:,:,None]
    fs2=fs1[:,:,1,:]-fs1[:,:,0,:]  # 5*sigma - (-5)* sigma
    fs3=fs2/np.linalg.norm(fs2,axis=-1)[:,:,None]
    fs3=fs3.mean(axis=1)
    fs3=fs3/np.linalg.norm(fs3,axis=-1)[:,None]
    print(f"GetFs {fs3.shape}")
    return fs3

# Load PCA basis vectors
def load_components(dump_name, inst=None):
    global components, state, use_named_latents
    data = np.load(dump_name, allow_pickle=False)
    X_comp = data['act_comp']
    X_mean = data['act_mean']
    X_stdev = data['act_stdev']
    Z_comp = data['lat_comp']
    Z_mean = data['lat_mean']
    Z_stdev = data['lat_stdev']
    # random_stdev_act = np.mean(data['random_stdevs'])
    n_comp = X_comp.shape[0]
    data.close()
    return X_comp
    # Transfer to GPU
    # components = SimpleNamespace(
    #     X_comp = torch.from_numpy(X_comp).cuda().float(),
    #     X_mean = torch.from_numpy(X_mean).cuda().float(),
    #     X_stdev = torch.from_numpy(X_stdev).cuda().float(),
    #     Z_comp = torch.from_numpy(Z_comp).cuda().float(),
    #     Z_stdev = torch.from_numpy(Z_stdev).cuda().float(),
    #     Z_mean = torch.from_numpy(Z_mean).cuda().float(),
    #     names = [f'Component {i}' for i in range(n_comp)],
    #     # latent_types = [inst.model.latent_space_name()]*n_comp,
    #     # ranges = [(0, inst.model.get_max_latents())]*n_comp,
    # )
def w_stat(w_samples, std=True, num_samples=100000):
    w_samples = w_samples[:num_samples]
    shape = w_samples.shape[1]
    stat = np.zeros((1, 512),dtype='float32')
    for i in range(shape):
        if std:
            tmp = w_samples[:,i].std(axis=0)
        else:
            tmp = w_samples[:, i].mean(axis=0)
        stat[:, i] = tmp
    return stat[:, None, :]

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    print(f"Torch is Using Device {device}")
    parser = argparse.ArgumentParser(description='Configuration for W space version of delta image in CLIP space')
    parser.add_argument('--use_w', action="store_true")
    parser.add_argument('--n_layer', type=int, default=None, help='Index of layer to manipulate')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to encode CLIP')
    parser.add_argument('--file_path', type=str, default="./npy/ffhq/", help="Path where W/S statistcs are stored")
    parser.add_argument('--save', action="store_true", help="Saves manipulated image")
    args = parser.parse_args()
    
    dump_name = args.file_path + "stylegan2-ffhq_style_ipca_c50_n1000000_w.npz"
    comp = load_components(dump_name) # 1, 50, 512
    
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

    #* Load W mean *#
    tmp = args.file_path+"W.npy"
    w_samples = np.load(tmp)
    w_std = w_stat(w_samples, std=True) # 1, 512
    w_mean = w_stat(w_samples, std=False)
    w_samples = w_samples[:args.num_images]

    #* Load CLIP language/Image encoder *#
    model, preprocess = clip.load("ViT-B/32", device=device)

    if args.use_w:
        all_features = []
        for layer_idx in range(generator.n_latent):
            if args.n_layer is not None:
                layer_idx = args.n_layer
            for pc_idx, pc in enumerate(comp):
                pc = np.expand_dims(pc, 0) # 1, 512 --> 1, 1, 512
                out=np.zeros((args.num_images, 2, 1024, 1024, 3), dtype='uint8')
                i = 1
                for img_idx in range(args.num_images):
                    w_plus = np.tile(w_samples[img_idx], (1, 18, 1))
                    delta_w = np.zeros_like(w_plus)
                    for alpha_idx, alpha in enumerate([50, -50]):
                        print(i)

                        delta_w[:, layer_idx, :] = pc
                        alpha *= w_std
                        latent = torch.Tensor(w_plus + alpha * delta_w).to(device)
                        style_space, style_names, noise_constants = encoder(generator, latent)
                        image = decoder(generator, style_space, latent, noise_constants)
                        out[img_idx, alpha_idx, :, :, :] = visual(image, save=args.save, name=str(i))
                        i += 1

                tmp = []
                img = out.reshape([-1]+list(out.shape[2:]))
                for i in range(len(img)): # 2 * 100
                    imgs=Image.fromarray(img[i])
                    _ = visual(imgs, save=args.save, name="manipulated")
                    image = preprocess(imgs).unsqueeze(0).to(device)
                    tmp.append(image)
                print(f"Length of tmp {len(tmp)}")
                image = torch.cat(tmp)
                with torch.no_grad():
                    image_features = model.encode_image(image)

                image_features = image_features.cpu().numpy()
                image_features = image_features.reshape(list(out.shape[:2])+[512])
            if args.n_layer is not None:
                break
            all_features.append(image_features)           
        all_features = np.array(all_features)
        all_features = GetFs(all_features)
        
        np.save(args.file_path+'fs3',all_features)
    else:
        sys.quit("Only W space manipulation is supported!")
