# generate fs3.npy for W PC version
import clip
import torch
import numpy as np

from PIL import Image
from types import SimpleNamespace
from stylegan2.models import Generator # for manipulation code

# def GetImgF(out,model,preprocess,device)
# def GetFs(fs)

# Shift images to CLIP image feature 
def GetImgF(out,model,preprocess,device):
    imgs=out
    imgs1=imgs.reshape([-1]+list(imgs.shape[2:]))
    
    tmp=[]
    for i in range(len(imgs1)):
        
        img=Image.fromarray(imgs1[i])
        image = preprocess(img).unsqueeze(0).to(device)
        tmp.append(image)
    
    image=torch.cat(tmp)
    with torch.no_grad():
        image_features = model.encode_image(image)
    
    image_features1=image_features.cpu().numpy()
    image_features1=image_features1.reshape(list(imgs.shape[:2])+[512])
    
    return image_features1

def GetFs(fs):
    # Take mean of given features 
    # Returned output fs3 : (6048, 512)
    tmp=np.linalg.norm(fs,axis=-1)
    fs1=fs/tmp[:,:,:,None]
    fs2=fs1[:,:,1,:]-fs1[:,:,0,:]  # 5*sigma - (-5)* sigma
    fs3=fs2/np.linalg.norm(fs2,axis=-1)[:,:,None]
    fs3=fs3.mean(axis=1)
    fs3=fs3/np.linalg.norm(fs3,axis=-1)[:,None]
    return fs3

# Load PCA basis vectors
def load_components(dump_name, inst):
    global components, state, use_named_latents

    data = np.load(dump_name, allow_pickle=False)
    X_comp = data['act_comp']
    X_mean = data['act_mean']
    X_stdev = data['act_stdev']
    Z_comp = data['lat_comp']
    Z_mean = data['lat_mean']
    Z_stdev = data['lat_stdev']
    random_stdev_act = np.mean(data['random_stdevs'])
    n_comp = X_comp.shape[0]
    data.close()

    # Transfer to GPU
    components = SimpleNamespace(
        X_comp = torch.from_numpy(X_comp).cuda().float(),
        X_mean = torch.from_numpy(X_mean).cuda().float(),
        X_stdev = torch.from_numpy(X_stdev).cuda().float(),
        Z_comp = torch.from_numpy(Z_comp).cuda().float(),
        Z_stdev = torch.from_numpy(Z_stdev).cuda().float(),
        Z_mean = torch.from_numpy(Z_mean).cuda().float(),
        names = [f'Component {i}' for i in range(n_comp)],
        latent_types = [inst.model.latent_space_name()]*n_comp,
        ranges = [(0, inst.model.get_max_latents())]*n_comp,
    )

    # state.component_class = class_name # invalidates cache
    # use_named_latents = False
    print('Loaded components for from', dump_name)

if __name__ == "__main__":
    dump_name = ""
    # LOAD PCA vectors to global var components
    load_components(dump_name)