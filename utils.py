import torch
import copy
import clip
import pickle
import numpy as np

imagenet_templates = [
    'a bad photo of a {}.',
#    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]



def GetDt(classnames, model):
    """
    classnames: [target, neutral]
    model: CLIP 
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in imagenet_templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        text_features= torch.stack(zeroshot_weights, dim=1).cuda().t()
    
    dt=text_features[0]-text_features[1] # target embedding - neutral embedding
    dt=dt.cpu().numpy()
    dt=dt/np.linalg.norm(dt)
    return dt

def GetBoundary(fs3, dt, threshold, style_space, style_names):
    """
    fs3: collection of predefined styles (6048, 512)
    dt: Embedding delta text (, 512)
    
    ds_imp: deviation of style
        channelwise style movement * dText
    """
    tmp=np.dot(fs3,dt) # (6048, )
    ds_imp=copy.copy(tmp)
    zeros=np.abs(tmp) < threshold # Select channels that did not exceed threshold
    num_c=np.sum(~zeros)

    ds_imp[zeros]=0 # deviation of style is set to 0

    tmp=np.abs(ds_imp).max()
    ds_imp/=tmp # Normalize deviation

    boundary_tmp2, dlatents=SplitS(ds_imp, style_names, style_space)
    return boundary_tmp2, num_c, dlatents
        
def SplitS(ds_p, style_names, style_space):
    """
    6048개의 channel을 styleGAN 사이즈에 맞게 잘 잘라줌 ^^
    """
    all_ds=[]
    start=0
    tmp="./npy/ffhq/"+'S'
    with open(tmp, "rb") as fp:   #Pickling
        _, dlatents=pickle.load(fp)
    tmp="./npy/ffhq/"+'S_mean_std'
    with open(tmp, "rb") as fp:   #Pickling
        m, std=pickle.load( fp)

    for i, name in enumerate(style_names):

        if "torgb" not in name:
            tmp=style_space[i].shape[1]
            end=start+tmp
            tmp=ds_p[start:end] * std[i]
            all_ds.append(tmp)
            start=end
        else:
            tmp = np.zeros(len(dlatents[i][0]))
            all_ds.append(tmp)
    return all_ds, dlatents

def MSCode(dlatent_tmp, boundary_tmp, manipulate_layers, num_images, alpha, device):
    """
    dlatent_tmp: 이미지의 latent w를 s space로 mapping 해준것
    boundary_tmp: 6048 tmp를 s space 사이즈 맞춰 잘라줌 
    Retunrs manipulated Style Space
    """
    step=len(alpha)
    dlatent_tmp1=[tmp.reshape((num_images,-1)) for tmp in dlatent_tmp]
    dlatent_tmp2=[np.tile(tmp[:,None],(1,step,1)) for tmp in dlatent_tmp1]

    l=np.array(alpha)
    l=l.reshape([step if axis == 1 else 1 for axis in range(dlatent_tmp2[0].ndim)])
    if manipulate_layers is None:
        tmp=np.arange(len(boundary_tmp))
    else:
        tmp = [manipulate_layers]

    for i in tmp:
        dlatent_tmp2[i] += l * boundary_tmp[i]
    
    codes=[]
    for i in range(len(dlatent_tmp2)):
        tmp=list(dlatent_tmp[i].shape)
        tmp.insert(1,step)
        code = torch.Tensor(dlatent_tmp2[i].reshape(tmp))
        codes.append(code.to(device))
    return codes
