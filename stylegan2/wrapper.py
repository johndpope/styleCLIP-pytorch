
import os
import re
import random
import requests

import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from abc import abstractmethod, ABC as AbstractBaseClass

from stylegan2.models import Generator

# Class BaseModel
# Class StyleGan2
# def get_model
# get get_instrumented_model

class BaseModel(AbstractBaseClass, torch.nn.Module):

    # Set parameters for identifying model from instance
    def __init__(self, model_name, class_name):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.outclass = class_name

    # Stop model evaluation as soon as possible after
    # given layer has been executed, used to speed up
    # netdissect.InstrumentedModel::retain_layer().
    # Validate with tests/partial_forward_test.py
    # Can use forward() as fallback at the cost of performance.
    @abstractmethod
    def partial_forward(self, x, layer_name):
        pass

    # Generate batch of latent vectors
    @abstractmethod
    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        pass

    # Maximum number of latents that can be provided
    # Typically one for each layer
    def get_max_latents(self):
        return 1

    # Name of primary latent space
    # E.g. StyleGAN can alternatively use W
    def latent_space_name(self):
        return 'Z'

    def get_latent_shape(self):
        return tuple(self.sample_latent(1).shape)

    def get_latent_dims(self):
        return np.prod(self.get_latent_shape())

    def set_output_class(self, new_class):
        self.outclass = new_class

    # Map from typical range [-1, 1] to [0, 1]
    def forward(self, x):
        out = self.model.forward(x)
        return 0.5*(out+1)

    # Generate images and convert to numpy
    def sample_np(self, z=None, n_samples=1, seed=None):
        if z is None:
            z = self.sample_latent(n_samples, seed=seed)
        elif isinstance(z, list):
            z = [torch.tensor(l).to(self.device) if not torch.is_tensor(l) else l for l in z]
        elif not torch.is_tensor(z):
            z = torch.tensor(z).to(self.device)
        img = self.forward(z)
        img_np = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        return np.clip(img_np, 0.0, 1.0).squeeze()

    # For models that use part of latent as conditioning
    def get_conditional_state(self, z):
        return None

    # For models that use part of latent as conditioning
    def set_conditional_state(self, z, c):
        return z

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)

# PyTorch port of StyleGAN 2
class StyleGAN2(BaseModel):
    def __init__(self, device, class_name, truncation=1.0, use_w=False):
        super(StyleGAN2, self).__init__('StyleGAN2', class_name or 'ffhq')
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = use_w # use W as primary latent space?
        self.name = f'StyleGAN2-{self.outclass}'
        self.has_latent_residual = True
        self.load_model()
        self.set_noise_seed(0)

    def latent_space_name(self):
        return 'W' if self.w_primary else 'Z'

    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False

    def load_model(self):
        model_dict = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
        self.model = Generator(size= 1024, style_dim=model_dict["latent"], n_mlp=model_dict["n_mlp"], channel_multiplier=model_dict["channel_multiplier"]).to(self.device)

        checkpoint = "model/stylegan2-ffhq-config-f.pt"
        if not os.path.exists(checkpoint):
            print("checkpoint doesn't exist")
            exit()
        
        ckpt = torch.load("model/stylegan2-ffhq-config-f.pt")
        self.model.load_state_dict(ckpt['g_ema'], strict=False)
        self.model.eval()
        self.latent_avg = ckpt['latent_avg'].to(self.device)

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max) # use (reproducible) global rand state

        rng = np.random.RandomState(seed)
        z = torch.from_numpy(
                rng.standard_normal(512 * n_samples)
                .reshape(n_samples, 512)).float().to(self.device) #[N, 512]
        
        if self.w_primary:
            z = self.model.style(z)

        return z

    def get_max_latents(self):
        return self.model.n_latent

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError('StyleGAN2: cannot change output class without reloading')
    
    def forward(self, x):
        x = x if isinstance(x, list) else [x]
        out, _ = self.model(x, noise=self.noise,
            truncation=self.truncation, truncation_latent=self.latent_avg, input_is_latent=self.w_primary)
        return 0.5*(out+1)

    def partial_forward(self, x, layer_name):
        styles = x if isinstance(x, list) else [x]
        inject_index = None
        noise = self.noise

        if not self.w_primary:
            styles = [self.model.style(s) for s in styles]

        if len(styles) == 1:
            # One global latent
            inject_index = self.model.n_latent
            latent = self.model.strided_style(styles[0].unsqueeze(1).repeat(1, inject_index, 1)) # [N, 18, 512]
        elif len(styles) == 2:
            # Latent mixing with two latents
            if inject_index is None:
                inject_index = random.randint(1, self.model.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.model.n_latent - inject_index, 1)

            latent = self.model.strided_style(torch.cat([latent, latent2], 1))
        else:
            # One latent per layer
            assert len(styles) == self.model.n_latent, f'Expected {self.model.n_latents} latents, got {len(styles)}'
            styles = torch.stack(styles, dim=1) # [N, 18, 512]
            latent = self.model.strided_style(styles)

        if 'style' in layer_name:
            return

        out = self.model.input(latent)
        if 'input' == layer_name:
            return

        out = self.model.conv1(out, latent[:, 0], noise=noise[0])
        if 'conv1' in layer_name:
            return

        skip = self.model.to_rgb1(out, latent[:, 1])
        if 'to_rgb1' in layer_name:
            return

        i = 1
        noise_i = 1

        for conv1, conv2, to_rgb in zip(
            self.model.convs[::2], self.model.convs[1::2], self.model.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise[noise_i])
            if f'convs.{i-1}' in layer_name:
                return

            out = conv2(out, latent[:, i + 1], noise=noise[noise_i + 1])
            if f'convs.{i}' in layer_name:
                return
            
            skip = to_rgb(out, latent[:, i + 2], skip)
            if f'to_rgbs.{i//2}' in layer_name:
                return

            i += 2
            noise_i += 2

        image = skip

        raise RuntimeError(f'Layer {layer_name} not encountered in partial_forward')

    def set_noise_seed(self, seed):
        torch.manual_seed(seed)
        self.noise = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=self.device)]

        for i in range(3, self.model.log_size + 1):
            for _ in range(2):
                self.noise.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=self.device))


def get_model(name, output_class, device, **kwargs):
    # Check if optionally provided existing model can be reused
    inst = kwargs.get('inst', None)
    model = kwargs.get('model', None)
    
    if inst or model:
        cached = model or inst.model
        
        network_same = (cached.model_name == name)
        outclass_same = (cached.outclass == output_class)
        can_change_class = ('BigGAN' in name)
        
        if network_same and (outclass_same or can_change_class):
            cached.set_output_class(output_class)
            return cached
    
    if name == 'StyleGAN2':
        model = StyleGAN2(device, output_class)
    else:
        raise RuntimeError(f'Unknown model {name}')

    return model

# return instrumented model
def get_instrumented_model(name, output_class, layers, device, **kwargs):
    model = get_model(name, output_class, device, **kwargs)
    model.eval()

    inst = kwargs.get('inst', None)
    if inst:
        inst.close()

    if not isinstance(layers, list):
        layers = [layers]

    # Verify given layer names
    module_names = [name for (name, _) in model.named_modules()]
    for layer_name in layers:
        if not layer_name in module_names:
            print(f"Layer '{layer_name}' not found in model!")
            print("Available layers:", '\n'.join(module_names))
            raise RuntimeError(f"Unknown layer '{layer_name}''")
    
    # Reset StyleGANs to z mode for shape annotation
    if hasattr(model, 'use_z'):
        model.use_z()

    from netdissect.modelconfig import create_instrumented_model
    inst = create_instrumented_model(SimpleNamespace(
        model = model,
        layers = layers,
        cuda = device.type == 'cuda',
        gen = True,
        latent_shape = model.get_latent_shape()
    ))

    if kwargs.get('use_w', False):
        model.use_w()

    return inst