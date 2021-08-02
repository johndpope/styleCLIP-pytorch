import os
import sys

import torch 
import scipy
import numpy as np
import datetime
import argparse
from tqdm import trange
from stylegan2.wrapper import get_instrumented_model # call model

# ganspace folder import
from ganspace.PCA_estimator import get_estimator  # PCA module

### From here, Reference code is https://github.com/harskish/ganspace.git
# Especially decomposition.py 

# Define SEED 
SEED_SAMPLING = 1
SEED_RANDOM_DIRS = 2
SEED_LINREG = 3
SEED_VISUALIZATION = 5

# random direction
def get_random_dirs(components, dimensions):
    gen = np.random.RandomState(seed=SEED_RANDOM_DIRS)
    dirs = gen.normal(size=(components, dimensions))
    dirs /= np.sqrt(np.sum(dirs**2, axis=1, keepdims=True))
    return dirs.astype(np.float32)

# Solve for directions in latent space that match PCs in activaiton space
def linreg_lstsq(comp_np, mean_np, stdev_np, inst, config):
    global B
    print('Performing least squares regression', flush=True)

    torch.manual_seed(SEED_LINREG)
    np.random.seed(SEED_LINREG)

    comp = torch.from_numpy(comp_np).float().to(inst.model.device)
    mean = torch.from_numpy(mean_np).float().to(inst.model.device)
    stdev = torch.from_numpy(stdev_np).float().to(inst.model.device)

    n_samp = max(10_000, config.n) // B * B # make divisible
    n_comp = comp.shape[0]
    latent_dims = inst.model.get_latent_dims()
    
    # We're looking for M s.t. M*P*G'(Z) = Z => M*A = Z
    #   Z = batch of latent vectors (n_samples x latent_dims)
    #   G'(Z) = batch of activations at intermediate layer
    #   A = P*G'(Z) = projected activations (n_samples x pca_coords)
    #   M = linear mapping (pca_coords x latent_dims)

    # Minimization min_M ||MA - Z||_l2 rewritten as min_M.T ||A.T*M.T - Z.T||_l2
    # to match format expected by pytorch.lstsq

    # TODO: regression on pixel-space outputs? (using nonlinear optimizer)
    # min_M lpips(G_full(MA), G_full(Z))

    # Tensors to fill with data
    # Dimensions other way around, so these are actually the transposes
    A = np.zeros((n_samp, n_comp), dtype=np.float32)
    Z = np.zeros((n_samp, latent_dims), dtype=np.float32)
    
    # Project tensor X onto PCs, return coordinates
    def project(X, comp):
        N = X.shape[0]
        K = comp.shape[0]
        coords = torch.bmm(comp.expand([N]+[-1]*comp.ndim), X.view(N, -1, 1))
        return coords.reshape(N, K)

    for i in trange(n_samp // B, desc='Collecting samples', ascii=True):
        z = inst.model.sample_latent(B)
        inst.model.partial_forward(z, config.layer)
        act = inst.retained_features()[config.layer].reshape(B, -1)

        # Project X onto the PC components
        act = act - mean
        coords = project(act, comp)
        coords_scaled = coords / stdev

        A[i*B:(i+1)*B] = coords_scaled.detach().cpu().numpy()
        Z[i*B:(i+1)*B] = z.detach().cpu().numpy().reshape(B, -1)

    # Solve least squares fit

    # gelsd = divide-and-conquer SVD; good default
    # gelsy = complete orthogonal factorization; sometimes faster
    # gelss = SVD; slow but less memory hungry
    M_t = scipy.linalg.lstsq(A, Z, lapack_driver='gelsd')[0] # torch.lstsq(Z, A)[0][:n_comp, :]
    
    # Solution given by rows of M_t
    Z_comp = M_t[:n_comp, :]
    Z_mean = np.mean(Z, axis=0, keepdims=True)

    return Z_comp, Z_mean

def regression(comp, mean, stdev, inst, config):
    # Sanity check: verify orthonormality
    M = np.dot(comp, comp.T)
    if not np.allclose(M, np.identity(M.shape[0])):
        det = np.linalg.det(M)
        print(f'WARNING: Computed basis is not orthonormal (determinant={det})')

    return linreg_lstsq(comp, mean, stdev, inst, config)

# Sample Z instances and compute PCA
# Components are returned in 
def compute(config, device):
    global B

    timestamp = lambda : datetime.datetime.now().strftime("%d.%m %H:%M")
    print(f'[{timestamp()}] Computing')

    # Ensure reproducibility
    torch.manual_seed(0) # also sets cuda seeds
    np.random.seed(0)

    # Speed up backend
    torch.backends.cudnn.benchmark = True
    layer_key = config.layer

    # get StyleGANv2 model -> wrapped with StyleGanv2 module and instrumented model
    inst = get_instrumented_model(config.model, config.output_class, layer_key, device)
    model = inst.model
    
    # Regress back to w space
    if config.use_w:
        print('Using W latent space')
        model.use_w()

    # BigGAN에 사용한 방법을 동일하게 StyleGAN에 적용
    inst.retain_layer(layer_key)
    model.partial_forward(model.sample_latent(1), layer_key)
    sample_shape = inst.retained_features()[layer_key].shape 
    sample_dims = np.prod(sample_shape) # 512
    print('Feature shape:', sample_shape) # 1, 512

    input_shape = inst.model.get_latent_shape()
    input_dims = inst.model.get_latent_dims()

    # number of components to keep
    config.components = min(config.components, sample_dims)
    # IPCAEstimator: batch size of max(100, 2*n_components)
    transformer = get_estimator(config.estimator, config.components) # PCA component

    X = None
    X_global_mean = None

    # Figure out batch size if not provided
    B = config.batch_size if config.batch_size is not None else 1
    # Divisible by B (ignored in output name)
    N = config.n // B * B

    # Compute maximum batch size based on RAM + pagefile budget
    target_bytes = 20 * 1_000_000_000 # GB
    feat_size_bytes = sample_dims * np.dtype('float64').itemsize
    N_limit_RAM = np.floor_divide(target_bytes, feat_size_bytes)
    if not transformer.batch_support and N > N_limit_RAM:
        print('WARNING: estimator does not support batching, ' \
            'given config will use {:.1f} GB memory.'.format(feat_size_bytes / 1_000_000_000 * N))

    # 32-bit LAPACK gets very unhappy about huge matrices (in linalg.svd)
    # if config.estimator == 'ica':
    #     lapack_max_N = np.floor_divide(np.iinfo(np.int32).max // 4, sample_dims) # 4x extra buffer
    #     if N > lapack_max_N:
    #         raise RuntimeError(f'Matrices too large for ICA, please use N <= {lapack_max_N}')

    print('B={}, N={}, dims={}, N/dims={:.1f}'.format(B, N, sample_dims, N/sample_dims), flush=True)

    # Must not depend on chosen batch size (reproducibility)
    NB = max(B, max(2_000, 3*config.components)) # ipca: as large as possible!
    
    samples = None
    if not transformer.batch_support:
        samples = np.zeros((N + NB, sample_dims), dtype=np.float32)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Use exactly the same latents regardless of batch size
    # Store in main memory, since N might be huge (1M+)
    # Run in batches, since sample_latent() might perform Z -> W mapping
    n_lat = ((N + NB - 1) // B + 1) * B
    latents = np.zeros((n_lat, *input_shape[1:]), dtype=np.float32)
    with torch.no_grad():
        for i in trange(n_lat // B, desc='Sampling latents'):
            latents[i*B:(i+1)*B] = model.sample_latent(n_samples=B).cpu().numpy()

    # Decomposition on non-Gaussian latent space
    samples_are_latents = layer_key in ['g_mapping', 'style'] and inst.model.latent_space_name() == 'W'

    canceled = False
    try:
        X = np.ones((NB, sample_dims), dtype=np.float32)
        action = 'Fitting' if transformer.batch_support else 'Collecting'
        for gi in trange(0, N, NB, desc=f'{action} batches (NB={NB})', ascii=True):
            for mb in range(0, NB, B):
                z = torch.from_numpy(latents[gi+mb:gi+mb+B]).to(device)
                
                if samples_are_latents:
                    # Decomposition on latents directly (e.g. StyleGAN W)
                    batch = z.reshape((B, -1))
                else:
                    # Decomposition on intermediate layer
                    with torch.no_grad():
                        model.partial_forward(z, layer_key)
                    
                    # Permuted to place PCA dimensions last
                    batch = inst.retained_features()[layer_key].reshape((B, -1))

                space_left = min(B, NB - mb)
                X[mb:mb+space_left] = batch.cpu().numpy()[:space_left]

            if transformer.batch_support:
                if not transformer.fit_partial(X.reshape(-1, sample_dims)):
                    break
            else:
                samples[gi:gi+NB, :] = X.copy()
    except KeyboardInterrupt:
        if not transformer.batch_support:
            sys.exit(1) # no progress yet
        
        canceled = True
        
    if not transformer.batch_support:
        X = samples # Use all samples
        X_global_mean = X.mean(axis=0, keepdims=True, dtype=np.float32) # TODO: activations surely multi-modal...!
        X -= X_global_mean
        
        print(f'[{timestamp()}] Fitting whole batch')
        t_start_fit = datetime.datetime.now()

        transformer.fit(X)
        
        print(f'[{timestamp()}] Done in {datetime.datetime.now() - t_start_fit}')
        assert np.all(transformer.transformer.mean_ < 1e-3), 'Mean of normalized data should be zero'
    else:
        X_global_mean = transformer.transformer.mean_.reshape((1, sample_dims))
        X = X.reshape(-1, sample_dims)
        X -= X_global_mean

    X_comp, X_stdev, X_var_ratio = transformer.get_components()
    
    assert X_comp.shape[1] == sample_dims \
        and X_comp.shape[0] == config.components \
        and X_global_mean.shape[1] == sample_dims \
        and X_stdev.shape[0] == config.components, 'Invalid shape'

    # 'Activations' are really latents in a secondary latent space
    if samples_are_latents:
        Z_comp = X_comp
        Z_global_mean = X_global_mean
    else:
        Z_comp, Z_global_mean = regression(X_comp, X_global_mean, X_stdev, inst, config)

    # Normalize
    Z_comp /= np.linalg.norm(Z_comp, axis=-1, keepdims=True)

    # Random projections
    # We expect these to explain much less of the variance
    random_dirs = get_random_dirs(config.components, np.prod(sample_shape))
    n_rand_samples = min(5000, X.shape[0])
    X_view = X[:n_rand_samples, :].T
    assert np.shares_memory(X_view, X), "Error: slice produced copy"
    X_stdev_random = np.dot(random_dirs, X_view).std(axis=1)

    # Inflate back to proper shapes (for easier broadcasting)
    X_comp = X_comp.reshape(-1, *sample_shape)
    X_global_mean = X_global_mean.reshape(sample_shape)
    Z_comp = Z_comp.reshape(-1, *input_shape)
    Z_global_mean = Z_global_mean.reshape(input_shape)

    # Compute stdev in latent space if non-Gaussian
    lat_stdev = np.ones_like(X_stdev)
    if config.use_w:
        samples = model.sample_latent(5000).reshape(5000, input_dims).detach().cpu().numpy()
        coords = np.dot(Z_comp.reshape(-1, input_dims), samples.T)
        lat_stdev = coords.std(axis=1)

    dump_name = "{}-{}_{}_{}_n{}{}{}.npz".format(
        config.model.lower(),
        config.output_class.replace(' ', '_'),
        config.layer.lower(),
        transformer.get_param_str(),
        config.n,
        '_w' if config.use_w else '',
        f'_seed{config.seed}' if config.seed else ''
    )

    # dump_path = 'cache' / 'components' / dump_name

    # Save components File 
    # os.makedirs(dump_name.parent, exist_ok=True)
    print(f"save npz files")
    print(f"X_comp {X_comp.shape}")
    np.savez_compressed(dump_name, **{
        'act_comp': X_comp.astype(np.float32),
        'act_mean': X_global_mean.astype(np.float32),
        'act_stdev': X_stdev.astype(np.float32),
        'lat_comp': Z_comp.astype(np.float32),
        'lat_mean': Z_global_mean.astype(np.float32),
        'lat_stdev': lat_stdev.astype(np.float32),
        'var_ratio': X_var_ratio.astype(np.float32),
        'random_stdevs': X_stdev_random.astype(np.float32),
    })
    
    if canceled:
        sys.exit(1)

    del inst
    del model

    del X
    del X_comp
    del random_dirs
    del batch
    del samples
    del latents
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAN component analysis config')
    parser.add_argument('--model', dest='model', type=str, default='StyleGAN2', help='The network to analyze') # StyleGAN, DCGAN, ProGAN, BigGAN-XYZ
    parser.add_argument('--layer', dest='layer', type=str, default='g_mapping', help='The layer to analyze') # Analyze all layers if None
    parser.add_argument('--class', dest='output_class', type=str, default='ffhq', help='Output class to generate (BigGAN: Imagenet, ProGAN: LSUN)')
    parser.add_argument('--est', dest='estimator', type=str, default='ipca', help='The algorithm to use [pca, fbpca, cupca, spca, ica]')
    parser.add_argument('--sparsity', type=float, default=1.0, help='Sparsity parameter of SPCA')
    parser.add_argument('--video', dest='make_video', action='store_true', help='Generate output videos (MP4s)')
    parser.add_argument('--batch', dest='batch_mode', action='store_true', help="Don't open windows, instead save results to file")
    parser.add_argument('-b', dest='batch_size', type=int, default=10_000, help='Minibatch size, leave empty for automatic detection')
    parser.add_argument('-c', dest='components', type=int, default=50, help='Number of components to keep')
    parser.add_argument('-n', type=int, default=1_000_000, help='Number of examples to use in decomposition')
    parser.add_argument('--use_w', action='store_true', help='Use W latent space (StyleGAN(2))')
    parser.add_argument('--sigma', type=float, default=2.0, help='Number of stdevs to walk in visualize.py')
    parser.add_argument('--inputs', type=str, default=None, help='Path to directory with named components')
    parser.add_argument('--seed', type=int, default=0, help='Seed used in decomposition')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Torch is Using Device {device}")
    compute(args, device)