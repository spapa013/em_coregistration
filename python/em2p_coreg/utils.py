import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import torch
from torch.nn import functional as F
import re
from PIL import Image
import json
import urllib
from scipy import ndimage

def create_grid(um_sizes, desired_res=1):
    """ Create a grid corresponding to the sample position of each pixel/voxel in a FOV of
     um_sizes at resolution desired_res. The center of the FOV is (0, 0, 0).
    In our convention, samples are taken in the center of each pixel/voxel, i.e., a volume
    centered at zero of size 4 will have samples at -1.5, -0.5, 0.5 and 1.5; thus edges
    are NOT at -2 and 2 which is the assumption in some libraries.
    :param tuple um_sizes: Size in microns of the FOV, .e.g., (d1, d2, d3) for a stack.
    :param float or tuple desired_res: Desired resolution (um/px) for the grid.
    :return: A (d1 x d2 x ... x dn x n) array of coordinates. For a stack, the points at
    each grid position are (x, y, z) points; (x, y) for fields. Remember that in our stack
    coordinate system the first axis represents z, the second, y and the third, x so, e.g.,
    p[10, 20, 30, 0] represents the value in x at grid position 10, 20, 30.
    """
    # Make sure desired_res is a tuple with the same size as um_sizes
    if np.isscalar(desired_res):
        desired_res = (desired_res,) * len(um_sizes)

    # Create grid
    out_sizes = [int(round(um_s / res)) for um_s, res in zip(um_sizes, desired_res)]
    um_grids = [np.linspace(-(s - 1) * res / 2, (s - 1) * res / 2, s, dtype=np.float32)
                for s, res in zip(out_sizes, desired_res)] # *
    full_grid = np.stack(np.meshgrid(*um_grids, indexing='ij')[::-1], axis=-1)
    # * this preserves the desired resolution by slightly changing the size of the FOV to
    # out_sizes rather than um_sizes / desired_res.

    return full_grid


def resize(original, um_sizes, desired_res):
    """ Resize array originally of um_sizes size to have desired_res resolution.
    We preserve the center of original and resized arrays exactly in the middle. We also
    make sure resolution is exactly the desired resolution. Given these two constraints,
    we cannot hold FOV of original and resized arrays to be exactly the same.
    :param np.array original: Array to resize.
    :param tuple um_sizes: Size in microns of the array (one per axis).
    :param int or tuple desired_res: Desired resolution (um/px) for the output array.
    :return: Output array (np.float32) resampled to the desired resolution. Size in pixels
        is round(um_sizes / desired_res).
    """

    # Create grid to sample in microns
    grid = create_grid(um_sizes, desired_res) # d x h x w x 3

    # Re-express as a torch grid [-1, 1]
    um_per_px = np.array([um / px for um, px in zip(um_sizes, original.shape)])
    torch_ones = np.array(um_sizes) / 2 - um_per_px / 2  # sample position of last pixel in original
    grid = grid / torch_ones[::-1].astype(np.float32)

    # Resample
    input_tensor = torch.from_numpy(original.reshape(1, 1, *original.shape).astype(
        np.float32))
    grid_tensor = torch.from_numpy(grid.reshape(1, *grid.shape))
    resized_tensor = F.grid_sample(input_tensor, grid_tensor, padding_mode='border')
    resized = resized_tensor.numpy().squeeze()

    return resized


def affine_product(X, A, b):
    """ Special case of affine transformation that receives coordinates X in 2-d (x, y)
    and affine matrix A and translation vector b in 3-d (x, y, z). Y = AX + b
    :param torch.Tensor X: A matrix of 2-d coordinates (d1 x d2 x 2).
    :param torch.Tensor A: The first two columns of the affine matrix (3 x 2).
    :param torch.Tensor b: A 3-d translation vector.
    :return: A (d1 x d2 x 3) torch.Tensor corresponding to the transformed coordinates.
    """
    return torch.einsum('ij,klj->kli', (A, X)) + b

def sample_grid(volume, grid):
    """ 
    Sample grid in volume.

    Assumes center of volume is at (0, 0, 0) and grid and volume have the same resolution.

    :param torch.Tensor volume: A d x h x w tensor. The stack.
    :param torch.Tensor grid: A d1 x d2 x 3 (x, y, z) tensor. The coordinates to sample.

    :return: A d1 x d2 tensor. The grid sampled in the stack.
    """
    # Make sure input is tensor
    volume = torch.as_tensor(volume, dtype=torch.float32)
    grid = torch.as_tensor(grid, dtype=torch.float32)

    # Rescale grid so it ranges from -1 to 1 (as expected by F.grid_sample)
    norm_factor = torch.as_tensor([s / 2 - 0.5 for s in volume.shape[::-1]])
    norm_grid = grid / norm_factor

    # Resample
    resampled = F.grid_sample(volume[None, None, ...], norm_grid[None, None, ...], padding_mode='zeros')
    resampled = resampled.squeeze() # drop batch and channel dimension

    return resampled

def html_to_json(url_string, return_parsed_url=False, fragment_prefix='!'):
    # Parse neuromancer url to logically separate the json state dict from the rest of it.
    full_url_parsed = urllib.parse.urlparse(url_string)
    # Decode percent-encoding in url, and skip "!" from beginning of string.
    decoded_fragment = urllib.parse.unquote(full_url_parsed.fragment)
    if decoded_fragment.startswith(fragment_prefix):
        decoded_fragment = decoded_fragment[1:]
    # Load the json state dict string into a python dictionary.
    json_state_dict = json.loads(decoded_fragment)

    if return_parsed_url:
        return json_state_dict, full_url_parsed
    else:
        return json_state_dict

def add_point_annotations(provided_link, ano_name, ano_list, voxelsize, overwrite=True):
    # format annotation list
    ano_list_dict = []
    if ano_list.ndim<2:
        ano_list = np.expand_dims(ano_list,0)
    if ano_list.ndim>2:
        return print('The annotation list must be 1D or 2D')
    for i, ano in enumerate(ano_list):
        ano_list_dict.append({'point':ano.tolist(), 'type':'point', 'id':str(i+1)})

    json_data, parsed_url = html_to_json(provided_link, return_parsed_url=True)
    # if annotation layer doesn't exist, create it
    if re.search(ano_name,json.dumps(json_data)) is None:
        json_data['layers'].append({'tool': 'annotatePoint',
                               'type': 'annotation',
                               'annotations': [],
                               'annotationTags': [],
                               'voxelSize': voxelsize,
                               'name': ano_name})
        print('annotation layer does not exist... creating it')
    annotation_dict = list(filter(lambda _: _['name'] == ano_name, json_data['layers']))
    annotation_ind = np.where(np.array(json_data['layers']) == annotation_dict)[0][0].squeeze()
    # test if voxel size of annotation matches provided voxel size
    if json_data['layers'][annotation_ind]['voxelSize']!=voxelsize:
        return print('The annotation layer already exists but does not match your provided voxelsize')
    # add annotations
    if overwrite:
        json_data['layers'][annotation_ind]['annotations'] = ano_list_dict
    else:
        json_data['layers'][annotation_ind]['annotations'].extend(ano_list_dict)

    return urllib.parse.urlunparse([parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, parsed_url.query, '!'+ urllib.parse.quote(json.dumps(json_data))])

def coordinate(grid_to_transform):
    x = grid_to_transform.shape[0]
    y = grid_to_transform.shape[1]
    return grid_to_transform.reshape(x*y,-1)

def uncoordinate(transformed_coordinates,x,y):
    return transformed_coordinates.reshape(x,y,-1)

def lcn(image, sigmas=(12, 12)):
    """ Local contrast normalization.
    Normalize each pixel using mean and stddev computed on a local neighborhood.
    We use gaussian filters rather than uniform filters to compute the local mean and std
    to soften the effect of edges. Essentially we are using a fuzzy local neighborhood.
    Equivalent using a hard defintion of neighborhood will be:
        local_mean = ndimage.uniform_filter(image, size=(32, 32))
    :param np.array image: Array with raw two-photon images.
    :param tuple sigmas: List with sigmas (one per axis) to use for the gaussian filter.
        Smaller values result in more local neighborhoods. 15-30 microns should work fine
    """
    local_mean = ndimage.gaussian_filter(image, sigmas)
    local_var = ndimage.gaussian_filter(image ** 2, sigmas) - local_mean ** 2
    local_std = np.sqrt(np.clip(local_var, a_min=0, a_max=None))
    norm = (image - local_mean) / (local_std + 1e-7)

    return norm


def sharpen_2pimage(image, laplace_sigma=0.7, low_percentile=3, high_percentile=99.9):
    """ Apply a laplacian filter, clip pixel range and normalize.
    :param np.array image: Array with raw two-photon images.
    :param float laplace_sigma: Sigma of the gaussian used in the laplace filter.
    :param float low_percentile, high_percentile: Percentiles at which to clip.
    :returns: Array of same shape as input. Sharpened image.
    """
    sharpened = image - ndimage.gaussian_laplace(image, laplace_sigma)
    clipped = np.clip(sharpened, *np.percentile(sharpened, [low_percentile, high_percentile]))
    norm = (clipped - clipped.mean()) / (clipped.max() - clipped.min() + 1e-7)
    return norm

def normalize(data, clip_bounds=None, scale=1, astype=np.float):
    assert type(data)==type(np.array(1))
    if clip_bounds is not None:
        data = np.clip(data,clip_bounds[0], clip_bounds[1]) 
    return (scale*((data - data.min()) / data.ptp())).astype(astype)