import numpy as np
import datajoint as dj
import pandas as pd
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import re
from PIL import Image
import json
import urllib
from scipy import ndimage

plat = dj.create_virtual_module('pipeline_platinum','pipeline_platinum')
meso = dj.create_virtual_module('pipeline_meso', 'pipeline_meso')
reso = dj.create_virtual_module('pipeline_reso', 'pipeline_reso')
tune = dj.create_virtual_module('pipeline_tune', 'pipeline_tune')
stack = dj.create_virtual_module('pipeline_stack', 'pipeline_stack')
anatomy = dj.create_virtual_module('pipeline_anatomy', 'pipeline_anatomy')
radtune = dj.create_virtual_module('pipeline_radtune','pipeline_radtune')
spattune = dj.create_virtual_module('pipeline_spattune','pipeline_spattune')

def get_munit_ids(scan_relation, stack_key, brain_area, tuning=None, oracle_threshold=0.2, von_p_threshold=0.05, snr_threshold=1.3, n_scan_threshold=1, limit=10, as_list_dict=True):
    """
    Queries the database to find cells from the list of specified scans,
    uniquely registered into the specified structural stack and listed under the specified
    brain area. User can specify whether they want cells that are radially tuned, spatially 
    tuned, neither or both with specified quality thresholds including oracle and minimum number of scans.
    
    :param scan_relation: Datajoint table relation containing the desired scans
    :param stack_key: Key to restrict to the desired structural stack
    :param brain_area: Desired brain area. Relevant areas: "V1", "LM", "AL", "RL".
    :param tuning: Specifies whether cells must be radially or spatially tuned, both or neither. 
    :param oracle_threshold: minimum oracle score
    :param von_p_threshold: minimum threshold of significance for tuning
    :param snr_threshold: minimum threshold for spatial receptive field significance
    :param n_scan_threshold: minimum number of functional scans per munit id 
    :param limit: the maximum number of cells to return
    :param as_list_dict: output format. If TRUE returns a list of dictionaries of munit id's otherwise just a list
    
    :return: returns munit id's (unique cell within the structural stack) according to the specified criteria. 
    
             If tuning = 'rad', returns cells that are radially tuned
             If tuning = 'spat', returns cells with spatial receptive fields
             If tuning = 'both', returns cells with both radial tuning and spatial receptive fields
             If tuning = 'none', removes specification for both radial and spatial tuning
             
             If as_list_dict = True, returns a list of dictionaries of munit id's to be used for restricting 
             downstream datajoint tables, else it returns a list of munit id's. 
    """
    
    oracle_thre = f'pearson >= {oracle_threshold}' # manually set threshold for oracle
    von_p_thre = f'von_p_value < {von_p_threshold}' 
    snr_thre = f'snr > {snr_threshold}'
    rad = (radtune.VonFit.Unit & 'animal_id = 17797' & 'vonfit_method = 3' & 'ori_type = "dir"' & von_p_thre).proj('von_p_value', scan_session = 'session')
    spat = (spattune.STA.Loc() & 'animal_id = 17797' & 'stimgroup_id = 1' & 'center_x BETWEEN 5 AND 155 and center_y BETWEEN 5 AND 85' & snr_thre).proj('snr', scan_session = 'session')
   
    tot_munits = stack.StackSet.Match() & (stack.CorrectedStack & stack_key).proj(stack_session = 'session') & scan_relation.proj(scan_session = 'session')
    munits = tot_munits * anatomy.AreaMembership.proj('brain_area', scan_session = 'session') 
    munit_oracle = (tune.MovieOracle.Total & oracle_thre).proj('trials', 'pearson', scan_session = 'session') * munits
    
    if tuning == 'rad':
        munit_oracle_ext = munit_oracle * rad
        good_oracle = dj.U('munit_id', 'brain_area').aggr(munit_oracle_ext, n_scan = 'count(DISTINCT(scan_idx))', avg_pearson = 'avg(pearson)', avg_von_p = 'avg(von_p_value)')
        df = pd.DataFrame(good_oracle.fetch())
        df_sorted = (df.sort_values(['n_scan', 'avg_pearson', 'avg_von_p'], ascending = [0, 0, 1])).reset_index(drop=True)
        
    elif tuning == 'spat':
        munit_oracle_ext = munit_oracle * spat
        good_oracle = dj.U('munit_id', 'brain_area').aggr(munit_oracle_ext, n_scan = 'count(DISTINCT(scan_idx))', avg_pearson = 'avg(pearson)', avg_snr = 'avg(snr)')
        df = pd.DataFrame(good_oracle.fetch())
        df_sorted = (df.sort_values(['n_scan', 'avg_pearson', 'avg_snr'], ascending = [0, 0, 0])).reset_index(drop=True)
    
    elif tuning == 'both':
        munit_oracle_ext = munit_oracle * rad * spat
        good_oracle = dj.U('munit_id', 'brain_area').aggr(munit_oracle_ext, n_scan = 'count(DISTINCT(scan_idx))', avg_pearson = 'avg(pearson)', avg_von_p = 'avg(von_p_value)', avg_snr = 'avg(snr)')
        df = pd.DataFrame(good_oracle.fetch())
        df_sorted = (df.sort_values(['n_scan', 'avg_pearson', 'avg_von_p', 'avg_snr'], ascending = [0, 0, 0, 0])).reset_index(drop=True)
    
    else:
        good_oracle = dj.U('munit_id', 'brain_area').aggr(munit_oracle, n_scan = 'count(DISTINCT(scan_idx))', avg_pearson = 'avg(pearson)')
        df = pd.DataFrame(good_oracle.fetch())
        df_sorted = (df.sort_values(['n_scan', 'avg_pearson'], ascending = [0, 0])).reset_index(drop=True)
        
    df_by_area = df_sorted[df_sorted['brain_area'] == brain_area]
    df_by_area_scan = df_by_area[df_by_area['n_scan'] >= n_scan_threshold][:limit]['munit_id']
    
    if as_list_dict:
        out = []
        for munit in df_by_area_scan.values:
            out.append({'munit_id':munit})
        return out
    
    else:
        return df_by_area_scan.values    

def get_munit_coords(munit_list_dict, stack_key, ref_frame='motor', ng_scaling=[250,250,1]):
    """
    function that takes in a list of munits and returns coordinates according to the specified reference frame
    
    :param munit_list_dict: list of dictionaries of munits id's to restrict the stack datajoint table relation
    :param stack_key: the key to restrict the stack datajoint table relation to the specified structural stack
    :param ref_frame: the choice of reference frame for the ouput coordinates. Options are 'motor', 'numpy', 'ng'. 
    :param ng_scaling: the scaling factor to apply to 'numpy' coordinate output for use in neuroglancer.
    
    :return: If ref_frame is 'motor' returns the microscope motor coordinates in units of microns.
             If ref_frame is 'numpy' returns the coordinates with (x=0,y=0,z=0) in the top, rear, left corner of stack.
             If ref_frame is 'ng' returns the numpy coordinates with a scaling factor applied for neuroglancer
    
    """
    
    rel = np.stack((stack.StackSet.Unit & stack_key & munit_list_dict).fetch('munit_id', 'munit_x', 'munit_y', 'munit_z')).T

    if len(munit_list_dict) > 1:
        rel_ordered = rel[np.array([np.where(rel==munit_list_dict[i]['munit_id'])[0][0] for i in range(len(munit_list_dict))]),...] # reorders output motor coords according to input order
        motor_coords = rel_ordered.copy()[:,1:]
    else:
        motor_coords = rel.copy()[:,1:]

    if ref_frame == 'motor':
        return motor_coords
    
    center_xyz_um = np.array([*(stack.CorrectedStack() & stack_key).fetch1('x', 'y', 'z')])
    lengths_xyz_um = np.array([*(stack.CorrectedStack() & stack_key).fetch1('um_width', 'um_height', 'um_depth')])
    np_coords = np.array(motor_coords) - np.array(center_xyz_um) + np.array(lengths_xyz_um) / 2
    
    if ref_frame == 'numpy':
         return np_coords
    
    if ref_frame == 'ng':
        return np_coords*ng_scaling

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
    import torch.nn.functional as F

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

# This functions are slightly modified from pipeline.stack
def get_grid(self, type='affine', desired_res=1):
    """ Get registered grid for this registration. 
    
    type: 'rigid', affine', 'nonrigid'
    desired_res: In um/px.
    """
    import torch

    # Get field
    field_key = self.proj(session='scan_session')
    field_dims = (reso.ScanInfo & field_key or meso.ScanInfo.Field &
                  field_key).fetch1('um_height', 'um_width')

    # Create grid at desired resolution
    grid = create_grid(field_dims, desired_res=desired_res)  # h x w x 2
    grid = torch.tensor(grid, dtype=torch.float32) # for torch v0.4.1: torch.as_tensor(grid, dtype=torch.float32) 

    # Apply required transform
    if type == 'rigid':
        params = (stack.Registration.Rigid & self).fetch1('reg_x', 'reg_y', 'reg_z')
        delta_x, delta_y, delta_z = params
        linear = torch.eye(3)[:, :2]
        translation = torch.tensor([delta_x, delta_y, delta_z])

        pred_grid = affine_product(grid, linear, translation)
    elif type == 'affine':
        params = (stack.Registration.Affine & self).fetch1('a11', 'a21', 'a31', 'a12',
                                                     'a22', 'a32', 'reg_x', 'reg_y',
                                                     'reg_z')
        a11, a21, a31, a12, a22, a32, delta_x, delta_y, delta_z = params
        linear = torch.tensor([[a11, a12], [a21, a22], [a31, a32]])
        translation = torch.tensor([delta_x, delta_y, delta_z])

        pred_grid = affine_product(grid, linear, translation)
    elif type == 'nonrigid':
        params = (stack.Registration.NonRigid & self).fetch1('a11', 'a21', 'a31', 'a12',
                                                       'a22', 'a32', 'reg_x', 'reg_y',
                                                       'reg_z', 'landmarks',
                                                       'deformations')
        rbf_radius = (stack.Registration.Params & self).fetch1('rbf_radius')
        a11, a21, a31, a12, a22, a32, delta_x, delta_y, delta_z, landmarks, deformations = params
        linear = torch.tensor([[a11, a12], [a21, a22], [a31, a32]])
        translation = torch.tensor([delta_x, delta_y, delta_z])
        landmarks = torch.from_numpy(landmarks)
        deformations = torch.from_numpy(deformations)

        affine_grid = affine_product(grid, linear, translation)
        grid_distances = torch.norm(grid.unsqueeze(-2) - landmarks, dim=-1)
        grid_scores = torch.exp(-(grid_distances * (1 / rbf_radius)) ** 2)
        warping_field = torch.einsum('whl,lt->wht', (grid_scores, deformations))

        pred_grid = affine_grid + warping_field
    else:
        raise PipelineException('Unrecognized registration.')

    return pred_grid.numpy()


def plot_grids(self, desired_res=5):
    """ Plot the grids for this different registrations as 3-d surfaces."""
    # Get grids at desired resoultion
    rig_grid = get_grid(self, 'rigid', desired_res)
    affine_grid = get_grid(self, 'affine', desired_res)
    nonrigid_grid = get_grid(self, 'nonrigid', desired_res)

    # Plot surfaces
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
    ax = fig.gca(projection='3d')
    ax.plot_surface(rig_grid[..., 0], rig_grid[..., 1], rig_grid[..., 2], alpha=0.5)
    ax.plot_surface(affine_grid[..., 0], affine_grid[..., 1], affine_grid[..., 2],
                    alpha=0.5)
    ax.plot_surface(nonrigid_grid[..., 0], nonrigid_grid[..., 1],
                    nonrigid_grid[..., 2], alpha=0.5)
#     ax.set_aspect('equal')
    ax.invert_zaxis()

    return fig

def get_fields(chosen_cell, scans, stack_key):
    """
    function that takes in an munit_id, a set of scans, and a stack, and returns the field key and a set of summary images for each scan_session and scan_idx the munit appears in:
    The summary images are as follows:
    1) scan summary image which will be one of: average, correlation, l6norm, hybrid image depending on specification in argument 'functional_image'
    2) the relevant stack field after registering the imaging field inside the stack
    3) the relevant 3D segmentation image after registering the imaging field
    
    :param chosen_cell: dictionary of munit id formatted as such: {'munit_id':00000}
    :param scans: the key to restrict the stack datajoint table relation to the specified functional scans
    :param stack_key: the relevant stack to restrict with

    :return: List of field keys For each scan_session and scan_idx returns:
    """
    
    field_munit_relation_table = meso.ScanSet.Unit * (stack.StackSet.Match & stack_key & chosen_cell).proj('munit_id', session='scan_session') & scans
    field_munit_relation_table

    # choose an example cell and get unit_id's (unique id's per scan)
    field_munit_relation_keys = (dj.U('animal_id', 'stack_session', 'stack_idx', 'segmentation_method', 'session','scan_idx','field', 'munit_id') & field_munit_relation_table).fetch('KEY')

    return field_munit_relation_keys

def plot_fields(field_key, EM_grid, EM_center, EM_data=None, functional_image='average', cell_stack=None, vessel_stack=None, figsize=(10,10), dpi=100, locate_cell=True, share=True, enhance=True):
    """
    function that takes in a field key, and optional 2P and EM stacks, and returns a plot of the imaging field and slices of the field from the stacks.
    The images are as follows:
    1) functional scan image which will be one of: average, correlation, l6norm, hybrid image depending on specification in argument 'functional_image'
    2) the relevant stack field after registering the imaging field inside the stack
    3) the relevant 3D segmentation image after registering the imaging field
    
    :param chosen_cell: dictionary of munit id formatted as such: {'munit_id':00000}
    :param scans: the key to restrict the stack datajoint table relation to the specified functional scans
    :param stack_key: the relevant stack to restrict with
    :param functional_image: specifies what kind of summary image to use for the Scan summary image. Can be a single image type or a mathematical combination of multiple images. 
                             Individual image options are 'average', 'correlation', 'l6norm', 'oracle'.
                             Example mathematical combinations:
                                1) functional_image='average*correlation'
                                2) functional_image='oracle**2'
                                3) functional_image='(average*correlation)-l6norm'
    :param stack_data: optional, can include the stack desired to sample from using sample_grid(). Default is None, which provides the stack image in stack.Registration.Affine()
    :param figsize: specify the size of the figure
    :param dpi: desired dpi of summary images
    :param: if plot is True, will return summary images

    :return: For each scan_session and scan_idx returns:
                Field key
                if plot is True:
                    Scan summary image 
                    Stack summary image 
                    3D segmentation image

    """
    ## Get grid
    grid = get_grid(stack.Registration() & field_key & {'scan_session':field_key['session']}, desired_res=1)
    distance_mask = np.sqrt(((grid - get_munit_coords({'munit_id': field_key['munit_id']}, {'stack_idx':field_key['stack_idx']}, ref_frame='motor'))**2).sum(-1)) # for localizing cell
    stack_x, stack_y, stack_z = (stack.CorrectedStack & {'animal_id':field_key['animal_id'], 'stack_idx':field_key['stack_idx']}).fetch1('x', 'y', 'z')

    ## Functional image
    dict_images = {'average': '(meso.SummaryImages.Average() & field_key).fetch1("average_image")',
                    'correlation': '(meso.SummaryImages.Correlation() & field_key).fetch1("correlation_image")',
                    'l6norm': '(meso.SummaryImages.L6Norm() & field_key).fetch1("l6norm_image")',
                    'oracle': '(tune.OracleMap() & field_key).fetch1("oracle_map")'}
        
    
    temp = functional_image
    out = []
    for image_type in ('average', 'correlation', 'l6norm', 'oracle'):
        match = re.findall(image_type, temp)
        out.append(match)
    filtered_match = list(filter(None, out))
    for match in filtered_match:
        temp = temp.replace(match[0], dict_images[match[0]])
            
    # resize the scan field image to match the dimensions of the grid
    scan_field = eval(temp)
    if enhance:
        scan_field = sharpen_2pimage(lcn(scan_field, 2.5))
        enhance_string = 'enhanced'
    else:
        enhance_string = None
    
    scan_field_reshape = resize(scan_field, (meso.ScanInfo.Field() & field_key).fetch1('um_height', 'um_width'), desired_res=1)
    
    ## Stack image
    if cell_stack is not None:
        recentered_grid = grid - np.array([stack_x, stack_y, stack_z]) # move center of stack to be (0, 0, 0)
        stack_field = sample_grid(cell_stack, recentered_grid).numpy()
        stack_name = 'the provided stack'
    else:
        stack_field = (stack.Registration.Affine() & field_key & {'scan_session':field_key['session']}).fetch1('reg_field')
        stack_name = 'Registration.Affine() stack'

    ## Vessel image
    vessel_field = sample_grid(vessel_stack, recentered_grid).numpy()

    ## Segmentation image
    segm_field = (stack.FieldSegmentation() & field_key & {'scan_session':field_key['session']}).fetch1('segm_field')
    
    ## EM Image
    temp_grid = (EM_grid - EM_center)/ 1000
    EM_field = sample_grid(EM_data, temp_grid).numpy()

    if share:
        fig, axes = plt.subplots(1, 5, figsize=figsize, sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(1, 5, figsize=figsize)
    axes[0].imshow(np.array(scan_field_reshape))
    axes[1].imshow(stack_field)
    axes[2].imshow(vessel_field)
    axes[3].imshow(segm_field)
    axes[4].imshow(-EM_field, cmap='gray')

    if locate_cell:
        for ax in axes:
            ax.imshow((distance_mask<20) & (distance_mask>15), cmap='gray', alpha=0.2)

    axes[0].set_title(f'Scan field: {enhance_string} \n {functional_image}')
    axes[1].set_title(f'Field taken from \n {stack_name}')
    axes[2].set_title(f'Field taken from \n vessels')
    axes[3].set_title('3-d segmentation')
    axes[4].set_title('Field taken from \n resized EM')

    fig.suptitle(f'scan_session: {field_key["session"]}, scan_idx: {field_key["scan_idx"]}, field: {field_key["field"]}, units: $\mu$m \n munit_id: {field_key["munit_id"]}', y=0.75, fontsize=18)
    fig.set_dpi(dpi)

def update_link_voxels(provided_link, voxels, version='none'):
    """
    Function that takes in a neuroglancer link, parses the link to identify the region containing the voxel coordinates, then replaces with the desired voxel coordinates
        
    :param provided_link: the neuroglancer link within which you wish to replace the voxel coordinates
    :param EM_voxels: a 1 or 2D numpy array of voxel coordinates
    :param version: specify whether the provided link originates from the EM neuroglancer or 2P neuroglancer
    
    :return: if version=='EM' returns updated EM neuroglancer link with provided voxel coordinates
             if version=='2P' returns 2P neuroglancer link with provided voxel coordinates
    """
    
    # generate new snippet with new coordinates
    if version == 'EM':
        new_coordinates_template = '%22:%5BXXX%2CYYY%2CZZZ%5D%7D%2C%22'
    if version == '2P':
        new_coordinates_template = '%22:%5BXXX%2CYYY%2CZZZ%5D%7D%7D%2C%22'
    if version == 'none':
        return print('argument version must be "2P" or "EM"')
    x,y,z = voxels.squeeze()
    new_coordinates_snippet = new_coordinates_template.replace('XXX',str(x))
    new_coordinates_snippet = new_coordinates_snippet.replace('YYY',str(y))
    new_coordinates_snippet = new_coordinates_snippet.replace('ZZZ',str(z))
    
    # find relevant section in provided link and create snippet
    matches = re.search('voxelCoordinates', provided_link)
    snippet = provided_link[matches.start()+len(matches.group()):matches.start()+150]
    next_keyword = re.findall('[a-z]+',snippet)[0]
    next_keyword_match = re.search(next_keyword,provided_link)
    snippet_to_replace = provided_link[matches.start()+len(matches.group()):next_keyword_match.start()]
    updated_link = provided_link.replace(snippet_to_replace, new_coordinates_snippet)
    
    return updated_link

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