import numpy as np
import datajoint as dj
import pandas as pd
import torch

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
    function that takes in a list of munits
    
    :param munit_list_dict: list of dictionaries of munits id's to restrict the stack datajoint table relation
    :param stack_key: the key to restrict the stack datajoint table relation to the specified structural stack
    :param ref_frame: the choice of reference frame for the ouput coordinates. Options are 'motor', 'numpy', 'ng'. 
    :param ng_scaling: the scaling factor to apply to 'numpy' coordinate output for use in neuroglancer.
    
    :return: If ref_frame is 'motor' returns the microscope motor coordinates in units of microns.
             If ref_frame is 'numpy' returns the coordinates with (x=0,y=0,z=0) in the top, rear, left corner of stack.
             If ref_frame is 'ng' returns the numpy coordinates with a scaling factor applied for neuroglancer
    
    """
    
    motor_coords = np.stack((stack.StackSet.Unit & stack_key & munit_list_dict).fetch('munit_x', 'munit_y', 'munit_z')).T
    
    if ref_frame == 'motor':
        return motor_coords
    
    center_xyz_um = np.array([*(stack.CorrectedStack() & stack_key).fetch1('x', 'y', 'z')])
    lengths_xyz_um = np.array([*(stack.CorrectedStack() & stack_key).fetch1('um_width', 'um_height', 'um_depth')])
    
    if ref_frame == 'numpy':
        return np.array(motor_coords) - np.array(center_xyz_um) + np.array(lengths_xyz_um) / 2 
    
    if ref_frame == 'ng':
        return np.round((np.array(motor_coords) - np.array(center_xyz_um) + np.array(lengths_xyz_um) / 2)*ng_scaling)

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