import datajoint as dj
from .utils import *

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

# This functions are slightly modified from pipeline.stack
def get_grid(self, type='affine', desired_res=1):
    """ Get registered grid for this registration. 
    
    type: 'rigid', affine', 'nonrigid'
    desired_res: In um/px.
    """
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
        raise dj.PipelineException('Unrecognized registration.')

    return pred_grid.numpy()


def plot_grids(self, desired_res=5):
    """ Plot the grids for this different registrations as 3-d surfaces."""
    # Get grids at desired resoultion
    rig_grid = get_grid(self, 'rigid', desired_res)
    affine_grid = get_grid(self, 'affine', desired_res)
    nonrigid_grid = get_grid(self, 'nonrigid', desired_res)

    # Plot surfaces
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

