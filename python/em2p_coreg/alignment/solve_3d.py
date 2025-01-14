import argschema
from .schemas import SolverSchema
from .data_handler import DataLoader
from .transform import Transform, StagedTransform
import numpy as np
import scipy
import copy

example1 = {
        'data': {
            'landmark_file' : '/src/em2p_coreg/python/em2p_coreg/data/17797_2Pfix_EMmoving_20190414_PA_1018_Deliverable20180415.csv',
            'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'actions': ['invert_opty'],
            'sd_set': {'src': 'em', 'dst': 'opt'}
        },
        #'output_json': '/allen/programs/celltypes/workgroups/em-connectomics/danielk/em_coregistration/tmp_out/transform.json',
        'output_json': '/src/em2p_coreg/python/em2p_coreg/outputs/ex1_transform.json',
        'model': 'TPS',
        'npts': 10,
        'regularization': {
            'translation': 1e-15,
            'linear': 1e-15,
            'other': 1e-15,
            }
}
example2 = {
        'data': {
            'landmark_file' : '/src/em2p_coreg/python/em2p_coreg/data/17797_2Pfix_EMmoving_20190414_PA_1018_Deliverable20180415.csv',
            'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'actions': ['invert_opty', 'em_nm_to_neurog'],
            'sd_set': {'src': 'opt', 'dst': 'em'}
        },
        #'output_json': '/allen/programs/celltypes/workgroups/em-connectomics/danielk/em_coregistration/tmp_out/transform.json',
        'output_json': '/src/em2p_coreg/python/em2p_coreg/outputs/ex2_transform.json',
        'model': 'TPS',
        'npts': 10,
        'regularization': {
            'translation': 1e-10,
            'linear': 1e-10,
            'other': 1e-10,
            }
}

def control_pts_from_bounds(data, npts):
    """create thin plate spline control points
    from the bounds of provided data.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        ndata x 3 Cartesian coordinates of data.
    npts : int
        number of control points per axis. total
        number of control points will be npts^3

    Returns
    -------
    control_pts : :class:`numpy.ndarray`
        npts^3 x 3 Cartesian coordinates of controls.

    """
    x, y, z = [
            np.linspace(data[:, i].min(), data[:, i].max(), npts)
            for i in [0, 1, 2]]
    xt, yt, zt = np.meshgrid(x, y, z)
    control_pts = np.vstack((
        xt.flatten(),
        yt.flatten(),
        zt.flatten())).transpose()
    return control_pts


def solve(A, w, r, dst):
    """regularized linear least squares

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        ndata x nparameter array
    w : :class:`numpy.ndarray`
        ndata x ndata diagonal weight matrix
    r : :class:`numpy.ndarray`
        nparameter x nparameter diagonal
        regularization matrix
    dst : :class:`numpy.ndarray`
        ndata x 3 Cartesian coordinates
        of transform destination.

    Returns
    -------
    x : :class:`numpy.ndarray`
        nparameter x 3 solution

    """
    ATW = A.transpose().dot(w)
    K = ATW.dot(A) + r
    lu, piv = scipy.linalg.lu_factor(K, overwrite_a=True)
    solution = []
    x = np.zeros((A.shape[1], dst.shape[1]))
    for i in range(dst.shape[1]):
        rhs = ATW.dot(dst[:, i])
        x[:, i] = scipy.linalg.lu_solve(
                (lu, piv), rhs)
    return x


def create_regularization(n, d):
    """create diagonal regularization matrix

    Parameters
    ----------
    n : int
        number of parameters per Cartesian axis
    d : dict
        regularization dict from input

    Returns
    -------
    R : :class:`numpy.ndarray`
        n x n diagonal matrix containing regularization
        factors
        
    """
    r = np.ones(n)
    r[0] = d['translation']
    r[1:3] = d['linear']
    r[4:] = d['other']
    i = np.diag_indices(n)
    R = np.eye(n)
    R[i] = r
    return R


def write_src_dst_to_file(fpath, src, dst):
    """csv output of src and dst

    Parameters
    ----------
    fpath : str
        valid path
    src : :class:`numpy.ndarray`
        ndata x 3 source points
    dst : :class:`numpy.ndarray`
        ndata x 3 destination points

    """
    out = np.hstack((src, dst))
    np.savetxt(fpath, out, fmt='%0.8e', delimiter=',')
    print('wrote %s' % fpath)


def list_points_by_res_mag(res, labels, n=np.inf, factor=0.001):
    """print to stdout point labels and residuals

    Parameters
    ----------
    res : :class:`numpy.ndarray`
        ndata x 3 residuals
    labels : list
        ndata length list of point labels
    n : int
        limit to print only highest n residuals
    factor : float
        scales the residuals

    """
    mag = np.linalg.norm(res, axis=1)
    ind = np.argsort(mag)[::-1]
    i = 0
    while (i < n) & (i < ind.size):
        print('%10s, %0.1f' % (labels[ind][i], mag[ind][i] * factor))
        i += 1

def leave_out(data, index):
    if index is None:
        return data, None
    else:
        keep = np.ones(data['labels'].size).astype(bool)
        keep[index] = False
        kdata = {
                'src': data['src'][keep],
                'dst': data['dst'][keep],
                'labels': data['labels'][keep]
                }
        keep = np.invert(keep)
        ldata = {
                'src': data['src'][keep],
                'dst': data['dst'][keep],
                'labels': data['labels'][keep]
                }
        return kdata, ldata

class Solve3D(argschema.ArgSchemaParser):
    """class to solve a 3D coregistration problem"""
    default_schema = SolverSchema

    def run(self, control_pts=None):
        """run the solve

        Parameters
        ----------
        control_pts : :class:`numpy.ndarray`
            user-supplied ncntrl x 3 Cartesian coordinates
            of control points. default None will create
            control points from bounds of input data.

        """
        d = DataLoader(input_data=self.args['data'], args=[])
        d.run()
        self.data = d.data


        if control_pts is None:
            if self.args['npts']:
                control_pts = control_pts_from_bounds(
                        self.data['src'],
                        self.args['npts'])
        self.data, self.left_out = leave_out(self.data, self.args['leave_out_index'])

        self.transform = Transform(
                self.args['model'], control_pts=control_pts)

        # unit weighting per point
        self.wts = np.eye(self.data['src'].shape[0])

        self.A = self.transform.kernel(self.data['src'])

        self.reg = create_regularization(
                self.A.shape[1], self.args['regularization'])

        # solve the system of equations
        self.x = solve(
                self.A,
                self.wts,
                self.reg,
                self.data['dst'])

        # set the parameters for the transform
        self.transform.load_parameters(self.x)

        self.residuals = (
                self.data['dst'] -
                self.transform.transform(self.data['src']))

        print('average residual [dst units]: %0.4f' % (
            np.linalg.norm(self.residuals, axis=1).mean()))

        self.output(self.transform.to_dict(), indent=2)

class StagedSolve2PEM():
    def __init__(self, args):
        self.reg = args['reg']
        self.npts = args['npts']
        self.leave_out_index = args['leave_out_index']
        self.run()

    def run(self):
        # solve just with polynomial
        args_poly = copy.deepcopy(example2)
        args_poly['model'] = 'POLY'
        args_poly['leave_out_index'] = self.leave_out_index
        s_poly = Solve3D(input_data=args_poly, args=[])
        s_poly.run()
        tf_poly = s_poly.transform
        # write the transformed result to file
        # for input to the next stage
        tmp_path = '/src/em2p_coreg/python/em2p_coreg/outputs/poly_results.csv'
        write_src_dst_to_file(
                tmp_path,
                tf_poly.transform(s_poly.data['src']),
                s_poly.data['dst'])
        
        # solve with thin plate spline on top
        args_tps = copy.deepcopy(example2)
        args_tps['model'] = 'TPS'
        args_tps['npts'] = self.npts
        args_tps['data'] = {
                'landmark_file': tmp_path,
                'header': ['polyx', 'polyy', 'polyz', 'emx', 'emy', 'emz'],
                'sd_set': {'src': 'poly', 'dst': 'em'}
                }
        args_tps['regularization']['other'] = self.reg
        s_tps = Solve3D(input_data=args_tps, args=[])
        s_tps.run()
        tf_tps = s_tps.transform
        
        # this object combines the 2 transforms
        # it converts input em units into the final units
        # through both transforms
        self.transform = StagedTransform([tf_poly, tf_tps])

        # let's convince ourselves it works
        total_tfsrc = self.transform.transform(s_poly.data['src'])
        self.residuals = s_poly.data['dst'] - total_tfsrc
        # for 2p -> em this atol means the residuals are within 100nm
        # on any given axis, which is pretty good...
        assert np.all(np.isclose(self.residuals, s_tps.residuals, atol=100)) 

        # how far did the control points move for the thin plate part?
        csrc = tf_tps.control_pts
        cdst = tf_tps.transform(csrc)
        delta = cdst - csrc
        self.avdelta = np.linalg.norm(delta, axis=1).mean() * 0.001
        print('control points moved average of %0.1fum' % (self.avdelta))

        self.leave_out_res = None
        if self.leave_out_index is not None:
            self.leave_out_res = np.linalg.norm(
                    s_poly.left_out['dst'] - self.transform.transform(s_poly.left_out['src']))

class StagedSolveEM2P():
    def __init__(self, args):
        self.reg = args['reg']
        self.npts = args['npts']
        self.leave_out_index = args['leave_out_index']
        self.run()

    def run(self):
        # solve just with polynomial
        args_poly = copy.deepcopy(example1)
        args_poly['model'] = 'POLY'
        args_poly['leave_out_index'] = self.leave_out_index
        s_poly = Solve3D(input_data=args_poly, args=[])
        s_poly.run()
        tf_poly = s_poly.transform
        # write the transformed result to file
        # for input to the next stage
        tmp_path = '/src/em2p_coreg/python/em2p_coreg/outputs/poly_results.csv'
        write_src_dst_to_file(
                tmp_path,
                tf_poly.transform(s_poly.data['src']),
                s_poly.data['dst'])
        
        # solve with thin plate spline on top
        args_tps = copy.deepcopy(example1)
        args_tps['model'] = 'TPS'
        args_tps['npts'] = self.npts
        args_tps['data'] = {
                'landmark_file': tmp_path,
                'header': ['polyx', 'polyy', 'polyz', 'optx', 'opty', 'optz'],
                'sd_set': {'src': 'poly', 'dst': 'opt'}
                }
        args_tps['regularization']['other'] = self.reg
        s_tps = Solve3D(input_data=args_tps, args=[])
        s_tps.run()
        tf_tps = s_tps.transform
        
        # this object combines the 2 transforms
        # it converts input em units into the final units
        # through both transforms
        self.transform = StagedTransform([tf_poly, tf_tps])

        # let's convince ourselves it works
        total_tfsrc = self.transform.transform(s_poly.data['src'])
        self.residuals = s_poly.data['dst'] - total_tfsrc
        # for 2p -> em this atol means the residuals are within 100nm
        # on any given axis, which is pretty good...
        assert np.all(np.isclose(self.residuals, s_tps.residuals, atol=100)) 

        # how far did the control points move for the thin plate part?
        csrc = tf_tps.control_pts
        cdst = tf_tps.transform(csrc)
        delta = cdst - csrc
        self.avdelta = np.linalg.norm(delta, axis=1).mean()
        print('control points moved average of %0.1fum' % (self.avdelta * 1000))

        self.leave_out_res = None
        if self.leave_out_index is not None:
            self.leave_out_res = np.linalg.norm(
                    s_poly.left_out['dst'] - self.transform.transform(s_poly.left_out['src']))

if __name__ == '__main__':
    smod = Solve3D(input_data=example1)
    smod.run()
    smod = Solve3D(input_data=example2)
    smod.run()

