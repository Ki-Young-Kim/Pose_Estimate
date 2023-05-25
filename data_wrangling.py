from lbs import lbs
from util import get_model_path, glg, get_device, load_smpl_model, dtype_torch, dtype_np

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import copy
import functools
import time


SMPL_JOINT_LABELS = {
        0: "Pelvis",
        1: "L_Hip",
        2: "R_Hip",
        3: "Spine1",
        4: "L_Knee",
        5: "R_Knee",
        6: "Spine2",
        7: "L_Ankle",
        8: "R_Ankle",
        9: "Spine3",
        10: "L_Foot",
        11: "R_Foot",
        12: "Neck",
        13: "L_Collar",
        14: "R_Collar",
        15: "Head",
        16: "L_Shoulder",
        17: "R_Shoulder",
        18: "L_Elbow",
        19: "R_Elbow",
        20: "L_Wrist",
        21: "R_Wrist",
        22: "L_Hand",
        23: "R_Hand",
}

SMPL_LEG_JOINTS = (4, 5, 6, 7)
SMPL_LOWER_BODY_JOINTS = SMPL_LEG_JOINTS + (10, 11)
SMPL_UPPER_BODY_JOINTS = tuple(set(range(24)).difference((1, 2) + SMPL_LOWER_BODY_JOINTS))


def totorch(ndarray, do_copy=False, device=None):
    if do_copy:
        ndarray = copy.deepcopy(ndarray)
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    else:
        return torch.from_numpy(ndarray).to(device or get_device())


def set_get_device_func(func):
    global get_device
    get_device = func

def get_k(modeltype):
    try:
        return {'smpl': 23, 'smplh': 51, 'smplx': 53}[modeltype]
    except KeyError:
        raise ValueError


def model_type_from_njoints(njoints):
    try:
        return {24: 'smpl', 52: 'smplh', 54: 'smplx'}[njoints]
    except KeyError:
        raise ValueError


def correct_amass_rotation(poses, tp='axisangles'):
    r = Rotation.from_euler('zx', (-90, 270), degrees=True)
    if tp == 'axisangles':
        poses[..., 0, :] = (r * Rotation.from_rotvec(poses[..., 0, :])).as_rotvec()
    elif tp == 'translations':
        transls_flatb = poses.reshape((-1, 3))
        transls_flatb_rotated = r.apply(transls_flatb)
        poses[:] = transls_flatb_rotated.reshape(poses.shape)
    else:
        raise ValueError

def rotate_y_smpl(poses, deg, tp='axisangles'):
    r = Rotation.from_euler('y', deg, degrees=True)
    if tp == 'axisangles':
        poses[..., 0, :] = (r * Rotation.from_rotvec(poses[..., 0, :])).as_rotvec()
    elif tp == 'translations':
        transls_flatb = poses.reshape((-1, 3))
        transls_flatb_rotated = r.apply(transls_flatb)
        poses[:] = transls_flatb_rotated.reshape(poses.shape)

def adjust_framerate(og_framerate, target_framerate, seq):
    if target_framerate > og_framerate:
        raise NotImplementedError("Upsampling is not supported")

    downsample_factor = round(og_framerate / target_framerate)
    return seq[::downsample_factor], og_framerate / downsample_factor


def compute_joints(config, pose=None, transl=None, beta=None, beta_size=None,
        rot_mats=None, device=None, model=None, model_path=None, model_type=None, gender=None,
        njoints_cutoff=-1, joints_rest=None, pose_reshape_ok=True,
        relative=False, return_dict=False, return_torch_objects=False):
    """
    If model is not provided, it will be loaded from model_path,
    which will be determined by model_type and gender if it is not provided.
    pose must have shape (njoints x 3) or (nbatch x njoints x 3)

    # For batch only
    If joints_rest is provided, joint positions in rest pose
    will not have to be computed from model['v_template'] and model['J_regressor'],
    speeding up computation.
    """
    if rot_mats is None:
        if len(pose.shape) == 2:
            is_batch = False
        elif len(pose.shape) == 3:
            is_batch = True
        else:
            raise ValueError
    else:
        if len(rot_mats.shape) == 3:
            is_batch = False
        elif len(rot_mats.shape) == 4:
            is_batch = True
        else:
            raise ValueError

    if njoints_cutoff > 0:
        if njoints_cutoff > (pose if pose is not None else rot_mats).shape[1]:
            raise ValueError
        njoints = njoints_cutoff
        pose = pose[:, :njoints]
    else:
        njoints = (pose if pose is not None else rot_mats).shape[1]

    beta_size = beta_size or config['model_config']['beta_size']

    if model is None and model_path is None:
        if njoints >= 1 + get_k('smpl'):
            model_type = model_type_from_njoints(njoints)
        else:
            model_type = 'smpl'
        model_path = get_model_path(config, model_type, gender or 'male')

    if not is_batch:
        if pose is None:
            raise NotImplementedError("SMPL pose is required for non-batch joint computation")

        glg().debug("Computing joints via SMPLModel...")

        from smpl_np import SMPLModel
        if model is None:
            model = load_smpl_model(model_path, as_class=True, k=njoints-1)
        if beta is None:
            beta = np.zeros(beta_size)
        model.set_params(pose=pose, beta=beta, trans=transl, skip_verts_update=True)
        joints = model.pose_joints()
        if not return_dict:
            return joints
        else:
            return {
                'joints': joints,
                'model': model
            }
    else:  # batch
        if model is None:
            model = load_smpl_model(model_path)

        nbatch = (pose if pose is not None else rot_mats).shape[0]
        if beta is None:
            beta = np.zeros((nbatch, beta_size), dtype=dtype_np)
                    #dtype=np.float64 if 
                    #(pose if pose is not None else rot_mats).dtype in [torch.float64, np.float64]
                    #else np.float32)

        shapedirs = np.array(model['shapedirs'][:, :, :beta_size])
        posedirs = model['posedirs'].reshape([model['posedirs'].shape[0] * 3, -1]).T
        parents = model['kintree_table'].astype(np.int32)[0]
        weights = model['weights']
        if joints_rest is not None:
            v_template = None
            J_regressor = None
        else:
            v_template = model['v_template'][np.newaxis]
            J_regressor = model['J_regressor']
            if not isinstance(J_regressor, np.ndarray):
                J_regressor = J_regressor.todense()

        parents = parents[:njoints]
        weights = weights[:, :njoints]
        if J_regressor is not None:
            J_regressor = J_regressor[:njoints]

        glg().debug("Computing joints via lbs(...)...")
        tt = functools.partial(totorch, device=get_device())
        _verts, J, J_transformed, rot_mats, rot_mats_g = \
                lbs(
            betas=tt(beta), pose=tt(pose if rot_mats is None else rot_mats),
            pose2rot=rot_mats is None,
            shapedirs=tt(shapedirs), posedirs=tt(posedirs),
            parents=tt(parents).long(),
            lbs_weights=tt(weights),
            
            v_template=tt(v_template) if v_template is not None else None,
            J_regressor=tt(J_regressor) if J_regressor is not None else None,
            joints_rest=tt(joints_rest) if joints_rest is not None else None,
            pose_reshape_ok=pose_reshape_ok, compute_joints_only=True)

        if return_torch_objects:
            joints = J_transformed
        else:
            joints = J_transformed.detach().cpu().numpy()
            rot_mats = rot_mats.detach().cpu().numpy()
            rot_mats_g = rot_mats_g.detach().cpu().numpy()

        if relative:
            parents1 = parents.copy()
            parents1[parents1 < 0] = 0
            par_joints = joints[..., parents1, :]
            joints = joints - par_joints

        if transl is not None:
            raise NotImplementedError("TODO")

        if not return_dict:
            return joints
        else:
            return {
                'joints': joints,
                'rot_mats': rot_mats,
                'rot_mats_g': rot_mats_g,
                'joints_rest': J,
                'model': model
            }


def normalise(v):
    """
    Normalise vector(s) in last axis
    """
    if isinstance(v, torch.Tensor):
        t = torch.div(v, torch.linalg.vector_norm(v, dim=-1)[..., None])
        t[t != t] = 0  # remove NaN
        return t
    else:
        return v / np.expand_dims(np.linalg.norm(v, axis=-1), axis=-1)


def rot_6d_to_mat(r6d, recompute_first_two_cols=True):
    """
    r6d expected to have shape (... x 3 x 2)
    """
    if isinstance(r6d, torch.Tensor):
        rotmats = torch.zeros(tuple(r6d.shape[:-2]) + (3, 3),
                dtype=dtype_torch).to(r6d.device)
        if recompute_first_two_cols:
            rotmats[..., 0] = normalise(r6d[..., 0].clone())
            r6d1 = r6d[..., 1].clone()
            rm0 = rotmats[..., 0].clone()
            rotmats[..., 1] = normalise(
                    r6d1 - (torch.sum(rm0 * r6d1, dim=-1)[..., None] * rm0))
            #rotmats[..., 1] = normalise(
            #        r6d[...,1].clone() - (torch.sum(rotmats[...,0].clone() * r6d[...,1].clone(), dim=-1)[..., None] * rotmats[...,0].clone()))
        else:
            rotmats[..., [0, 1]] = r6d[..., [0, 1]]
        rotmats_d = rotmats.detach()
        rotmats[..., 2] = torch.linalg.cross(rotmats_d[..., 0], rotmats_d[..., 1], dim=-1)
    else:
        rotmats = np.zeros(tuple(r6d.shape[:-2]) + (3, 3), dtype=r6d.dtype)
        if recompute_first_two_cols:
            rotmats[..., 0] = normalise(r6d[..., 0])
            rotmats[..., 1] = normalise(
                r6d[..., 1] - 
                np.expand_dims(np.sum(rotmats[..., 0] * r6d[..., 1], axis=-1), axis=-1) * rotmats[..., 0])
        else:
            rotmats[..., [0, 1]] = r6d[..., [0, 1]]
        rotmats[..., 2] = np.cross(rotmats[..., 0], rotmats[..., 1])
    return rotmats


def rot_mat_to_vec(rotmats):
    og_shape = None
    if len(rotmats.shape) > 3:
        og_shape = rotmats.shape
        rotmats = rotmats.reshape((-1, 3, 3))

    rvec = Rotation.from_matrix(rotmats).as_rotvec()
    if og_shape is not None:
        rvec = rvec.reshape(tuple(og_shape[:-2]) + (3,))
    return rvec


def joints_to_vel(data, fps=1, initial='skip'):
    """
        Accepts inputs of form (nb x)* nf x nj x 3
    """
    if isinstance(data, torch.Tensor):
        if initial == 'skip':
            return fps * torch.diff(data, 1, dim=-3)
        elif initial == 'rest':
            return fps * torch.diff(data, 1, dim=-3, prepend=data[..., [0], :, :])
    else:
        if initial == 'skip':
            return fps * np.diff(data, 1, axis=-3)
        elif initial == 'rest':
            return fps * np.diff(data, 1, axis=-3, prepend=data[..., [0], :, :])
    raise ValueError


def vel_to_acc(data, fps=1):
    return joints_to_vel(data, fps)


def rot_mats_to_davel(data, fps):
    if isinstance(data, torch.Tensor):
        return fps * (data - torch.roll(data, 1, dims=-4))[..., 1:, :, :, :]
    else:
        return fps * (data - axis.roll(data, 1, axis=-4))[..., 1:, :, :, :]


def estimate_velocity_humor(data_seq, h):
    '''
    Modified davrempe/humor/humor/scripts/process_amass_data.py

    Given some data sequence of T timesteps in the shape (T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    - h : step size
    '''
    data_tp1 = data_seq[2:]
    data_tm1 = data_seq[0:-2]
    data_vel_seq = (data_tp1 - data_tm1) / (2*h)
    return data_vel_seq


def estimate_angular_velocity_humor(rot_seq, h):
    '''
    Modified davrempe/humor/humor/scripts/process_amass_data.py

    Given a sequence of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (T, ..., 3, 3)
    '''
    istorch = isinstance(rotmats, torch.Tensor)

    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_velocity_humor(rot_seq, h)
    R = rot_seq[1:-1]
    RT = torch.transpose(R, -1, -2) if istorch else np.swapaxes(R, -1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = dRdt @ RT

    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)

    return w


def angular_velocity_from_joints_rot_mats(rotmats, fps=1, avel_type='delta', initial='skip'):
    """
        Accepts inputs of form (nb x)* nf x nj x 3 x 3 
    """
    if isinstance(rotmats, torch.Tensor):
        if avel_type == 'delta':
            rm_s = torch.roll(rotmats, 1, dims=-4)
            # Initial rest
            rm_s[..., 0, :, :, :] = rotmats[..., 0, :, :, :]
            avel = rotmats @ torch.transpose(rm_s, -1, -2)
        elif avel_type == 'humor':
            avel = torch.swapaxes(
              estimate_angular_velocity_humor(torch.swapaxes(rotmats, 0, -4), 1 / fps), -4, 0)
        else:
            raise ValueError
    else:
        if avel_type == 'delta':
            rm_s = np.roll(rotmats, 1, axis=-4)
            # Initial rest
            rm_s[..., 0, :, :, :] = rotmats[..., 0, :, :, :]
            avel = rotmats @ np.moveaxis(rm_s, -1, -2)
        elif avel_type  == 'humor':
            avel = np.swapaxes(
              estimate_angular_velocity_humor(np.swapaxes(rotmats, 0, -4), 1 / fps), -4, 0)
        else:
            raise ValueError

    if initial == 'skip':
        return fps * avel[..., 1:, :, :, :]
    elif initial == 'rest':
        return fps * avel
    else:
        raise ValueError


def get_converter(config, src_type, dest_type, return_intermediates=False, model=None, **kw):
    """
        data expected to have shape (nb*) x nf * nj * ...

    """

    if kw.get('normalise_velocities', False):
        fps = 1
    else:
        fps = config['model_config']['target_framerate']
    rf2c = kw.get('recompute_first_two_cols', True)

    if model is None:
        print("get_converter - no SMPL model provided; loading one...")

        model_path = get_model_path(config, 'smpl', 'male')
        model = load_smpl_model(model_path, as_class=kw.get('load_smpl_model_as_class', False))

    lg = glg()

    def _convert(data, st, dt, rdepth=0, intm=None):
        def _cerr():
            raise NotImplementedError("Unknown conversion {} -> {}".format(st, dt))

        if rdepth > 10:
            raise RuntimeError("Invalid conversion from {} -> {} (recursion_depth={})"
                    .format(src_type, dest_type, recursion_depth))
        if intm is None:
            intm = {}

        out = None

        lg.debug("get_converter-_convert %s (%s) ->%s (d=%d, r2fc=%s, kw=%s)",
                st, data.shape, dt, rdepth, rf2c, kw)
        t0 = time.time()

        if st == 'rot_6d':
            rmats = rot_6d_to_mat(data, recompute_first_two_cols=rf2c)
            if dt == 'rot_mats':
                out = rmats
            else:
                intm['rot_mats'] = rmats
                out, intm = _convert(rmats, 'rot_mats', dt, rdepth=rdepth+1, intm=intm)
        elif st == 'rot_mats':
            if dt in ('joints', 'vel', 'acc') or dt in ('joints_rel', 'vel_rel'):
                isrel = dt.endswith("_rel")
                joints_lbl = "joints" + ("_rel" if isrel else "")
                t1 = time.time()
                if len(data.shape) == 4: # nf x nj x 3 x 3
                    batch_n_fr = None
                    reshp = None
                elif len(data.shape) == 5: # nb x nf x nj x 3 x 3
                    batch_n_fr = tuple(data.shape[:2])
                    reshp = (-1,) + tuple(data.shape[2:])
                else:
                    raise ValueError("Invalid rot_mat shape: {}".format(data.shape))

                if reshp is not None:
                    data = data.reshape(reshp)
                    
                joints = compute_joints(config, rot_mats=data, model=model, relative=isrel,
                        return_torch_objects=isinstance(data, torch.Tensor))
                lg.debug("get_converter-_convert rmats (%s)->%s (%s) took %fs",
                        data.shape, joints_lbl, joints.shape, time.time() - t1)

                if reshp is not None:
                    joints = joints.reshape(batch_n_fr + joints.shape[1:])

                if dt == joints_lbl:
                    out = joints
                else:
                    intm[joints_lbl] = joints
                    out, intm = _convert(joints, joints_lbl, dt, rdepth=rdepth+1, intm=intm)
            elif dt in ('avel', 'aacc'):
                avel = angular_velocity_from_joints_rot_mats(data, fps)
                if dt == 'avel':
                    out = avel
                else:
                    intm['avel'] = avel
                    out, intm = _convert(avel, 'avel', dt, rdepth=rdepth+1, intm=intm)
            elif dt == 'davel':
                out = rot_mats_to_davel(data, fps)
            else:
                _cerr()
        elif st in ('joints', 'joints_rel'):
            isrel = st.endswith('_rel')
            vel_lbl = "vel" + ("_rel" if isrel else "")
            vel = joints_to_vel(data, fps)
            if dt == vel_lbl:
                out = vel
            else:
                intm[vel_lbl] = vel
                out, intm = _convert(vel, vel_lbl, dt, rdepth=rdepth+1, intm=intm)
        elif st == 'vel':
            acc = vel_to_acc(data, fps)
            if dt == 'acc':
                out = acc
            else:
                _cerr()
        elif st == 'avel':
            aacc = angular_velocity_from_joints_rot_mats(data, fps)
            if dt == 'aacc':
                out = aacc
            else:
                _cerr()
        else:
            _cerr()

        lg.debug("get_converter-_convert conversion (%s->%s) took %fs",
                st, dt, time.time() - t0)

        return out, intm

    def wrapper_f(data):
        c, intm = _convert(data, src_type, dest_type)
        if return_intermediates:
            return c, intm
        else:
            return c

    return wrapper_f

