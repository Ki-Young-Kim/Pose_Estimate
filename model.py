from util import ConfigBase, get_device, glg, get_root_dir, dtype_np, dtype_torch
from data_wrangling import totorch, rot_mat_to_vec, rot_6d_to_mat, get_model_path, \
     compute_joints, get_converter, set_get_device_func, \
     SMPL_UPPER_BODY_JOINTS, SMPL_LOWER_BODY_JOINTS, SMPL_LEG_JOINTS
from data_loading import load_torch_model, load_smpl_model
from rnvp import LinearRNVP

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import os
import datetime
import time
import math
import json
from collections import defaultdict, OrderedDict
import random
import math
import pickle
import functools
import logging


LOSS_TYPES = ['MSE', 'R6DNORM', 'RMATNORM', 'MPJRE_DEG',
           'MPJPE', 'MPJVE', #'MPJPE_RC', 'MPJVE_RC', 'ROOT_POS', 'ROOT_VELN'
           'VEL', 'MPJPE_REL', 'MPJVE_REL', 'VEL_REL',
           'ACC_REG', 'AVEL', 'AACC_REG', 'DAVEL',
           'RMAT_UNIT', 'RMAT_ORTHO']
LOSSES_SEP = '+'
is_reg = lambda lt_: lt_.endswith("_REG")
HIDDEN_STATE_TYPES = ['no_randomisation', 'randomise_all', 'sometimes_randomise']
HIDDEN_STATE_RANDOMISATION_METHODS = ['zero', 'xavier_uniform', 'xavier_normal']


def get_required_convs_for_loss(loss_name, is_pred):
    if not is_pred:
        if is_reg(loss_name) or loss_name in ('RMAT_UNIT', 'RMAT_ORTHO'):
            return None

    c = {
        'R6DNORM': 'rot_6d',
        'RMATNORM': 'rot_mats',
        'MSE': 'rot_6d',
        'MPJRE_DEG': 'rot_mats',
        'MPJPE': 'joints',
        #'MPJPE_RC': 'joints+rot_mats',  # to correct root rotation
        #'ROOTJ_POS': 'joints',  # global?
        'MPJVE': 'vel',
        #'MPJVE_RC': 'vel',
        #'ROOTJ_VELN': 'vel',
        'VEL': 'vel',
        'MPJPE_REL': 'joints_rel',
        'VEL_REL': 'vel_rel',
        'MPJVE_REL': 'vel_rel',
        'AAC_REG': 'acc',
        'AVEL': 'avel',
        'AACC_REG': 'aacc',
        'DAVEL': 'davel',
        'RMAT_UNIT': 'rot_6d',
        'RMAT_ORTHO': 'rot_6d',
    }[loss_name]
    return c


def get_loss_func(model_config, loss_name, apply_multiplier=True, keep_batch_dims=False):
    if loss_name in('MPJPE', 'MPJPE_REL', 'R6DNORM', 'RMATNORM', 'MPJVE', 'MPJVE_REL'):
        if loss_name in ('MPJPE', 'MPJPE_REL', 'MPJVE', 'MPJVE_REL'):  # positional
            rj_coeff = model_config['root_joint_pos_error_coeff']
        else:  # rotational
            rj_coeff = model_config['root_joint_rot_error_coeff']

        if math.isclose(1.0, rj_coeff):
            if keep_batch_dims:
                lf = lambda p, o: torch.linalg.vector_norm(p - o, dim=-1)
            else:
                lf = lambda p, o: torch.linalg.vector_norm(p - o, dim=-1).mean()
        else:
            def lf(p, o):
                norms = torch.linalg.vector_norm(p - o, dim=-1)
                norms[..., 0] *= rj_coeff
                if keep_batch_dims:
                    return norms
                else:
                    return norms.mean()
    elif loss_name == 'MPJRE_DEG':
        RAD2DEG = 180 / math.pi
        def lf(p, o):
            d = torch.transpose(o, -1, -2) @ p
            trace = d.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
            angl = torch.acos(torch.clamp((trace - 1) / 2, 0.0, 1.0))
            angl_abs = torch.abs(angl)
            angl_deg_abs = RAD2DEG * angl_abs
            if keep_batch_dims:
                return angl_deg_abs
            else:
                return angl_deg_abs.mean()
    elif loss_name == 'RMAT_UNIT':
        if keep_batch_dims:
            lf = lambda p, o: torch.linalg.vector_norm(p, dim=-2)
        else:
            # (nb x)* nf x nj x 3 x nvecs
            lf = lambda p, o: torch.linalg.vector_norm(p, dim=-2).mean()
    elif loss_name == 'RMAT_ORTHO':
        if keep_batch_dims:
            lf = lambda p, o: torch.linalg.cross(p[..., 0], p[..., 1])
        else:
            # (nb x)* nf x nj x 3 x 2
            lf = lambda p, o: torch.linalg.cross(p[..., 0], p[..., 1]).mean()
    elif loss_name.endswith('_REG'):
        lf = lambda p, o: nn.functional.mse_loss(p, torch.zeros_like(p),
                reduction='none' if keep_batch_dims else 'mean')
    else:
        lf = lambda p, o: nn.functional.mse_loss(p, o,
                reduction='none' if keep_batch_dims else 'mean')

    if apply_multiplier:
        multiplier = model_config['loss_multiplier_{}'.format(loss_name)]
        return lambda p, o=None: multiplier * lf(p, o)
    else:
        return lambda p, o=None: lf(p, o)


class ModelConfig(ConfigBase):
    TARGET_FRAMERATE_DEFAULT = 30 or loss_name == 'MPJVE_REL'

    def __init__(self, data=None):
        super().__init__()

        self._set_default_values()
        if data:
            self.load_from_dict(data)

        self._verify()

    def _set_default_values(self):
        self._set_default('do_correct_amass_rotation', True)
        self._set_default('checkpoint_path', None)
        self._set_default('checkpoints_save_directory', "./checkpoints")

        self._set_default('window_size', 20)
        self._set_default('batch_size', 128)

        self._set_default('shuffle_opts', {})
        self._set_default('shuffle_opts_test', {})

        self._set_default('test_results_raw_fn', TEST_RESULTS_RAW_FN)

        # Input/Output Types
        self._set_default('input_global_joint_positions', False)
        self._set_default('global_joints_grounded', True)

        self._set_default('input_rotations', True)
        self._set_default('input_prev_frame_pose', False)
        self._set_default('input_prev_frame_pose__use_gt', False)
        self._set_default('input_delta_translation', False)
        self._set_default('input_velocities', False)
        self._set_default('input_angular_velocities', False)

        self._set_default('input_9d_rot', False)
        self._set_default('output_9d_rot', False)

        # Overall Architecture
        self._set_default('nn_architecture', 'RNN')

        self._set_default('prefer_6d_rotation', True)
        self._set_default('beta_size', 10)
        self._set_default('target_framerate', 30)
        self._set_default('normalise_shape', True)
        self._set_default('njoints_cutoff', 22)
        self._set_default('loss', 'MSE')
        self._set_default('loss_multiplier_MSE', 1.0)
        self._set_default('loss_multiplier_R6DNORM', 1.0)
        self._set_default('loss_multiplier_RMATNORM', 1.0)
        self._set_default('loss_multiplier_MPJRE_DEG', math.pi / 180)
        self._set_default('loss_multiplier_MPJPE', 1.0)
        self._set_default('loss_multiplier_MPJPE_REL', 1.0)
        self._set_default('loss_multiplier_MPJVE', 1.0)
        self._set_default('loss_multiplier_VEL', 1.0)
        self._set_default('loss_multiplier_MPJVE_REL', 1.0)
        self._set_default('loss_multiplier_VEL_REL', 1.0)
        self._set_default('loss_multiplier_ACC_REG', 0.2)
        self._set_default('loss_multiplier_AVEL', 1.0)
        self._set_default('loss_multiplier_AACC_REG', 0.2)
        self._set_default('loss_multiplier_DAVEL', 1.0)
        self._set_default('loss_multiplier_RMAT_UNIT', 0.1)
        self._set_default('loss_multiplier_RMAT_ORTHO', 0.1)

        self._set_default('root_joint_pos_error_coeff', 1.0)
        self._set_default('root_joint_rot_error_coeff', 1.0)

        self._set_default('normalise_global_joint_positions', True)
        self._set_default('normalise_global_joint_positions_y', False)
        self._set_default('normalise_global_joint_positions_divbystd', False)

        # Positional Encoding
        self._set_default('positional_encoding', False)
        self._set_default('input_encoder', 'none')
        self._set_default('simple_input_encoder_layernorm', False)
        self._set_default('input_encoder_dim', 256)

        self._set_default('pe_L_j3', 4)
        self._set_default('pe_L_rotations', 4)
        self._set_default('pe_L_velocities', 4)
        self._set_default('pe_L_angular_velocities', 4)
        self._set_default('pe_L_dtransls', 4)

        self._set_default('pe_max_freq_log2_j3', self['pe_L_j3'] - 1)
        self._set_default('pe_max_freq_log2_rotations', self['pe_L_rotations'] - 1)
        self._set_default('pe_max_freq_log2_velocities', self['pe_L_velocities'] - 1)
        self._set_default('pe_max_freq_log2_angular_velocities', self['pe_L_angular_velocities'] - 1)
        self._set_default('pe_max_freq_log2_dtransls', self['pe_L_dtransls'] - 1)


        # RNN
        self._set_default('hidden_size', 128)
        self._set_default('rnn_layers', 3)
        self._set_default('rnn_dropout', 0.2)
        self._set_default('hidden_state_type', 'sometimes_randomise')
        self._set_default('hidden_state_randomisation_ratio', 0.8)
        self._set_default('hidden_state_randomisation_method', 'xavier_uniform')

        # VAE
        self._set_default('input_seq_length', 1)
        self._set_default('resnet_kdiv3', 1)
        self._set_default('encoder_njoints', 3)
        self._set_default('latent_dim', 30)  # 15 30 60
        self._set_default('betavae_beta', 1)

        # Training
        self._set_default('learning_rate', 0.002)
        self._set_default('lr_decay_step_size', 15)
        self._set_default('lr_decay_gamma', 0.5)
        self._set_default('epochs', 60)
        self._set_default('log_every_n_global_steps', 150)

        self._set_default('random_root_rotation', True) # Randomly rotate root joint before data load

        # Misc
        self._set_default('target_framerate', self.TARGET_FRAMERATE_DEFAULT)

        self._set_default('viz_global_pose', False)

        # Deprecated settings
        self._set_default('training_data_percentage', 0.9)
        self._set_default('nsamples', 6)

    def _verify(self):
        def throwerr(field, defaultv=None, msg=None):
            raise ValueError("Invalid {}: {}{}".format(field, self.get(field, defaultv),
                        " ({})".format(msg) if msg else ""))

        if any(lt not in LOSS_TYPES for lt in self['loss'].split(LOSSES_SEP)):
            throwerr('loss')
        if self['hidden_state_type'] not in HIDDEN_STATE_TYPES:
            throwerr('hidden_state_type')
        if self['hidden_state_randomisation_method'] not in HIDDEN_STATE_RANDOMISATION_METHODS:
            throwerr('hidden_state_randomisation_method')
        if not (0 <= self.get('hidden_state_randomisation_ratio', 0) <= 1):
            throwerr('hidden_state_randomisation_ratio')
        if self['input_prev_frame_pose'] and not self['input_prev_frame_pose__use_gt']:
            raise NotImplementedError("With the current data loaders it is difficult to "
                    "feed previous frame pose prediction (at least during training")
        possible_inp_encs = ['none', 'simple']
        if not self['input_encoder'] in possible_inp_encs:
            throwerr('input_encoder', "Must be one of: {}".format(possible_inp_encs))

    @property
    def input_dim(self):
        sz = self.input_njoints * 3
        if self['input_rotations']:
            sz += self.input_njoints * self.in_rot_d
        if self['input_prev_frame_pose']:
            sz += self.in_rot_d * self.output_njoints
        if self['input_delta_translation']:
            sz += 3
        if self['input_velocities']:
            sz += self.input_njoints * 3
        if self['input_angular_velocities']:
            sz += self.input_njoints * self.in_rot_d
        return sz

    @property
    def rot_d(self):
        if self['prefer_6d_rotation']:
            return 6
        else:
            return 3
    
    @property
    def in_rot_d(self):
        if self['input_9d_rot']:
            return 9
        else:
            return self.rot_d

    @property
    def out_rot_d(self):
        if self['output_9d_rot']:
            return 9
        else:
            return self.rot_d

    @property
    def output_dim(self):
        return self.out_rot_d * self.output_njoints

    @property
    def input_joints(self):
        return [15, 20, 21]  # head, wrist, lwrist

    @property
    def input_njoints(self):
        return len(self.input_joints)

    @property
    def output_njoints(self):
        return 22

    @property
    def upper_body_joints(self):
        return tuple(k for k in SMPL_UPPER_BODY_JOINTS if k < self.output_njoints)

    @property
    def leg_joints(self):
        return SMPL_LEG_JOINTS

    @property
    def lower_body_joints(self):
        return SMPL_LOWER_BODY_JOINTS

    def __getitem__(self, key):
        dk_props = {'input_dim', 'rot_d', 'output_dim', 'input_joints', 'input_njoints', 'output_njoints',
        'leg_joints', 'lower_body_joints', 'upper_body_joints', 'in_rot_d', 'out_rot_d'}
        if key in dk_props:
            return getattr(self, key)
        else:
            return super().__getitem__(key)


class BaseNN(nn.Module):
    def __init__(self, config, cvt_kws=None):
        super().__init__()
        self.config = config
        self.mc = config['model_config']
        self.lg = glg()
        
        self._convs_p = []
        self._convs_o = []
        self._loss_fns = []
        for lss in self.mc['loss'].split(LOSSES_SEP):
            conv_p = get_required_convs_for_loss(lss, is_pred=True)
            conv_o = get_required_convs_for_loss(lss, is_pred=False)
            if conv_p is not None:
                self._convs_p.append(conv_p)
            if conv_o is not None:
                self._convs_o.append(conv_o)
            self._loss_fns.append(
                    (lss, is_reg(lss), get_loss_func(self.mc, lss), conv_p, conv_o))

        smplmodel_path = get_model_path(config, 'smpl', 'male')
        smplmodel = load_smpl_model(smplmodel_path, as_class=False)
        self.convert_pred = nn_out_converter(config=config, targets=self._convs_p,
                training=True, model=smplmodel, **(cvt_kws or {}))
        self.convert_outp = nn_out_converter(config=config, targets=self._convs_o,
                training=True, model=smplmodel, **(cvt_kws or {}))

        #self.inp_j3_begin = -1
        #self.inp_j3_endp1 = -1
        #self.input_njoints = self.mc.input_njoints
        #if self.mc['input_global_joint_positions'] and \
        #    self.mc['normalise_global_joint_positions']:
        #    self.inp_pos_normaliser = CoordinatesNormaliser(exceptaxis=1)
        #    input_types = get_input_types()
        #    for (st, edp1), inp_t in input_types:
        #        if inp_t == 'j3':
        #            self.inp_j3_begin = st
        #            self.inp_j3_endp1 = endp1
        #    assert self.inp_j3_begin >= 0 and self.inp_j3_endp1 >= 1
        #else:
        #    self.inp_pos_normaliser = None

        self.name = self.mc['name']
        self.device = get_device()

        self.test_converter = nn_out_converter(config=config,
                targets=['joints', 'vel', 'acc', 'avel', 'aacc'], training=False)

    def criterion(self, pred_convs, outp_convs):
        loss_sum = None
        loss_sum_view = None
        for loss_name, loss_isreg, lfn, conv_p, conv_o in self._loss_fns:
            pred_e = pred_convs.get(conv_p, None)
            outp_e = outp_convs.get(conv_o, None)
            loss = lfn(pred_e, outp_e)
            if loss_sum is None:
                loss_sum = loss
            else:
                loss_sum += loss

            if not loss_isreg:
                if loss_sum_view is None:
                    loss_sum_view = loss
                else:
                    loss_sum_view += loss
                
        return loss_sum, loss_sum_view

    def start_training(self, training_data_loader, validation_data_loader=None,
            checkpoints_save_dir=None, checkpoint_path=None, train_graph_path=None,
            nepochs=None, window_sz=None, **kw):
        global_step = 0
        nepochs = nepochs or self.mc['epochs']
        training_datas = set()
        val_stepsize = kw.get('validation_stepsize', 2)

        if checkpoint_path:
            glg().info("Loading checkpoint: %s", checkpoint_path)
            load_torch_model(self, checkpoint_path)
        checkpoints_save_dir = checkpoints_save_dir or self.mc['checkpoints_save_directory']
        if not train_graph_path:
            train_graph_path = os.path.join(checkpoints_save_dir, 'train_graph.png')
            os.makedirs(os.path.dirname(train_graph_path), exist_ok=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.mc['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                step_size=self.mc['lr_decay_step_size'],
                gamma=self.mc['lr_decay_gamma'])

        cur_step_custom_data = {}
        train_init_data = self._training_init_data()

        start_dt = datetime.datetime.now()
        glg().info("Started training '%s' at %s", self.name, start_dt)

        lg = self.lg
        losses_cur_epoch = []
        all_train_losses = {}
        all_val_losses = {}
        global_step == 0
        with SummaryWriter(self.mc['tensorboard_logdir']) as writer:
            for epoch in range(nepochs):
                lg.info("Epoch %d/%d", epoch+1, nepochs)
                self.train()
                train_loss_array = []
                n_processed_inputs_this_epoch = 0
                t0 = time.time()
                tepoch = t0
                file_idx = -1000
                for i_data, d in enumerate(training_data_loader(window_sz=window_sz)):
                    file_idx_new = d.get('file_idx', -1)
                    if i_data > 0 and file_idx_new >= 0 and file_idx_new == file_idx:
                        same_file = True
                    else:
                        same_file = False
                    file_idx = file_idx_new

                    lg.debug("Data loading time: %f", time.time() - t0)

                    inp = totorch(d['input'], device=self.device)
                    outp = totorch(d['output'], device=self.device)

                    #if self.inp_pos_normaliser is not None:
                    #    j3 = inp[..., self.inp_j3_begin:self.inp_j3_endp1]
                    #    bnf = tuple(j3.shape[:-1])
                    #    j3 = j3.reshape(bnf + (self.input_njoints, 3))
                    #    breakpoint()
                    #    inp[..., self.inp_j3_begin:self.inp_j3_endp1] = \
                    #        self.inp_pos_normaliser(j3).reshape(bnf + (-1,))
                    #    breakpoint()

                    meta = {
                        'same_file': same_file,
                        'data': d,
                        'writer': writer
                    }

                    optimizer.zero_grad()

                    t0 = time.time()

                    loss, cur_step_custom_data = \
                        self._training_step_callback(inp, outp, meta=meta,
                                prev_step_custom_data=cur_step_custom_data,
                                train_init_data=train_init_data,
                                train_get=lambda k: cur_step_custom_data.get(k,
                                    train_init_data.get(k, None)))

                    lg.debug("Train step time: %f", time.time() - t0)

                    optimizer.step()

                    if loss is not None:
                        losses_cur_epoch.append(loss)

                    if isinstance(d['path'], str):
                        training_datas.add(d['path'])
                    else:  # is a list of paths
                        training_datas.update(d['path'])

                    if global_step % self.mc['log_every_n_global_steps'] == 0:
                        cum_loss = np.mean(losses_cur_epoch)
                        lg.info("%s - epoch: %d; global step: %d; loss: %f",
                            datetime.datetime.now(), epoch, global_step, cum_loss)

                    global_step += 1
                    t0 = time.time()
                
                loss_epoch = np.mean(losses_cur_epoch or [np.nan])
                loss_epoch_std = np.std(losses_cur_epoch or [np.nan])
                loss_epoch_med = np.median(losses_cur_epoch or [np.nan])
                losses_cur_epoch.clear()
                all_train_losses[epoch] = loss_epoch
                lg.info(("The train loss of '{}' is {:.6f} (std={:.4f}, median={:.6f})."
                        " Epoch {} took {:.1f}s").format(
                            self.name, loss_epoch, loss_epoch_std, loss_epoch_med,
                            epoch+1, time.time() - tepoch))

                val_loss = None
                val_loss_std = None
                val_loss_med = None
                if validation_data_loader is not None:
                    if epoch > 0 and epoch % val_stepsize == 0:
                        lg.info("Validating...")
                        self.eval()
                        with torch.no_grad():
                            val_loss_d = self._validate(validation_data_loader,
                                    writer=writer, train_init_data=train_init_data,
                                    window_sz=window_sz)
                            val_loss = val_loss_d['mean']
                            val_loss_std = val_loss_d['std']
                            val_loss_med = val_loss_d['median']
                        self.train()
                else:
                    lg.warning("Validation data not provided. Skipping validation.")

                if val_loss is not None:
                    all_val_losses[epoch] = val_loss
                    lg.info(("Validation loss of '{}' is {:.6f}"
                            " (std={:.4f}, median={:.6f})").format(
                                self.name, val_loss, val_loss_std, val_loss_med))

                fig = self._plot_losses(all_train_losses, all_val_losses)
                lg.info("Saving training graph at: %s", train_graph_path)
                fig.savefig(train_graph_path)

                try:
                    writer.add_scalar("'{}' Train Loss".format(self.name),
                            loss_epoch, epoch)
                    if val_loss is not None:
                        writer.add_scalar("'{}' Validation Loss".format(self.name),
                                val_loss, epoch)
                except:
                    lg.warning("Failed to write train loss to writer")
                checkpoint_save_path = os.path.join(checkpoints_save_dir,
                        "checkpoint_{}_epoch_{:02}_of_{:02}({}).chkpt".format(
                    start_dt.strftime('%Y%m%dT%H%M%S'), epoch+1, nepochs,
                    datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
                os.makedirs(checkpoints_save_dir, exist_ok=True)
                torch.save(self.state_dict(), checkpoint_save_path)
                lg.info("Saved checkpoint at %s", checkpoint_save_path)
                training_datas_path = os.path.splitext(checkpoint_save_path)[0] + "_training_data.json"
                with open(training_datas_path, 'w', encoding='utf-8') as f:
                    json.dump(list(training_datas), f)

                scheduler.step()
                lg.info("Last lr: {}".format(scheduler.get_last_lr()))

                try:
                    writer.flush()
                except:
                    lg.warning("Failed to flush writer")

            lg.info("Training done")

    def _validate(self, validation_data_loader, writer, train_init_data, window_sz=None):
        file_idx = -1000
        losses = []
        cur_step_custom_data = {}
        for i_data, d in enumerate(validation_data_loader(window_sz=window_sz)):
            file_idx_new = d.get('file_idx', -1)
            if i_data > 0 and file_idx_new >= 0 and file_idx_new == file_idx:
                same_file = True
            else:
                same_file = False
            file_idx = file_idx_new

            inp = totorch(d['input'], device=self.device)
            outp = totorch(d['output'], device=self.device)

            meta = {
                'same_file': same_file,
                'data': d,
                'writer': writer
            }

            loss, cur_step_custom_data = \
                self._validation_step_callback(inp, outp, meta=meta,
                        prev_step_custom_data=cur_step_custom_data,
                        train_init_data=train_init_data,
                        train_get=lambda k: cur_step_custom_data.get(k,
                            train_init_data.get(k, None)))

            if loss is not None:
                losses.append(loss)
        mean = np.mean(losses)
        std = np.std(losses)
        med = np.median(losses)
        return {
            'mean': mean,
            'std': std,
            'median': med
        }

    def _plot_losses(self, train_losses: dict, validation_losses: dict):
        train_col = "#1E90FF"
        valid_col = "#FF8C00"

        hasvalid = len(validation_losses) > 0

        ephs_train = np.zeros(len(train_losses), dtype=np.int)
        trainlosses_a = np.zeros(len(train_losses), dtype=np.float32)
        for i, (eph, lss) in enumerate(train_losses.items()):
            ephs_train[i] = eph
            trainlosses_a[i] = lss
        mintrainloss_i = np.argmin(trainlosses_a)
        mintrainloss = trainlosses_a[mintrainloss_i]
        mineph, maxeph = np.min(ephs_train), np.max(ephs_train)

        if hasvalid:
            ephs_valid = np.zeros(len(validation_losses), dtype=np.int)
            validlosses_a = np.zeros(len(validation_losses), dtype=np.float32)
            for i, (eph, lss) in enumerate(validation_losses.items()):
                ephs_valid[i] = eph
                validlosses_a[i] = lss
            minvalidloss_i = np.argmin(validlosses_a)
            minvalidloss = validlosses_a[minvalidloss_i]
        else:
            ephs_valid = None
            validlosses_a = None
            minvalidloss_i = None
            minvalidloss = None

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(ephs_train, trainlosses_a, label='Train', color=train_col)
        if hasvalid:
            ax.plot(ephs_valid, validlosses_a, label='Validation', color=valid_col)
        ax.legend(loc="upper right");

        ax.plot([mineph, maxeph], [mintrainloss, mintrainloss], color='#A9A9A9', linestyle='--')
        mintrain_p = (ephs_train[mintrainloss_i], mintrainloss)

        ax.plot([mintrain_p[0]], [mintrain_p[1]], color=train_col, marker='o')
        ax.text(*mintrain_p, "({}, {:.03f})".format(*mintrain_p), color="#000080",
                horizontalalignment='left', verticalalignment='bottom')

        if hasvalid:
            ax.plot([mineph, maxeph], [minvalidloss, minvalidloss], color='#676767', linestyle='--')
            minvalid_p = (ephs_valid[minvalidloss_i], minvalidloss)
            ax.plot([minvalid_p[0]], [minvalid_p[1]], color=valid_col, marker='o')
            ax.text(*minvalid_p, "({}, {:.03f})".format(*minvalid_p), color="#A0522D",
                    horizontalalignment='left', verticalalignment='bottom')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        fig.tight_layout()
        return fig

    def start_testing(self, testing_data_loader, results_save_dir=None, window_sz=None,
            inference_callback=None, dset_path_index=-5):
        if results_save_dir:
            os.makedirs(results_save_dir, exist_ok=True)
        else:
            glg().info("No results_save_dir specified; test results will not be saved on disk")

        test_init_data = self._testing_init_data()

        def gcf(l):
            lf = get_loss_func(self.mc, l, apply_multiplier=False, keep_batch_dims=True)
            def _compute_flattened_loss(p, o):
                _arr = lf(p, o).detach().cpu().numpy()
                # Average across joints/output values for a single frame
                _arr = np.mean(_arr, axis=-1)
                return _arr.flatten()
            return _compute_flattened_loss

        compute_mse = gcf('MSE')
        compute_mpjpe = gcf('MPJPE')
        compute_mpjre = gcf('MPJRE_DEG')
        compute_mpjve = gcf('MPJVE')
        #compute_accerr = gcf('ACC')
        #compute_avelerr = gcf('AVEL')
        #compute_aaccerr = gcf('AACC')

        result_lists = {
            'losses': defaultdict(lambda: np.array([], dtype_np)),
            'datasets': [],
            'groups': [],
            'genders': [],
            'nframes': []
        }

        def concat_losses(key, lss):
            result_lists['losses'][key] = np.append(result_lists['losses'][key], lss)

        file_idx = -1000
        cur_step_custom_data = {}

        self.eval()
        with torch.no_grad():
            for i, d in enumerate(tqdm(testing_data_loader(window_sz=window_sz))):
                file_idx_new = d.get('file_idx', -1)
                if i > 0 and file_idx_new >= 0 and file_idx_new == file_idx:
                    same_file = True
                else:
                    same_file = False
                inp = totorch(d['input'], device=self.device)
                outp = totorch(d['output'], device=self.device)

                meta = {
                    'same_file': same_file,
                    'data': d,
                    'index': i
                }

                (pred, outp), cur_step_custom_data = \
                    self._testing_step_callback(inp, outp, meta=meta,
                            prev_step_custom_data=cur_step_custom_data,
                            testing_init_data=test_init_data,
                            test_get=lambda k: cur_step_custom_data.get(k,
                                test_init_data.get(k, None)),
                            is_inference=inference_callback is not None)

                # Batchify
                if len(inp.shape) == 2:
                    isbatch = False
                    nbatch = 1
                elif len(inp.shape) == 3:
                    isbatch = True
                    nbatch = inp.shape[0]
                else:
                    raise ValueError
                if not isbatch:
                    outp = outp[None]
                    pred = pred[None]
                
                paths = list(d['path']) if not isinstance(d['path'], str) \
                        else nbatch * [d['path']]

                dsets = []
                groups = []
                nfrms = list(d['nframes']) if not (isinstance(d['nframes'], int) or isinstance(d['nframes'], np.int64)) \
                        else nbatch * [d['nframes']]
                genders = list(d['gender']) if not isinstance(d['gender'], str) \
                          else nbatch * [d['gender']]
                preprocessed_rootdir_abs = os.path.abspath(
                        get_root_dir(self.config, 'preprocessing'))
                for ip in range(len(paths)):
                    #paths[ip] = os.path.relpath(os.path.abspath(paths[ip]), preprocessed_rootdir_abs)
                    #ds = os.path.normpath(paths[ip]).replace("..{}".format(os.path.sep), "") \
                    #    .split(os.path.sep)[1]
                    pathsplit = os.path.normpath(paths[ip]).split(os.path.sep)
                    ds = pathsplit[dset_path_index]
                    motion_name = os.path.sep.join(pathsplit[-2:])
                    dsets.append(ds)
                    groups.append("{}: {}".format(ds, motion_name))

                c_pred = self.test_converter(pred)
                c_outp = self.test_converter(outp)

                if inference_callback is not None:
                    pred_j = c_pred['joints'].cpu().numpy()
                    outp_j = c_outp['joints'].cpu().numpy()
                    pred_j_l = []
                    outp_j_l = []
                    for b in range(pred_j.shape[0]):  # Iterate within batch
                        pred_j_l.append(model_output_joints_to_smpl_joints(pred_j[b]))
                        outp_j_l.append(model_output_joints_to_smpl_joints(outp_j[b]))
                    pred_j = np.array(pred_j_l)
                    outp_j = np.array(outp_j_l)
                    inference_callback(pred_j, outp_j, data=d, meta=meta)

                l_mse = compute_mse(pred, outp)
                rm_p, rm_o = c_pred['rot_mats'], c_outp['rot_mats']
                l_mpjre = compute_mpjre(rm_p, rm_o)
                l_mpjre_ub = compute_mpjre(rm_p[..., self.mc.upper_body_joints, :, :],
                        rm_o[..., self.mc.upper_body_joints, :, :])
                l_mpjre_lb = compute_mpjre(rm_p[..., self.mc.lower_body_joints, :, :],
                        rm_o[..., self.mc.lower_body_joints, :, :])
                l_mpjre_legs = compute_mpjre(rm_p[..., self.mc.leg_joints, :, :],
                        rm_o[..., self.mc.leg_joints, :, :])
                j_p, j_o = c_pred['joints'], c_outp['joints']
                l_mpjpe = compute_mpjpe(j_p, j_o)
                l_mpjpe_ub = compute_mpjpe(j_p[..., self.mc.upper_body_joints, :],
                        j_o[..., self.mc.upper_body_joints, :])
                l_mpjpe_lb = compute_mpjpe(j_p[..., self.mc.lower_body_joints, :],
                        j_o[..., self.mc.lower_body_joints, :])
                l_mpjpe_legs = compute_mpjpe(j_p[..., self.mc.leg_joints, :],
                        j_o[..., self.mc.leg_joints, :])
                l_mpjve = compute_mpjve(c_pred['vel'], c_outp['vel'])
                #l_accerr = compute_accerr(c_pred['acc'], c_outp['acc'])
                #l_avelerr = compute_avelerr(c_pred['avel'], c_outp['avel'])
                #l_aaccerr = compute_aaccerr(c_pred['aacc'], c_outp['aacc'])

                concat_losses('mse', l_mse)
                concat_losses('mpjre', l_mpjre)
                concat_losses('mpjre_upper_body', l_mpjre_ub)
                concat_losses('mpjre_lower_body', l_mpjre_lb)
                concat_losses('mpjre_legs', l_mpjre_legs)
                concat_losses('mpjpe', l_mpjpe)
                concat_losses('mpjpe_upper_body', l_mpjpe_ub)
                concat_losses('mpjpe_lower_body', l_mpjpe_lb)
                concat_losses('mpjpe_legs', l_mpjpe_legs)
                concat_losses('mpjve', l_mpjve)
                #concat_losses('accerr', l_accerr)
                #concat_losses('avelerr', l_avelerr)
                #concat_losses('aaccerr', l_aaccerr)
                result_lists['datasets'].extend(dsets)
                result_lists['groups'].extend(groups)
                result_lists['nframes'].extend(nfrms)
                result_lists['genders'].extend(genders)

            if results_save_dir is not None:
                analyse_losses(losses=result_lists['losses'],
                        meta={k: v for (k, v) in result_lists.items() if k != 'losses'},
                        save_dir=results_save_dir, model_config=self.mc)

                raw_file_fp = os.path.join(results_save_dir, TEST_RESULTS_RAW_FN)
                glg().info("Saving raw file to \"%s\"...", raw_file_fp)
                result_lists['losses'] = dict(result_lists['losses'])
                with open(raw_file_fp, 'wb') as f:
                    d = {
                        'config': self.config,
                        'model_config': self.mc,
                        'lists': result_lists
                    }
                    pickle.dump(d, f)

    def _training_init_data(self):
        return {}

    def _testing_init_data(self):
        return {}

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            training_init_data, train_get):
        raise AssertionError("Training step callbaack not implemented")
        loss = None
        return loss, prev_step_custom_data

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            training_init_data, train_get):
        glg().warning("Validation step callback not implemented")
        return None, prev_step_custom_data

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        raise AssertionError("Testing step callbaack not implemented")
        return (None, None), prev_step_custom_data


def get_input_types(model_config):
    from preprocessing import preprocessed_to_model_inout
    _, _, intm = preprocessed_to_model_inout(None, model_config=model_config,
            train_or_test='train', return_intermediate_values=True, test_run=True)
    input_types = intm['input_types']
    assert len(input_types) > 0
    return input_types


#class CoordinatesNormaliser(nn.Module):
#    def __init__(self, exceptaxis=1):
#        super().__init__(self)
#        self.exclude_axis = 1
#
#    def forward(self, pos):
#        # pos.detach() unnecessary in most cases because pos is usually the input
#        # which is not affected by gradient update
#        pos_mean = torch.mean(pos, dim=-2, keepdim=True)
#        pos_std = torch.std(pos, dim=-2, keepdim=True)
#        pos_mean[..., exclude_axis] = 0.0
#        pos_std[..., exclude_axis] = 1.0
#        return (pos - pos_mean) / pos_std


# yenchenlin/nerf-pytorch
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# yenchenlin/nerf-pytorch
def get_embedder(in_d, multires, max_freq_log2=None, i=0):
    if i == -1:
        return nn.Identity(), in_d

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : in_d,
                'max_freq_log2' : max_freq_log2 or (multires - 1),
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
                #'add': False
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class PositionalEncoder(nn.Module):
    def __init__(self, model_config, input_types=None):
        """
        input_types: from preprocessing.preprocessed_to_model_inout
        """
        super().__init__()
        self.mc = model_config
        self.input_types = input_types
        self.embedders = []
        self.out_dim = -1
        self.__setup_embedders(input_types)

    def __setup_embedders(self, input_types):
        self.embedders.clear()
        if not input_types:
            input_types = get_input_types(self.mc)

        out_dim = 0
        for (st, edp1), inp_t in input_types:
            embedder, od = get_embedder(in_d=edp1 - st,
                    multires=self.mc['pe_L_{}'.format(inp_t)],
                    max_freq_log2=self.mc['pe_max_freq_log2_{}'.format(inp_t)])
            self.embedders.append((((st, edp1), inp_t), embedder))
            out_dim += od
        self.out_dim = out_dim

    def forward(self, x):
        assert len(self.embedders) > 0

        out = []
        for i, (((st, edp1), _inp_t), embedder) in enumerate(self.embedders):
            out.append(embedder(x[..., st:edp1]))
        return torch.cat(out, dim=-1)


class SimpleSparseInputEncoder(nn.Module):
    def __init__(self, model_config, in_dim, **kw):
        super().__init__()
        self.mc = model_config
        dim = model_config['input_encoder_dim']
        self.dim = dim
        nets = [nn.Linear(in_dim, dim),
                nn.BatchNorm(dim) if kw.get('use_batchnorm', False) 
                    else nn.Dropout(kw.get('dropout_pct', 0.2)),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(dim, dim)]
        if model_config['simple_input_encoder_layernorm']:
            nets.append(nn.LayerNorm(dim))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net.forward(x)

    @property
    def out_dim(self):
        return self.dim


class RNNBasedNN(BaseNN):
    def __init__(self, config):
        super().__init__(config=config)
        hidden_size = self.mc['hidden_size']
        nlayers = self.mc['rnn_layers']
        dropout = self.mc['rnn_dropout']

        dim = self.mc.input_dim
        if self.mc['positional_encoding']:
            self.pe = PositionalEncoder(model_config=self.mc)
            dim = self.pe.out_dim
        else:
            self.pe = None

        if self.mc['input_encoder'] == 'simple':
            self.inp_enc = SimpleSparseInputEncoder(model_config=self.mc, in_dim=dim)
            dim = self.inp_enc.out_dim
        else:
            self.inp_enc = None

        self.rnn = nn.LSTM(input_size=dim, hidden_size=hidden_size,
                         num_layers=nlayers, batch_first=True, dropout=dropout)
        self.linear_out = nn.Linear(in_features=hidden_size, out_features=self.mc['output_dim'])

    def forward(self, x, hidden=None):
        if self.pe is not None:
            x = self.pe(x)
        if self.inp_enc is not None:
            x = self.inp_enc(x)

        x, hidden = self.rnn(x, hidden)
        x = self.linear_out(x)
        return x, hidden

    def _training_init_data(self):
        hst = self.mc['hidden_state_type']
        return {
            'hst': hst,
        }

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get, is_training=True):
        hst = train_init_data['hst']
        pred = train_get('pred')
        hidden_train = train_get('hidden_train')
        same_file = meta['same_file']

        # Autoregression
        if self.mc['input_prev_frame_pose'] and \
            not self.mc['input_prev_frame_pose__use_gt']:
            i1 = meta['data']['intm_vals']['prev_frame_pose_idx_begin']
            i2 = meta['data']['intm_vals']['prev_frame_pose_idx_end']
            if pred is None or not same_file:
                pred_inp = outp
            else:
                pred_inp = pred
            inp[..., i1:i2+1] = pred_inp

        # FP
        random_hidden = False
        if not same_file:
            hidden_train = None  # TODO?
            random_hidden = True
        pred, hidden_train = self(inp, hidden_train)

        # Determine next hidden state input
        h_0, c_0 = hidden_train
        h_0, c_0 = h_0.detach(), c_0.detach()
        if hst == 'no_randomisation' or (hst == 'sometimes_randomise' and
                random.random() < (1 - self.mc['hidden_state_randomisation_ratio'])):
            hidden_train = (h_0, c_0)
        else:
            random_hidden = True
            hsrm = self.mc['hidden_state_randomisation_method']
            if hsrm == 'zero':
                hidden_train = (torch.zeros_like(h_0), torch.zeros_like(c_0))
            elif hsrm == 'xavier_uniform':
                hidden_train = (nn.init.xavier_uniform_(h_0),
                        nn.init.xavier_uniform_(c_0))
                assert hidden_train[0] is not None
            elif hsrm == 'xavier_normal':
                hidden_train = (nn.init.xavier_normal_(h_0),
                        nn.init.xavier_normal_(c_0))
                assert hidden_train[0] is not None
            else:
                raise ValueError("Unknown hidden_state_randomisation_method")

        if random_hidden:
            # Evaluate last frame only (flat input assumed)
            pred_e = pred[..., -3:, :].contiguous()
            outp_e = outp[..., -3:, :]
        else:
            pred_e = pred
            outp_e = outp

        t0 = time.time()

        pred_cvt = self.convert_pred(pred_e)
        outp_cvt = self.convert_outp(outp_e)

        self.lg.debug("RNNBasedModel conversion time: %f", time.time() - t0)

        t0 = time.time()
        
        loss, loss_view = self.criterion(pred_cvt, outp_cvt)

        self.lg.debug("RNNBasedModel loss computation time: %f", time.time() - t0)

        t0 = time.time()

        if is_training:
            loss.backward()

        self.lg.debug("RNNBasedModel BP time: %f", time.time() - t0)
        
        return (loss_view.item(),
                {
                    'pred': pred.detach(),
                    'hidden_train': hidden_train
                })

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get):
        return self._training_step_callback(inp, outp, meta, prev_step_custom_data,
                train_init_data, train_get, is_training=False)

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        pred = test_get('pred')
        hidden_predict = test_get('hidden_predict')
        same_file = meta['same_file']

        if self.mc['input_prev_frame_pose'] and \
            not self.mc['input_prev_frame_pose__use_gt']:
            i1 = d['intm_vals']['prev_frame_pose_idx_begin']
            i2 = d['intm_vals']['prev_frame_pose_idx_end']

            if pred is not None:
                pred_inp = pred
            else:
                # TODO using zero pose for 1st autoregression for now
                pred_inp = torch.zeros((i2 - i1 + 1,), dtype=inp.dtype)

            inp[..., i1:i2+1] = pred_inp

        random_hidden = False
        if not same_file:
            hidden_predict = None
            random_hidden = True

        pred, hidden_predict = self(inp, hidden_predict)
        h_0, c_0 = hidden_predict
        hidden_predict = (h_0.detach(), c_0.detach())
        pred = pred.detach()

        if not is_inference and random_hidden:
            # Evaluate last frame only (flat input assumed)
            pred_e = pred[..., -3:, :].contiguous()
            outp_e = outp[..., -3:, :]
        else:
            pred_e = pred
            outp_e = outp

        return ((pred_e.detach(), outp_e),
                {
                    'pred': pred,
                    'hidden_predict': hidden_predict,
                })


class LinearResidualBlock(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 dropout_p=0.2):
        super().__init__()

        self.net = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(in_size, out_size),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(in_size, out_size),
                #nn.BatchNorm1d(out_size)
            )
        self.alpha = nn.Parameter(torch.tensor([0.0], dtype=dtype_torch),
                                  requires_grad=True)

    def forward(self, x):
        return x + self.alpha * self.net(x)


class AddCoords(nn.Module):
    """
    https://github.com/walsvid/CoordConv/
    """
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv1d(nn.modules.conv.Conv1d):
    """
    https://github.com/walsvid/CoordConv/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class ConvolutionalResidualBlock(nn.Module):
    def __init(self, inout_size,
            kernel_size=3, stride=1, padding=1,
            dropout_p=0.2):
        super().__init__()

        assert kernel % 2 == 1

        self.net = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.01),
                CoordConv1d(inout_size, inout_size, kernel_size=kernel_size,
                    stride=stride, padding=padding),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.01),
                CoordConv1d(inout_size, inout_size, kernel_size=kernel_size,
                    stride=stride, padding=padding))

        self.alpha = nn.Parameter(torch.tensor([0.0], dtype=dtype_torch),
                                  requires_grad=True)

    def forward(self, x):
        return x + self.alpha * self.net(x)


class VAEBasedNN(BaseNN):
    def __init__(self, config):
        super().__init__(config=config)
        self.seq_len = self.mc['input_seq_length']
        self.k = 3 * self.mc['resnet_kdiv3']
        self.latent_dim = self.mc['latent_dim']
        self.beta = self.mc['betavae_beta']

        self.linear_features_fixed = 256
        self.conv_features_fixed = 128

        in_njoints = self.mc['encoder_njoints']
        if in_njoints == 3:
            self.input_fullpose = False
        elif in_njoints == 22:
            self.input_fullpose = True
        else:
            raise ValueError("Invalid #inputjoints: {}".format(in_njoints))

        self.__init_encoder(in_njoints)
        self.__init_decoder()

    def __init_encoder(self, input_njoints):
        in_size = self.seq_len * input_njoints * \
                  ((0 if self.input_fullpose else 3) + self.mc.in_rot_d)
        modules = []
        if self.seq_len == 1:
            modules.append(nn.Linear(in_size, self.linear_features_fixed))
            for i in range(self.k):
                rb = LinearResidualBlock(
                        self.linear_features_fixed, self.linear_features_fixed)
                modules.append(rb)
            #modules.append(nn.LeakyReLU())
            modules.append(
                    nn.Linear(self.linear_features_fixed, 2*self.latent_dim))
        elif self.seq_len == 16:
            modules.append(nn.Linear(in_size, self.conv_features_fixed))
            for i in range(self.k // 3):
                rb = ConvolutionalResidualBlock(
                        self.conv_features_fixed, self.conv_features_fixed)
                modules.append(rb)
            modules.append(
                    nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

            for i in range(self.k // 3):
                rb = ConvolutionalResidualBlock(
                        self.conv_features_fixed, self.conv_features_fixed)
                modules.append(rb)
            modules.append(
                    nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

            modules.append(nn.Flatten())
            modules.append(nn.LayerNorm((16 // 4) * self.conv_features_fixed)) # 512

            modules.append(nn.Linear((16 // 4) * self.conv_features_fixed, # 512
                        self.linear_features_fixed))

            for i in range(self.k // 3):
                rb = LinearResidualBlock(
                        self.self.linear_features_fixed, self.self.linear_features_fixed)
                modules.append(rb)
            modules.append(
                    nn.Linear(self.linear_features_fixed, 2*self.latent_dim))
        else:
            raise AssertionError()

        self.encoder = nn.Sequential(*modules)

    def __init_decoder(self):
        modules = []
        modules.append(nn.Linear(self.latent_dim, self.linear_features_fixed))
        for i in range(self.k):
            rb = LinearResidualBlock(
                    self.linear_features_fixed, self.linear_features_fixed)
            modules.append(rb)
        modules.append(
            nn.Linear(self.linear_features_fixed,
                self.mc.out_rot_d * self.mc.output_njoints))
        #modules.append(nn.LeakyReLU())
        #modules.append(
        #    nn.Linear(self.mc.out_rot_d * self.mc.output_njoints,
        #        self.mc.out_rot_d * self.mc.output_njoints))
        self.decoder = nn.Sequential(*modules)

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(std)
        return e * std + mu

    def forward(self, x):
        """
            x: B x (nseq*nj*(xyz+inrot))
        """
        enc = self.encoder(x)
        mu = enc[..., :self.latent_dim]
        logvar = enc[..., self.latent_dim:]
        z = self.sample_z(mu, logvar)
        decoded = self.decoder(z)
        return {
            'recon': decoded,
            'mu': mu,
            'logvar': logvar
        }

    def compute_recon_loss(self, pred_r6d, outp_r6d):
        if len(pred_r6d.shape) == 2:
            pred_r6d = pred_r6d[:, None]
            outp_r6d = outp_r6d[:, None]
        pred_cvt = self.convert_pred(pred_r6d)
        outp_cvt = self.convert_outp(outp_r6d)
        loss, _ = self.criterion(pred_cvt, outp_cvt)
        return loss

    def compute_loss(self, gt, recon, mu, logvar):
        recon_loss = self.compute_recon_loss(gt, recon)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1),
                dim=0)

        loss = recon_loss + self.beta * kld_loss
        return loss

    def _training_init_data(self):
        return {}

    def _testing_init_data(self):
        return {}

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get, is_training=True):
        # Merge batch and frames
        assert inp.shape[1] == self.seq_len
        inp = inp.reshape((-1, inp.shape[-1]))
        outp = outp.reshape((-1, outp.shape[-1]))

        if not self.input_fullpose:
            d = self(inp)
        else:
            d = self(outp)
        loss = self.compute_loss(gt=outp, recon=d['recon'], mu=d['mu'], logvar=d['logvar'])

        if is_training:
            loss.backward()

        return (loss.item(), {})

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get):
        return self._training_step_callback(inp, outp, meta, prev_step_custom_data,
                train_init_data, train_get, is_training=False)

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        # Merge batch and frames
        inp_r = inp.reshape((-1, inp.shape[-1]))
        outp_r = outp.reshape((-1, outp.shape[-1]))

        if not self.input_fullpose:
            d = self(inp_r)
        else:
            d = self(outp_r)
        return ((d['recon'].reshape(outp.shape), outp), {})


class MyFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MyUnflatten(nn.Module):
    def forward(self, input, size=1024):
        #return input.view(input.size(0), size, 1, 1)
        return input.view(input.size(0), 32, 6)


# https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
class RefVAENN(BaseNN):
    def __init__(self, config):
        super().__init__(config=config)
        mc = config['model_config']
        h_dim = 192
        z_dim = mc['latent_dim']
        self.seq_len = self.mc['input_seq_length']
        in_njoints = self.mc['encoder_njoints']
        if in_njoints == 3:
            self.input_fullpose = False
        elif in_njoints == 22:
            self.input_fullpose = True
        self.beta = self.mc['betavae_beta']

        in_size = self.seq_len * in_njoints * \
                  ((0 if self.input_fullpose else 3) + self.mc.in_rot_d)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            MyFlatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            MyUnflatten(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 4, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 1, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Linear(148, 132),
            #nn.Sigmoid(),
        )
        self.device = get_device()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return {
            'recon': recon,
            'mu': mu,
            'logvar': logvar
        }

    def compute_recon_loss(self, pred_r6d, outp_r6d):
        if len(pred_r6d.shape) == 2:
            pred_r6d = pred_r6d[:, None]
            outp_r6d = outp_r6d[:, None]
        pred_cvt = self.convert_pred(pred_r6d)
        outp_cvt = self.convert_outp(outp_r6d)
        loss, _ = self.criterion(pred_cvt, outp_cvt)
        #loss = nn.functional.mse_loss(pred_r6d, outp_r6d)
        return loss

    def compute_loss(self, gt, recon, mu, logvar):
        recon_loss = self.compute_recon_loss(gt, recon)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1),
                dim=0)

        loss = recon_loss + self.beta * kld_loss
        return loss

    def _training_init_data(self):
        return {}

    def _testing_init_data(self):
        return {}

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get, is_training=True):
        # Merge batch and frames
        assert inp.shape[1] == self.seq_len
        inp = inp.reshape((-1, inp.shape[-1]))
        outp = outp.reshape((-1, outp.shape[-1]))
        # Conv1d expects the form (NB, Cin, L)
        # We assume input channel size 1
        inp = inp[:, None]
        outp = outp[:, None]

        if not self.input_fullpose:
            d = self(inp)
        else:
            #d = self(outp.clone())
            d = self(outp)
        loss = self.compute_loss(gt=outp, recon=d['recon'], mu=d['mu'], logvar=d['logvar'])

        if is_training:
            loss.backward()

        return (loss.item(), {})

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get):
        return self._training_step_callback(inp, outp, meta, prev_step_custom_data,
                train_init_data, train_get, is_training=False)

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        # Merge batch and frames
        inp_r = inp.reshape((-1, inp.shape[-1]))
        outp_r = outp.reshape((-1, outp.shape[-1]))
        inp_r = inp_r[:, None]
        outp_r = outp_r[:, None]

        if not self.input_fullpose:
            d = self(inp_r)
        else:
            d = self(outp_r)
        return ((d['recon'].reshape(outp.shape), outp), {})


class FlowBasedNN(BaseNN):
    def __init__(self, config):
        super().__init__(config=config)
        self.mc = config['model_config']

        self.seq_len = 1
        in_njoints = self.mc['encoder_njoints']
        if in_njoints == 3:
            self.input_fullpose = False
            raise NotImplementedError("TODO implement transformer predictor")
        elif in_njoints == 22:
            self.input_fullpose = True
        else:
            raise ValueError("Invalid #inputjoints: {}".format(in_njoints))

        in_size = self.seq_len * in_njoints * \
                  ((0 if self.input_fullpose else 3) + self.mc.in_rot_d)

        coupling_topology = self.mc.get('rnvp_coupling_topology', None)
        bn = self.mc.get('rnvp_add_batch_normalisation', True)
        up = self.mc.get('rnvp_use_permutation', True)
        sf = self.mc.get('rnvp_use_single_functions', True)

        FLOW_N = 9 # Number of affine coupling layers
        self.rnvp = LinearRNVP(input_dim=in_size, coupling_topology=coupling_topology,
                flow_n=FLOW_N, batch_norm=bn,
                mask_type='odds', conditioning_size=0,
                use_permutation=up, single_function=sf)

        # etc
        #ndata = inps.shape[0]
        #model_samples = nf_model.sample(ndata).detach().numpy()
        #print("model samples:", model_samples.shape)

    def compute_recon_loss(self, pred_r6d, outp_r6d):
        if len(pred_r6d.shape) == 2:
            pred_r6d = pred_r6d[:, None]
            outp_r6d = outp_r6d[:, None]
        pred_cvt = self.convert_pred(pred_r6d)
        outp_cvt = self.convert_outp(outp_r6d)
        loss, _ = self.criterion(pred_cvt, outp_cvt)
        return loss

    def compute_loss(self, gt, recon, mu, logvar):
        recon_loss = self.compute_recon_loss(gt, recon)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1),
                dim=0)

        loss = recon_loss + self.beta * kld_loss
        return loss

    def _training_init_data(self):
        return {}

    def _testing_init_data(self):
        return {}

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get, is_training=True):
        # Merge batch and frames
        assert inp.shape[1] == self.seq_len
        inp = inp.reshape((-1, inp.shape[-1]))
        outp = outp.reshape((-1, outp.shape[-1]))

        if not self.input_fullpose:
            u, logdet = self(inp)
        else:
            u, logdet = self(outp)

        # Train via maximum likelihood
        prior_logprob = self.rnvp.logprob(u)
        log_prob = -torch.mean(prior_logprob.sum(1) + logdet)

        if is_training:
            log_prob.backward()

        return (log_prob.item(), {})

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get):
        return self._training_step_callback(inp, outp, meta, prev_step_custom_data,
                train_init_data, train_get, is_training=False)

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        # Merge batch and frames
        inp_r = inp.reshape((-1, inp.shape[-1]))
        outp_r = outp.reshape((-1, outp.shape[-1]))

        if not self.input_fullpose:
            d = self(inp_r)
        else:
            d = self(outp_r)
        return ((d['recon'].reshape(outp.shape), outp), {})

    def forward(self, x, y=None, return_step=False):
        return self.rnvp.forward(x, y, return_step)

    def backward(self, u, y=None, return_step=False):
        return self.rnvp.backward(u, y, return_step)

    def sample(self, samples=1, y=None, return_step=False, return_logdet=False):
        return self.rnvp.sample(samples, y, return_step, return_logdet)

def get_nn(config):
    model_config = config['model_config']
    nn_arch = model_config['nn_architecture']
    if nn_arch == 'RNN':
        return RNNBasedNN(config=config)
    elif nn_arch == 'VAE':
        return VAEBasedNN(config=config)
    elif nn_arch == 'VAE_REF':
        return RefVAENN(config=config)
    elif nn_arch == 'FLAG':
        return FlowBasedNN(config=config)
    else:
        raise ValueError("Unknown NN architecture: {}".format(nn_arch))


def init_model(config, model_config_path=None, append=False, cuda_index=0,
        tensorboard_logdir=None):
    if model_config_path:
        glg().info("Loading model config from %s%s...",
                model_config_path, " (a)" if append else "")
        if not append:
            mc = ModelConfig(None)
        else:
            mc = ModelConfig(config.get('model_config', None))
        mc.load_from_file(model_config_path, append=append)
    else:
        mc = ModelConfig(config.get('model_config', None))

    if not mc.get('name', None):
        if model_config_path:
            mc['name'] = os.path.splitext(os.path.basename(model_config_path))[0]
        else:
            mc['name'] = "#"

    global get_device
    get_device = functools.partial(get_device, cuda_index=cuda_index)
    set_get_device_func(get_device)

    if not tensorboard_logdir:
        tensorboard_logdir = "./allruns/{}".format(
                datetime.datetime.now().strftime("%Y%m%dT%_H%M%S"))
    mc['tensorboard_logdir'] = tensorboard_logdir

    config['model_config'] = mc


class Node:
    def __init__(self, parent=None):
        self.parent = parent

    def __repr__(self):
        return "Node : {}".format(self.parent)


cvt_nodes = {
    'rot_6d': Node(),
    'rot_mats': Node('rot_6d'),
    'joints': Node('rot_mats'),
    'vel': Node('joints'),
    'acc': Node('vel'),
    'joints_rel': Node('rot_mats'),
    'vel_rel': Node('joints_rel'),
    'avel': Node('rot_mats'),
    'aacc': Node('avel'),
    'davel': Node('rot_mats')
}


def find_ancestor_cvt_nodes(node, include_self=False):
    if not isinstance(node, str):
        raise ValueError
    ancestors = [node] if include_self else []
    cn = cvt_nodes[node].parent
    while cn is not None:
        ancestors.append(cn)
        cn = cvt_nodes[cn].parent

    return ancestors


def nn_out_converter(config, targets, training, **kw):
    """
        It is assumed that the output of the NN are 3d or 6d rotations for each joint.
        Expected shape: nb x nf x (outnj*6)
    """

    mc = config['model_config']
    #if 'model' in kw:
    #    model = kw['model']
    #else:
    #    model_path = get_model_path(config, 'smpl', 'male')
    #    model = load_smpl_model(model_path,
    #            as_class=kw.get('load_smpl_model_as_class', False))

    if mc.out_rot_d != 6:
        raise NotImplementedError

    r2fc = kw.get(
            'recompute_first_two_cols_of_6d_rot', True)

    lg = glg()
    comp_targets = [targets[0]]
    ct_ancestors = [set(find_ancestor_cvt_nodes(targets[0], include_self=True))]
    lg.debug("initial comp targets: %s", comp_targets)
    for t in targets[1:]:
        ta = set(find_ancestor_cvt_nodes(t, include_self=True))
        lg.debug("iter for '%s' (anc=%s)", t, ta)
        for ci, (ct, cta) in enumerate(zip(comp_targets, ct_ancestors)):
            if ta.issubset(cta):
                break
            elif cta.issubset(ta):
                comp_targets[ci] = t
                ct_ancestors[ci] = ta
                break
        else:
            comp_targets.append(t)
            ct_ancestors.append(ta)
    lg.debug("enditer comp targets: %s", comp_targets)
    #breakpoint()
    assert all('rot_6d' in cta.union({ct}) for ct, cta in zip(comp_targets, ct_ancestors))

    converters = []
    for ct in comp_targets:
        if ct == 'rot_6d':
            continue
        converters.append(get_converter(config, 'rot_6d', ct, return_intermediates=True,
                    model=kw.get('model', None), normalise_velocities=training))

    lg = glg()

    def closure(data):
        if len(data.shape) != 3:
            raise ValueError("Invalid data shape for data: {}".format(data.shape))

        batch_n_fr = tuple(data.shape[:2])
        reshp = batch_n_fr + (mc['output_njoints'], 3, 2)

        data_r6d = data.reshape(reshp)  # nb x nf x nj x 3 x 2
        conv_results = {
            'rot_6d': data_r6d
        }
        lg.debug("#conversions needed: %d", len(comp_targets))
        for cvt, ct in zip(converters, comp_targets):
            t0 = time.time()
            c, intm = cvt(data_r6d)
            conv_results[ct] = c
            conv_results = {**conv_results, **intm}
            lg.debug("Conversion from input to '%s' took %fs (also converted to: %s)",
                    ct, time.time() - t0, intm.keys())
        return conv_results

    return closure


def train(config, training_data_loader, validation_data_loader=None,
        checkpoints_save_dir=None, checkpoint_path=None, nepochs=None, window_sz=None):
    model_config = config['model_config']

    device = get_device()
    glg().info("Using device: %s", device)
    model = get_nn(config=config).to(device)
    glg().info("Model Summary:\n%s", str(model))
    glg().info("Start training...")
    model.start_training(training_data_loader,
            validation_data_loader=validation_data_loader,
            checkpoints_save_dir=checkpoints_save_dir,
            checkpoint_path=checkpoint_path,
            nepochs=nepochs,
            window_sz=window_sz)


def predict(config, test_data_loader, checkpoint_path, preserve_batches_shapes=False, return_torch=False):
    model_config = config['model_config']
    device = get_device()
    model = get_nn(config=config).to(device)
    if checkpoint_path:
        glg().info("Loading checkpoint: %s", checkpoint_path)
        load_torch_model(model, checkpoint_path)
    else:
        glg().info("No checkpoint provided!")

    if not preserve_batches_shapes:
        result = torch.Tensor().to(device)
    else:
        result = []

    model.eval()
    hidden_predict = None
    for d in tqdm(test_data_loader(batch=True, batch_size=model_config['batch_size'])):
        inp = totorch(d['input'], device=device)
        outp = totorch(d['output'], device=device)

        if model_config['input_prev_frame_pose'] and \
            not model_config['input_prev_frame_pose__use_gt']:
            raise NotImplementedError

        pred_X, hidden_predict = model(inp, hidden_predict)
        h_0, c_0 = hidden_predict
        hidden_predict = (h_0.detach(), c_0.detach())
        cur_pred = torch.squeeze(pred_X.detach(), dim=0)
        if not preserve_batches_shapes:
            result = torch.cat([result, cur_pred], dim=0)
        else:
            result.append(cur_pred)

    if not preserve_batches_shapes:
        if return_torch:
            return result
        else:
            return result.cpu().numpy()
    else:
        if return_torch:
            return result
        else:
            return [r.cpu().numpy() for r in result]


TEST_RESULTS_RAW_FN = "test_results.pkl"


def test(config, test_data_loader, checkpoint_path, results_save_dir=None, window_sz=None,
        inference_callback=None, **kw):
    mc = config['model_config']
    device = get_device()
    glg().info("Using device: %s", device)
    model = get_nn(config=config).to(device)
    glg().info("Model Summary:\n%s", str(model))
    glg().info("Start testing...")
    if checkpoint_path:
        glg().info("Loading checkpoint: %s", checkpoint_path)
        load_torch_model(model, checkpoint_path)
    else:
        glg().info("!!! No checkpoint provided !!!")
    model.start_testing(test_data_loader, results_save_dir=results_save_dir,
            window_sz=None, inference_callback=inference_callback, **kw)


def analyse_losses(losses, meta, save_dir, model_config=None):
    def fmtdf(_df):
        return _df.to_html()

    losses_cols = list(losses.keys())
    meta_cols = list(meta.keys())
    losses_datas = [losses[c] for c in losses_cols]
    meta_datas = [meta[c] for c in meta_cols]
    df = pd.DataFrame(list(zip(*(losses_datas + meta_datas))),
            columns=losses_cols + meta_cols)

    os.makedirs(save_dir, exist_ok=True)
    analysis_html_fp = os.path.join(save_dir, "analysis.html")
    analysis_res_dir_rel = "res/"
    analysis_res_dir = os.path.join(save_dir, analysis_res_dir_rel)
    os.makedirs(analysis_res_dir, exist_ok=True)

    figw = 12

    with open(analysis_html_fp, 'w') as text_writer:
        text_writer.write("Analysis date and time: {} <br>".format(
                    datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
        text_writer.write("Model Config: {} <br>".format(model_config))

        text_writer.write("<h1>Summary</h1>")
        text_writer.write(fmtdf(df.describe()))

        fig = plt.figure(figsize=(figw, round(figw * 2/3)))
        ax = fig.add_subplot()
        ax.boxplot(df[losses_cols], showfliers=False)
        ax.set_xticks(range(1, len(losses_cols)+1), losses_cols, rotation=10)
        fig_fp = os.path.join(analysis_res_dir, "summary.jpg")
        fig.savefig(fig_fp)
        
        text_writer.write("<img src=\"{}\"> <br>".format(
                    os.path.join(analysis_res_dir_rel, os.path.basename(fig_fp))))

        for m in tqdm(meta_cols):
            glg().info("Meta = %s", m)
            text_writer.write("<h2>Group by: \"{}\"</h2>".format(m))
            gb = df.groupby(m)
            text_writer.write(fmtdf(gb.mean()))
            g_desc = gb.describe()
            
            m_vals = df[m].unique()
            grps = [gb.get_group(v) for v in m_vals]

            plots_ncols = 4
            plots_nrows = int(math.ceil(len(losses_cols) / plots_ncols))
            fig, axes = plt.subplots(plots_nrows, plots_ncols,
                    figsize=(figw, figw * (plots_nrows / plots_ncols)))

            for i_l, ll in tqdm(enumerate(losses_cols), desc="Plots"):
                if plots_ncols == 1:
                    ax = axes[i_l]
                else:
                    ax = axes[i_l // plots_ncols, i_l % plots_ncols]
                ax.boxplot([g[ll] for g in grps], showfliers=False)
                ax.set_xticks(range(1, len(m_vals)+1), m_vals, rotation=90)
                ax.set_xlabel(m)
                ax.set_ylabel(ll)
            fig.tight_layout()
            fig_fp = os.path.join(analysis_res_dir, "meta={}_tables.jpg".format(m))
            fig.savefig(fig_fp)#, bbox_inches='tight')mc['output_njoints'])

            text_writer.write("<img src=\"{}\"> <br>".format(
                        os.path.join(analysis_res_dir_rel, os.path.basename(fig_fp))))

            for ll in tqdm(losses_cols, desc="Tables"):
                text_writer.write("<h4>{}</h4>".format(ll))
                text_writer.write(fmtdf(g_desc[ll]))
                text_writer.write("<br>")

            text_writer.write("<br><br>")


def analyse_test_results(test_result_paths, save_dir):
    import pandas as pd
    import matplotlib.pyplot as plt

    if not save_dir:
        raise ValueError("You must provide a valid directory in which to save results")
    else:
        os.makedirs(save_dir, exist_ok=True)

    all_losses = {}
    loss_types = OrderedDict()

    # Load all losses
    for itr, trfp in enumerate(test_result_paths):
        with open(trfp, 'rb') as f:
            test_result = pickle.load(f)

        result_lists = test_result['lists']
        name = test_result.get('model_config', {}).get('name',
                "TestResult_{}".format(itr+1))

        all_losses[name] = result_lists['losses']
        for loss_name in result_lists['losses'].keys():
            if loss_name not in loss_types:
                loss_types[loss_name] = loss_name

    # Analyse and write results to files
    html_fp = os.path.join(save_dir, "test_results_analysis.html")
    res_fn = "res/"
    res_fp = os.path.join(save_dir, res_fn)
    os.makedirs(res_fp, exist_ok=True)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()

    dataframes = {}
    with open(html_fp, 'w') as text_writer:
        for i_l, l_type in enumerate(loss_types):
            text_writer.write("<h2>{}</h2>".format(l_type))

            boxplt_inp = []
            boxplt_lbls = []
            for name, ll in all_losses.items():
                if l_type in ll:
                    boxplt_inp.append(ll[l_type])
                    boxplt_lbls.append(name)

            df = pd.DataFrame(list(zip(*boxplt_inp)), columns=boxplt_lbls)
            df_desc = df.describe()
            text_writer.write(df_desc.to_html())

            dataframes[l_type] = df
            dataframes[l_type + "_desc"] = df_desc

            ax.boxplot(boxplt_inp, showfliers=False)
            ax.set_xticks(range(1, len(boxplt_lbls)+1), boxplt_lbls, rotation=20)
            ax.set_xlabel("Test Results")
            ax.set_ylabel("Loss: {}".format(l_type))
            #ax.set_ylim([0, 0.015])

            graph_fn = "loss_{}.jpg".format(i_l+1)
            fig.savefig(os.path.join(res_fp, graph_fn))

            ax.clear()

            text_writer.write("<img src=\"{}\">".format(os.path.join(res_fn, graph_fn)))

    dfs_pkl_fp = os.path.join(save_dir, "test_results_analysis.pkl")
    with open(dfs_pkl_fp, 'wb') as f:
        pickle.dump(dataframes, f)


def model_output_joints_to_smpl_joints(outp_joints):
    p = np.zeros(tuple(outp_joints.shape[:-2]) + (24, 3), dtype=dtype_np)
    # Hands' orientations are the same as those of wrists
    p[..., :22, :] = outp_joints[..., :22, :]
    p[..., [22, 23], :] = p[..., [20, 21], :]
    return p


def model_output_to_smpl_poses(outp, recompute_first_two_cols_of_6d_rot=True):
    p = np.zeros(tuple(outp.shape[:-1]) + (24, 3), dtype=dtype_np)
    if outp.shape[-1] == 22*6:  # R[:, :2]
        r6d = outp.reshape((-1, 22, 3, 2))
        p[..., :22, :] = rot_mat_to_vec(
                rot_6d_to_mat(r6d, recompute_first_two_cols=recompute_first_two_cols_of_6d_rot))
    elif outp.shape[-1] == 22*3:  # Axis-angle repr
        p[..., :22, :] = outp.reshape((-1, 22, 3))
    else:
        raise NotImplementedError

    # Hands' orientations are the same as those of wrists
    p[..., [22, 23], :] = p[..., [20, 21], :]
    return p


def model_output_to_rot_mats(outp, recompute_first_two_cols_of_6d_rot=False):
    p = np.zeros(tuple(outp.shape[:-1]) + (24, 3, 3), dtype=dtype_np)
    if outp.shape[-1] == 22*6:  # R[:, :2]
        r6d = outp.reshape((-1, 22, 3, 2))
        p[..., :22, :, :] = rot_6d_to_mat(r6d, recompute_first_two_cols=recompute_first_two_cols_of_6d_rot)
    else:
        raise NotImplementedError

    # Hands' orientations are the same as those of wrists
    p[..., [22, 23], :, :] = p[..., [20, 21], :, :]
    return p
