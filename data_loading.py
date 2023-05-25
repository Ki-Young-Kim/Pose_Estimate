from util import glg, get_root_dir, load_smpl_model, dtype_torch, get_device

import numpy as np
import torch
from tqdm import tqdm

import json
import os
import random
import math
import functools
from collections import namedtuple


def init(cuda_index):
    global get_device
    get_device = functools.partial(get_device, cuda_index=cuda_index)


class Data2:
    """
    Data loading class that loads batches of continuous preprocessed data
    that is meant as a more efficient alternative to Data
    """

    def __init__(self, model_config, preprocessed_to_model_inout_func,
            training_data_dir=None, test_data_dir=None,
            training_datas=None, test_datas=None):
        self.train_pct = model_config['training_data_percentage']
        self.model_config = model_config
        self.preprocessed_to_model_inout_func = preprocessed_to_model_inout_func
        self.training_data_dir = training_data_dir
        self.test_data_dir = test_data_dir
        self.training_datas = training_datas or []
        self.test_datas = test_datas or []
        self.train_load_counter = 0

    def dataloader(self, train_or_test, shuffle_opts=None, **kw):
        is_training = train_or_test == 'train'
        if train_or_test == 'train':
            is_training = True
            data_dir = self.training_data_dir
            data_list = self.training_datas
        elif train_or_test == 'test':
            is_traning = False
            data_dir = self.test_data_dir
            data_list = self.test_datas
        else:
            raise ValueError()

        shuffle_opts_default = {
            'shuffle_batches': False,
            'shuffle_windows': False,
            'n_windows': None,
            'sample_windows_randomly': False,
            # This option will load all the batches into RAM once.
            # Use at own risk.
            'shuffle_everything': False,
            'shuffle_windows_offsets_every_n_epochs': 8 if train_or_test == 'train' else 0,
        }
        shuffle_opts = shuffle_opts or {}
        for k, v in shuffle_opts_default.items():
            if k not in shuffle_opts:
                shuffle_opts[k] = v

        window_sz = kw.get('window_sz', self.model_config['window_size'])
        if shuffle_opts['shuffle_everything']:
            glg().info("Loading dataset...")
            torch_dset = TorchDataset(data_dir,
              window_sz=window_sz,
              preprocessed_to_model_inout_func=self.preprocessed_to_model_inout_func,
              nwindows_cutoff_per_file=shuffle_opts['n_windows'],
              nwindows_cutoff_randomise=shuffle_opts['sample_windows_randomly'])
            def collate(d):
                return list(zip(*d))
            torch_dloader = torch.utils.data.DataLoader(torch_dset,
                    batch_size=kw.get('batch_sz', self.model_config['batch_size']),
                        collate_fn=collate, shuffle=True)
        else:
            torch_dset = None
            torch_dloader = None

        xxx = 33

        def load_data_closure(**kw2):
            """
                batch_pruning_thresh: value in [0, 1]. 1.0 to keep all (ie no pruning).
                                e.g., if value is 0.95, when the # of motions that have
                                ended exceeds (1 - 0.95) = 5%, the remaining frames for all
                                motions will be culled from the batch.
            """
            nonlocal window_sz
            window_sz = kw2.get('window_sz') or window_sz
            if is_training:
                self.train_load_counter += 1
            # Load from data dir
            if data_dir:
                if not shuffle_opts['shuffle_everything']:
                    data_dir_ls = os.listdir(data_dir)
                    if shuffle_opts['shuffle_batches']:
                        random.shuffle(data_dir_ls)
                    for i_file, batch_fn in enumerate(data_dir_ls):
                        batch_fp = os.path.join(data_dir, batch_fn)
                        glg().info("[%d/%d] loading batch: %s (ws=%d)",
                                i_file+1, len(data_dir_ls), batch_fp, window_sz or -1)
                        with np.load(batch_fp) as batch:
                            inp, outp, intm_vals = self.preprocessed_to_model_inout_func(batch,
                                    return_intermediate_values=True, train_or_test=train_or_test)
                            nfrms_max = intm_vals['max_frames']
                            batch_pruning_thresh = kw.get('batch_pruning_thresh', 0.95) or 0.95
                            if batch_pruning_thresh < 1.0:
                                nb = batch['poses'].shape[0]
                                batch_nframes = batch['nframes']
                                # Assumption: batch elements are sorted by length (shortest to longest)
                                #assert batch_nframes[0] <= batch_nframes[-1]  # Error may exist from framerate normalisation
                                cutoff_idx = math.floor((nb - 1) * (1 - batch_pruning_thresh))
                                nfrms_max_new = batch_nframes[cutoff_idx]
                                glg().debug(("Batch pruned by {:.00f}%: at index {}/{}, "
                                            "{} -> {} ({:.01f}% retrained)").format(
                                            100 * (1 - batch_pruning_thresh),
                                            cutoff_idx, nb, nfrms_max, nfrms_max_new,
                                            100 * nfrms_max_new / nfrms_max))
                                glg().debug(("nwindows (size={}): {} -> {} "
                                            "({:.01f}% windows retained)").format(
                                            window_sz, nfrms_max // window_sz,
                                            nfrms_max_new // window_sz,
                                            100 * ((nfrms_max_new // window_sz) / (nfrms_max // window_sz))))
                                nfrms_max = nfrms_max_new

                            nwindows = nfrms_max // window_sz

                            windows_inds = list(range(nwindows))
                            if shuffle_opts['shuffle_windows']:
                                windows_inds = random.shuffle(window_inds)
                            if shuffle_opts['n_windows'] is not None:
                                windows_inds = windows_inds[:shuffle_opts['n_windows']]

                            for i in windows_inds:
                                start_frm = i * window_sz
                                end_frm_p1 = start_frm + window_sz
                                if end_frm_p1 - 1 > nfrms_max - 1:
                                    continue

                                yield {
                                    'path': batch['paths'],
                                    'gender': batch['genders'],
                                    'nframes': nfrms_max,
                                    'input': inp[:, start_frm:end_frm_p1],
                                    'output': outp[:, start_frm:end_frm_p1],
                                    # TODO this might be problematic if windows are shuffled
                                    'file_idx': i_file,
                                    'intm_vals': intm_vals
                                }
                else:  # shuffle everything -> Use torch.nn.DataLoader
                    if is_training:
                        shufoffsets = shuffle_opts['shuffle_windows_offsets_every_n_epochs']
                        if (shufoffsets or 0) > 0 and (self.train_load_counter > 0 and
                                self.train_load_counter % shufoffsets == 0):
                            glg().info("shifting offsets... (shufoffsets={}, train_load_counter={})".format(shufoffsets, self.train_load_counter))
                            torch_dset.reload_shifted()
                    
                    batchsz = torch_dloader.batch_size
                    nbatches = len(torch_dloader)
                    ndata = batchsz * nbatches
                    glg().info(("({}) Start loading data from torch dataloader: "
                                "{}bx{}={} (ws={})").format(
                                train_or_test, batchsz, nbatches, ndata, window_sz))

                    for ibatch, tup in enumerate(tqdm(torch_dloader,
                                desc="{}bx{}={}".format(batchsz, nbatches, ndata))):
                        d = {}
                        for i in range(len(tup)):
                            d[torch_dset.keys[i]] = tup[i]
                        d['input'] = np.array(d['input'])
                        d['output'] = np.array(d['output'])
                        # Basically, there is no feasible way to keep track of files,
                        # so we just assign a unique value each time.
                        d['file_idx'] = ibatch
                        yield d
            # Load from data list
            for i_data, data in enumerate(data_list):
                inp, outp, intm_vals = self.preprocessed_to_model_inout_func(data,
                        return_intermediate_values=True, train_or_test=train_or_test)
                d = {
                    'path': data.get('path', "#{}".format(i_data+1)),
                    'gender': data.get('gender', 'male'),
                    'input': inp,
                    'output': outp,
                    'intm_vals': intm_vals
                }
                if 'max_frames' in intm_vals:
                    d['nframes'] = intm_vals['max_frames']
                yield d

        return load_data_closure


class TorchDataset(torch.utils.data.Dataset):
    keys = ['path', 'gender', 'nframes', 'input', 'output', 'file_idx']
    def __init__(self, batches_dir, window_sz, preprocessed_to_model_inout_func,
            nwindows_cutoff_per_file=None, nwindows_cutoff_randomise=True):
        # Create one large list
        self.batches_dir = batches_dir
        self.preprocessed_to_model_inout_func = preprocessed_to_model_inout_func
        self.window_sz = window_sz
        self.nwindows_cutoff_per_file = nwindows_cutoff_per_file
        self.nwindows_cutoff_randomise = nwindows_cutoff_randomise
        self.slices = []
        self.reload_shifted(shift=0)
        self.random_shifts = None

    def reload_shifted(self, shift=None):
        glg().info("Reloading all batches with window offsets %s",
                "randomly shifted" if shift is None else "shifted by {}".format(shift))
        self.slices.clear()
        filecounter = 0
        batch_fns = os.listdir(self.batches_dir)
        shiftval = shift
        window_sz = self.window_sz
        for ibfile, fn in enumerate(batch_fns):
            fp = os.path.join(self.batches_dir, fn)
            glg().info("Processing batch %d/%d:\n\t%s", ibfile+1, len(batch_fns), fp)
            with np.load(fp) as batch:
                inp_b, outp_b, intm_vals = self.preprocessed_to_model_inout_func(batch,
                        return_intermediate_values=True)
                nb = len(batch['poses'])
                frs_b = batch['nframes']
                paths = batch['paths']
                genders = batch['genders']
                slices_curbatch = []
                # Iterate within batch
                for ib in tqdm(list(range(nb)),
                        desc="B{}/{}".format(ibfile+1, len(batch_fns))):
                    nfrms = frs_b[ib]
                    pth = paths[ib]
                    gen = genders[ib]
                    nwinds = nfrms // window_sz
                    inp = inp_b[ib]
                    outp = outp_b[ib]
                    sls = []
                    cutoff = nwinds
                    if (self.nwindows_cutoff_per_file is not None and
                            not self.nwindows_cutoff_randomise):
                        cutoff = min(nwinds, self.nwindows_cutoff_per_file)
                    if shift is None:
                        shiftval = random.randint(0, window_sz - 1)
                    for iw in range(cutoff):
                        st = iw * window_sz + shiftval
                        edp1 = st + window_sz
                        if edp1 >= nfrms:
                            continue
                        sls.append(
                            (pth, gen, nfrms, inp[st:edp1], outp[st:edp1], filecounter))
                    if (self.nwindows_cutoff_per_file is not None
                            and self.nwindows_cutoff_randomise):
                        random.shuffle(sls)
                        sls = sls[:self.nwindows_cutoff_per_file]
                    slices_curbatch.extend(sls)
                    filecounter += 1
                glg().info("%d slices collected", len(slices_curbatch))
                self.slices.extend(slices_curbatch)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        return self.slices[idx]


def load_body_datas(config, root_dir=None, body_data_filter=None, load_additional_datas=False,
        json_save_path=None, prefer_cached=False):
    """
    body_data_filter: filter func that accepts bdata dict
    evaluated inside initial loop.
    load_additional_datas: if set to True, actual AMASS data files will be open for more metadata
    json_save_path: if set to a valid path, data will be saved to it,
    unless it already exists AND prefered_cached is True.
    prefer_cached: if json_save_path is specified and points to an existing file and
    this flag is set, the file will be loaded and returned instead.
    """
    if prefer_cached and json_save_path and os.path.isfile(json_save_path):
        glg().info("Loading cached body data from \"%s\"...", json_save_path)
        with open(json_save_path) as f:
            return json.load(f)

    root_dir = root_dir or get_root_dir(config, 'body_data')
    body_datas = []
    glg().info("Loading body data from: %s", root_dir)
    for dset_shortname in os.listdir(root_dir):
        glg().info(dset_shortname)
        dset_path = os.path.join(root_dir, dset_shortname)
        if not os.path.isdir(dset_path):
            continue

        for dset_type_dirname in os.listdir(dset_path):
            if dset_type_dirname.startswith("SMPLpH"):
                dset_type = 'smplh'
            elif dset_type_dirname.startswith("SMPL-X"):
                dset_type = 'smplx'
            else:
                raise ValueError("Unknown dataset type \"{}\" ({})".format(
                    dset_type_dirname, dset_path))
            dset_type_path = os.path.join(dset_path, dset_type_dirname)
            for subjects_dirs_cont_dirname in os.listdir(dset_type_path):
                subjects_dirs_cont_path = os.path.join(dset_type_path, subjects_dirs_cont_dirname)
                if not os.path.isdir(subjects_dirs_cont_path):
                    continue
                for subject in os.listdir(subjects_dirs_cont_path):
                    subject_dirpath = os.path.join(subjects_dirs_cont_path, subject)
                    if not os.path.isdir(subject_dirpath):
                        continue
                    for bdata_fn in os.listdir(subject_dirpath):
                        bdata_name, ext = os.path.splitext(bdata_fn)
                        if ext.lower() != '.npz':
                            continue

                        bdata_path = os.path.join(subject_dirpath, bdata_fn)

                        bdata = {
                            'dataset_type': dset_type,
                            'dataset_shortname': dset_shortname,
                            'subject': subject,
                            'path': bdata_path
                        }
                        if body_data_filter is None or body_data_filter(bdata):
                            body_datas.append(bdata)

    if load_additional_datas:
        glg().info("Loading additional data...")
        cull_list = []
        for ibdata_info, bdata_info in enumerate(tqdm(body_datas)):
            with np.load(bdata_info['path']) as bdata:
                if 'poses' in bdata:
                    bdata_info['n_frms'] = bdata['poses'].shape[0]
                    bdata_info['gender'] = str(bdata['gender'])
                    bdata_info['framerate'] = int(
                            bdata.get('mocap_framerate', bdata.get('mocap_frame_rate', None)))
                else:
                    cull_list.append(ibdata_info)
        body_datas = [e for i, e in enumerate(body_datas) if i not in cull_list]

    if json_save_path:
        with open(json_save_path, 'w') as f:
            json.dump(body_datas, f)

    glg().info("Done")

    return body_datas


def load_torch_model(model, checkpoint_path):
    if checkpoint_path:
        glg().info("Loading checkpoint: {}".format(checkpoint_path))
        state_d = torch.load(checkpoint_path, map_location=get_device())
        items = list(state_d.items())
        for k, v in items:
            if k.startswith("lstm."):
                state_d["rnn." + k[len("lstm."):]] = v
                del state_d[k]
            elif k.startswith("linear."):
                state_d["linear_out." + k[len("linear."):]] = v
                del state_d[k]
        model.load_state_dict(state_d)
    else:
        glg().info("No checkpoint given!")
