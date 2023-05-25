from data_loading import load_body_datas
import util

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import torch

import json
import subprocess
import os
import shutil
import traceback
import math
import functools
import time
import random
import glob


def test(ax):
    amass_fp = "Datasets/DFaust_67/50002/50002_chicken_wings_poses.npz"
    amass_fp = "Datasets/DFaust_67/50002/50002_jumping_jacks_poses.npz"
    #amass_fp = "Datasets_n/CNRS/SMPL-X G/CNRS/283/-01_L_1_stageii.npz"
    bdata = np.load(amass_fp)
    
    print("body data: {}".format(list(bdata.keys())))

    #model_fp = "SMPL/py/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl.npz"
    smpl_model_fp  = "SMPL/npz_models/basicModel_f_lbs_10_207_0_v1.0.0.npz"
    smplh_model_fp = "SMPL/mano_v1_2/models/SMPLH_male.pkl"
    smplx_model_fp = "SMPL/smplx_models/SMPLX_NEUTRAL.npz"

    print("poses:", bdata['poses'][0].shape[0] / 3)
    n_frms = bdata['poses'].shape[0]
    for frm in range(n_frms):
        model = SMPLModel(smplh_model_fp, k=51)
        #breakpoint()
        assert(bdata['poses'][frm].shape[0] == 3 * (model.k + 1))
        model.set_params(pose=bdata['poses'][frm, :3 * (model.k + 1)].reshape((-1, 3)), beta=bdata['betas'][:BETA_SZ], trans=bdata['trans'][frm, :])
        #model.set_params(pose=np.zeros_like(model.pose))
        #breakpoint()

        viz_model(model, ax=ax, posed=False, joints=model.pose_joints(), plt_show=False)

        if frm >= 0:
            plt.show(block=True)

        #time.sleep(0.5)


def test_loading_datas(config):
    print("Dittadi body datas: ")
    dittadi_bds = load_body_datas(config,
            body_data_filter=lambda bd: bd['dataset_shortname'] in ("CMU", "HDM05", "KIT"),
            json_save_path="body_data_dittadi.json", load_additional_datas=True)
    df_dittadi = pd.DataFrame(data=dittadi_bds)
    #print(df_dittadi.describe())
    print(df_dittadi.head())
    print(df_dittadi.groupby('dataset_shortname').count())

    print("\nWalking body datas: ")
    def walking_filt(bd_):
        pth = bd_['path'].lower()
        return 'walk' in pth or 'run' in pth
    walking_bds = load_body_datas(config, body_data_filter=walking_filt,
            json_save_path="body_data_walking.json", load_additional_datas=True)
    df_walking = pd.DataFrame(data=walking_bds)
    #print(df_walking.describe())
    print(df_walking.head())
    print(df_walking.groupby('dataset_shortname').count())

    breakpoint()


def test_preprocessing():
    dittadi_filt = lambda bd: bd['dataset_shortname'] in ("CMU", "HDM05", "KIT")
    bds = load_body_datas(body_data_filter=dittadi_filt, load_additional_datas=True,
            json_save_path="./body_data_a.json", prefer_cached=True)
    preprocess(body_datas=bds, normalise_shape=True, normalise_framerates=True,
            save_in_batch=True, n_body_datas_per_batch=64, sort_body_datas_by_length=True,
            njoints_cutoff=OUTPUT_NJOINTS, correct_rotation_amass=True, debug=True)


def test_preprocessed_data(path):
    with np.load(path) as d:
        print([k for k in d.keys()])

        b = 10
        f = 25

        pose = d['poses'][b, f]


        nframes = d['poses'][b].shape[0]
        poses = np.zeros((nframes, 24, 3))
        poses[:, :22, :] = d['poses'][b]
        poses[:, [22, 23], :] = poses[:, [20, 21], :]

        ani_d = {
                'type': 'smpl',
                'njoints': 24,#OUTPUT_NJOINTS,
                'poses': poses,
                'nframes': nframes,
                'shape': np.zeros(10)
        }
        ani_d['translations'] = np.zeros((ani_d['nframes'], 3))

        breakpoint()

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        play_animation(data_path="Datasets_n_anim/pre", fig=fig, ax=ax, data=ani_d,
                correct_root_rotation=False)


def preprocess_new(config):
    amass_root_dir = "./amass_datasets"
    #amass_root_dir = "./Datasets_n"
    dsets = {
        'train': ['KIT', 'CMU', 'BMLrub', 'HDM05'],
        'validation': ['HumanEva', 'BMLmovi', 'SOMA'],
        'test': ['DFaust', 'Transitions', 'SFU', 'EyesJapanDataset', 'ACCAD'],
        #'train': ['KIT', 'BMLrub'],
        #'validation': ['ACCAD'],
        #'test': ['CNRS'],
    }
    for dsetnames in dsets.values():
        assert(all(os.path.isdir(os.path.join(amass_root_dir, n)) for n in dsetnames))

    train_to_x_ratios = {
        'train': 1,
        # Take batch pruning into account
        'validation': 1,
        'test': 1,
        #'validation': 1 / 15,
        #'test': 1 / 10,
    }
    sampling_prob_modifier = {
        'train': 0.0,
        'validation': 0.0,
        'test': 0.1,
    }

    body_datas_save_dir = "./body_datas/1018"
    preprocessed_save_dir_base = "./batches/1008"
    # CUDA may OOM (during test set with current settings),
    # but using batch size of 128 resulted in poor training results for some reason
    batch_size = 256
    
    skip_sampling_if_dir_exists = False

    cui = 0

    if os.path.isdir(preprocessed_save_dir_base):
        print("preprocessed_save_dir_base ({}) exists! Removing...".format(
                    preprocessed_save_dir_base))
        shutil.rmtree(preprocessed_save_dir_base)
    
    # Generate body data jsons
    if not (skip_sampling_if_dir_exists and os.path.isdir(body_datas_save_dir)):
        if not os.path.isdir(body_datas_save_dir):
            os.makedirs(body_datas_save_dir, exist_ok=True)

        sd = int(time.time())
        random.seed(sd)
        print("Random seed: {}".format(sd))

        b2mib = lambda b: b / (1024**2)

        sizes = {}
        for dsettype, dsetnames in dsets.items():
            sz = 0
            for ds in dsetnames:
                sz += util.get_dirsize(os.path.join(amass_root_dir, ds),
                        exclude_filetypes=['.tar', '.bz2', 'gz2'])
            sizes[dsettype] = sz

        print("Sizes: {}".format(
                    {k:"{:.01f} MiB".format(b2mib(v))for k, v in sizes.items()}))

        for dsettype, dsetnames in dsets.items():
            print("##### Sampling {} - {}".format(dsettype, dsetnames))
            sampled_filesizes = []
            pct = train_to_x_ratios[dsettype] * sizes['train'] / sizes[dsettype]
            pct *= 1 + sampling_prob_modifier[dsettype]
            def filt(bd):
                if bd['dataset_shortname'] in dsets[dsettype]:
                    #filesz = os.path.getsize(bd['path'])
                    #do_sample =  random.random() < (filesz / sizes[dsettype])
                    do_sample = random.random() < pct
                    if do_sample:
                        filesz = os.path.getsize(bd['path'])
                        sampled_filesizes.append(filesz)
                    return do_sample

            bdatas = load_body_datas(config, body_data_filter=filt,
                    json_save_path=os.path.join(body_datas_save_dir, "body_datas_{}.json".format(dsettype)),
                    load_additional_datas=True)

            print("({}) {} files sampled ({:.02f} MiB, expected: {:.02f} MiB/{} = {:.02f} MiB)".format(
                        dsettype, len(sampled_filesizes), b2mib(sum(sampled_filesizes)),
                        b2mib(sizes['train']), 1/train_to_x_ratios[dsettype],
                        b2mib(sizes['train'] * train_to_x_ratios[dsettype])))
            print()
    else:
        print("body datas exist ({}). Sampling skipped.".format(body_datas_save_dir))

    # Preprocess
    assert all(os.path.isfile(os.path.join(body_datas_save_dir, fp)) for fp in os.listdir(body_datas_save_dir))
    for dsettype, dsetnames in dsets.items():
        print("##### Preprocess {} - {}".format(dsettype, dsetnames))
        bdatas_path = os.path.join(body_datas_save_dir, "body_datas_{}.json".format(dsettype))
        preprocessed_save_dir = os.path.join(preprocessed_save_dir_base, dsettype)
        cmd = [".venv/bin/python", "program.py", "preprocess",
            "--body_data", bdatas_path,
            "--preprocessed", preprocessed_save_dir,
            "--cuda_index", str(cui)]
        if batch_size is not None:
            cmd += ["--batch_size", str(batch_size)]
        subprocess.call(cmd)


def train_baselines(config):
    model_configs_dir = "ModelConfigs"
    train_outputs_dir_base = "checkpoints_all/baselines_0810"

    train_outputs_dir_base = os.path.join(train_outputs_dir_base, "train")
    datasets_dir = "batches/mine"
    train_datasets_dir = os.path.join(datasets_dir, 'train')
    val_datasets_dir = os.path.join(datasets_dir, 'validation')
    os.makedirs(train_outputs_dir_base, exist_ok=True)

    cui = 0
    # Configure via config.json
    shuffle = False
    shuffle_nwindows_cutoff = None  # 10
    batch_sz = 400
    train_batch_pruning = 0.92
    shuffle_windows_offsets = 8  # -1
    epochs = None  # 150
    
    shuffle_validation = True
    #valid_batch_pruning = 0.86

    #for model_config_fn in [None] + os.listdir(model_configs_dir):
    for model_config_fp in glob.glob(model_configs_dir + "/**/*.json", recursive=True):
        model_config_fn = os.path.basename(model_config_fp)
        if os.path.splitext(model_config_fn)[1] != ".json" or \
            model_config_fn.startswith("IGNORE"):
            continue
        print("#### Training {} ####".format(model_config_fn))
        if model_config_fn:
            train_outputs_dir = os.path.join(train_outputs_dir_base, os.path.splitext(model_config_fn)[0])
        else:
            model_config_fp = None
            train_outputs_dir = os.path.join(train_outputs_dir_base, "default_options")

        os.makedirs(train_outputs_dir, exist_ok=True)

        cmd = [".venv/bin/python", "program.py", "train",
                "--preprocessed", train_datasets_dir,
                "--validation", val_datasets_dir,
                "--log_directory", os.path.join(train_outputs_dir, "logs"),
                "--checkpoints_save_dir", os.path.join(train_outputs_dir, "checkpoints"),
                "--cuda_index", str(cui),
                "--prune_batch", str(train_batch_pruning)]
        if shuffle:
            cmd += ["--shuffle", "all"]
            cmd += ["--batch_size", str(batch_sz)]
            cmd += ["--shuffle_windows_offsets", str(shuffle_windows_offsets)]
            if shuffle_nwindows_cutoff is not None:
                cmd += ["--shuffle_n_windows_cutoff", str(shuffle_nwindows_cutoff)]
        if shuffle_validation:
            cmd += ["--shuffle_validation"]

        if epochs:
            cmd += ["--epochs", str(epochs)]

        if model_config_fp:
            cmd += ["--model_config", model_config_fp]

        try:
            subprocess.call(cmd)
        except KeyboardInterrupt:
            exit(1)
        except:
            traceback.print_exc()


def train_vaes(config):
    model_configs_dir = "ModelConfigs-VAE"
    train_outputs_dir_base = "checkpoints_all/vae_1007"

    train_outputs_dir_base = os.path.join(train_outputs_dir_base, "train")
    datasets_dir = "batches/dittadi_400"
    train_datasets_dir = os.path.join(datasets_dir, 'train')
    val_datasets_dir = os.path.join(datasets_dir, 'validation')
    os.makedirs(train_outputs_dir_base, exist_ok=True)

    cui = 0

    shuffle = True

    nseq = 1
    nwinds_randomise = True
    batch_sz = 64

    if nseq == 1:
        nwinds = 10
    elif nseq == 16:
        nwinds = 150
    else:
        raise ValueError("Invalid nseq: {}".format(nseq))


    #for model_config_fn in [None] + os.listdir(model_configs_dir):
    for model_config_fp in glob.glob(model_configs_dir + "/**/*.json", recursive=True):
        model_config_fn = os.path.basename(model_config_fp)
        if model_config_fn.startswith("IGNORE"): continue
        print("#### Training {} ####".format(model_config_fn))
        if model_config_fn:
            train_outputs_dir = os.path.join(train_outputs_dir_base, os.path.splitext(model_config_fn)[0])
        else:
            model_config_fp = None
            train_outputs_dir = os.path.join(train_outputs_dir_base, "default_options")

        os.makedirs(train_outputs_dir, exist_ok=True)

        cmd = [".venv/bin/python", "program.py", "train",
                "--preprocessed", train_datasets_dir,
                "--validation", val_datasets_dir,
                "--log_directory", os.path.join(train_outputs_dir, "logs"),
                "--checkpoints_save_dir", os.path.join(train_outputs_dir, "checkpoints"),
                "--cuda_index", str(cui)]

        if shuffle:
            cmd += ["--shuffle", "all",
                    "--shuffle_n_windows_cutoff", str(nwinds)]
            if batch_sz is not None and batch_sz > 0:
                cmd += ["--batch_size", str(batch_sz)]
            if nwinds_randomise:
                cmd += ["--shuffle_n_windows_random"]

        if model_config_fp:
            cmd += ["--model_config", model_config_fp]

        try:
            subprocess.call(cmd)
        except KeyboardInterrupt:
            exit(1)
        except:
            traceback.print_exc()


def test_baselines(config):
    model_configs_dir = "ModelConfigs"
    test_datasets_dir = "batches/mine"
    basedir = "checkpoints_all/baselines_0809"

    test_datasets_dir = os.path.join(test_datasets_dir, "test")
    test_outputs_dir_base = os.path.join(basedir, "test/individual")
    compar_testing_outputs_dir = os.path.join(basedir, "test/comparison")
    train_outputs_dir_base = os.path.join(basedir, "train")
    os.makedirs(test_outputs_dir_base, exist_ok=True)
    os.makedirs(compar_testing_outputs_dir, exist_ok=True)

    PYTHON = ".venv/bin/python"
    cui = 0

    shuffle = True
    
    # Individual tests
    for train_fn in os.listdir(train_outputs_dir_base):
        train_fp = os.path.join(train_outputs_dir_base, train_fn)
        print("#### Testing {} ####".format(train_fn))
        if train_fn == "default_options":
            continue
        train_dir = os.path.join(train_outputs_dir_base, train_fn)
        if not os.path.isdir(train_dir):
            continue
        test_outputs_dir = os.path.join(test_outputs_dir_base, train_fn)
        os.makedirs(test_outputs_dir, exist_ok=True)
        try:
            chkpt_fp = os.path.join(train_dir,
                next(iter(reversed(list(filter(lambda f: os.path.splitext(f)[1] == ".chkpt",
                            os.listdir(train_dir)))))))
        except:
            print("Checkpoint not found in: {}".format(train_dir))
            exit(1)
        try:
            model_config_fp = next(iter(filter(
                    lambda f: os.path.basename(f) == train_fn + ".json",
                    glob.glob(model_configs_dir + "/**/*.json", recursive=True))))
        except:
            print("Model config not found in: {}".format(model_configs_dir))
            exit(1)

        cmd = [PYTHON, "program.py", "test",
               "--preprocessed", test_datasets_dir,
               "--log_directory", os.path.join(test_outputs_dir, "logs"),
               "--checkpoint", chkpt_fp,
               "--test_results_directory", test_outputs_dir,
               "--model_config", model_config_fp,
               "--cuda_index", str(cui)]
        if shuffle:
            cmd += ["--shuffle_test"]

        subprocess.call(cmd)

    # Comparison test
    cmd = [PYTHON, "program.py", "test", "--compare_test_results",
           "--test_results_directory", test_outputs_dir_base,
           "--comparison_test_results_directory", compar_testing_outputs_dir,
           "--log_directory", compar_testing_outputs_dir]
    subprocess.call(cmd)


def test_baselines_qualitative(config):
    model_configs_dir = "ModelConfigs"
    qual_datasets_dir = "batches/mine"
    basedir = "checkpoints_all/baselines_0810"

    qual_datasets_dir = os.path.join(qual_datasets_dir, "qualitative")
    test_outputs_dir_base = os.path.join(basedir, "test/individual")
    train_outputs_dir_base = os.path.join(basedir, "train")
    os.makedirs(test_outputs_dir_base, exist_ok=True)

    PYTHON = ".venv/bin/python"
    cui = 0
    
    # Individual tests
    for train_fn in os.listdir(train_outputs_dir_base):
        train_fp = os.path.join(train_outputs_dir_base, train_fn)
        if train_fn == "default_options":
            continue
        train_dir = os.path.join(train_outputs_dir_base, train_fn)
        if not os.path.isdir(train_dir):
            continue
        test_outputs_dir = os.path.join(test_outputs_dir_base, train_fn, "qualitative")
        quantitative_outputs_dir = os.path.join(test_outputs_dir_base, train_fn,
                "quantitative_for_qualdset")
        os.makedirs(test_outputs_dir, exist_ok=True)
        try:
            chkpt_fp = os.path.join(train_dir,
                    next(iter(reversed(list(filter(lambda f: os.path.splitext(f)[1] == ".chkpt",
                                os.listdir(train_dir)))))))
        except:
            print("Checkpoint not found in: {}".format(train_dir))
            exit(1)
        try:
            model_config_fp = next(iter(filter(
                    lambda f: os.path.basename(f) == train_fn + ".json",
                    glob.glob(model_configs_dir + "/**/*.json", recursive=True))))
        except:
            print("Model config not found in: {}".format(model_configs_dir))
            print("Continuing execution...")
            continue

        cmd = [PYTHON, "program.py", "test",
               "--log_directory", os.path.join(test_outputs_dir, "logs"),
               "--checkpoint", chkpt_fp,
               "--amass", qual_datasets_dir,
               "--save_as_video", test_outputs_dir, str(2),
               "--test_results_directory", quantitative_outputs_dir,
               "--model_config", model_config_fp,
               "--cuda_index", str(cui)]
        subprocess.call(cmd)

        #print("Running quantitative tests...")
        #cmd = [PYTHON, "program.py", "test",
        #       "--log_directory", os.path.join(test_outputs_dir, "logs"),
        #       "--checkpoint", chkpt_fp,
        #       "--amass", qual_datasets_dir,
        #       "--test_quantitatively",
        #       "--test_results_directory", quantitative_outputs_dir,
        #       "--model_config", model_config_fp]
        #subprocess.call(cmd)


def move_qualitative_baseline_results(config):
    basedir = "checkpoints_all/baselines_0810"
    test_outputs_dir_base = os.path.join(basedir, "test/individual")
    dest_base_dir = os.path.join(basedir, "test/qualitative")
    os.makedirs(dest_base_dir, exist_ok=True)
    train_outputs_dir_base = os.path.join(basedir, "train")

    for train_fn in os.listdir(train_outputs_dir_base):
        if train_fn == "default_options":
            continue
        test_outputs_dir = os.path.join(test_outputs_dir_base, train_fn, "qualitative")
        for root, dirs, files in os.walk(test_outputs_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                dest_dir = os.path.splitext(os.path.join(
                            dest_base_dir, os.path.relpath(fp, test_outputs_dir)))[0]
                os.makedirs(dest_dir, exist_ok=True)
                target_fn = "{}__{}".format(train_fn, fn)
                shutil.copy(fp, os.path.join(dest_dir, target_fn))


def test_rotations(**kw):
    th = math.pi / 3
    rm1 = np.array([[math.cos(th), -math.sin(th), 0],
            [math.sin(th), math.cos(th), 0],
            [0, 0, 1]])
    rm2 = rm1.copy()
    rm2[:, 2] = -rm2[:, 2]

    print("rm1:\n{}\nrm2:\n{}\n".format(rm1, rm2))

    pt = np.array([[1, 0, 1]]).T
    print("pt:\n{}".format(pt))

    pt_rm1 = rm1 @ pt
    pt_rm2 = rm2 @ pt

    print("pt tfm by rm1:\n{}\npt tfm by rm2:\n{}".format(pt_rm1, pt_rm2))

    cr1 = np.cross(rm1[:, 0], rm1[:, 1])

    print("cross: {}".format(cr1))

    i1 = rm1 @ rm1.T
    i2 = rm2 @ rm2.T

    print("i1:\n{}\ni2:\n{}".format(i1, i2))

    breakpoint()


def test_rotation_3rd_cols(config):
    from data_loading import Data2
    from preprocessing import preprocessed_to_model_inout

    preprocessed_dir = "Datasets_n_preprocessed_batch/batches_20220804T162228"
    for i, fn in enumerate(os.listdir(preprocessed_dir)):
        print("batch {}".format(i+1))
        fp = os.path.join(preprocessed_dir, fn)
        with np.load(fp) as batch:
            rmats = batch['rot_mats']
            cr12 = np.cross(rmats[..., 0], rmats[..., 1])
            diff_plus = rmats[..., 2] - cr12
            diff_minus = rmats[..., 2] - (-cr12)

            print("diff plus sum: {}".format(np.sum(diff_plus)))  # -> ~0
            print("diff plus mean: {}".format(np.mean(diff_plus)))
            print("diff minus sum: {}".format(np.sum(diff_minus)))
            print("diff minus mean: {}".format(np.mean(diff_minus)))
        print()

    # 3D rotation group: https://en.wikipedia.org/wiki/3D_rotation_group


def test_zero_pose(config):
    import viz
    from data_wrangling import compute_joints

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    zero_pose_b = torch.zeros([1, 24, 3], dtype=util.dtype_torch)

    # Official
    model_official = util.load_smpl_model(config=config, as_class=True)
    j = compute_joints(config, model=model_official, pose=zero_pose_b[0].numpy())
    height_o = np.max(j[:, 1]) - np.min(j[:, 1])

    viz.viz_model(model_official, ax=ax, plt_title="Official", joints=j, plt_show=False)

    # Batch
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    model_batch = util.load_smpl_model(config=config, as_class=False)
    j = compute_joints(config, model=model_batch, pose=zero_pose_b)[0]
    height_b = np.max(j[:, 1]) - np.min(j[:, 1])

    viz.viz_model(model_official, ax=ax, joints=j, plt_title="Batch Processing", plt_show=False)

    print("Height (official): {}".format(height_o))
    print("Height (batch): {}".format(height_b))

    plt.show()

    breakpoint()


def eval_samples_rotated(config):
    # Path containing AMASS files to eval
    qual_datasets_dir = "batches/qualitative_t"
    outputs_dir_base = "checkpoints_all/temp"

    mcname = "mse"
    chkpt_dir_or_fp = "checkpoints_all/baselines_0809/train/{}".format(mcname)
    model_config_fp = "ModelConfigs/{}.json".format(mcname)
    outputdirlvl = 1

    rots = range(45, 360, 45)

    if not os.path.isdir(outputs_dir_base):
        os.makedirs(outputs_dir, exist_ok=True)

    if os.path.isfile(chkpt_dir_or_fp):
        chkpt_fp = chkpt_dir_or_fp
    elif os.path.isdir(chkpt_dir_or_fp):
        chkpt_fp = os.path.join(chkpt_dir_or_fp,
                    next(iter(filter(lambda f: os.path.splitext(f)[1] == ".chkpt",
                                os.listdir(chkpt_dir_or_fp)))))
    else:
        raise ValueError("Checkpoint not found: {}".format(chkpt_dir_or_fp))
    
    for r in rots:
        print("### Rot: {}".format(r))
        outputs_dir = os.path.join(outputs_dir_base, "{:03}deg".format(r))
        cmd = [".venv/bin/python", "program.py", "test",
               "--checkpoint", chkpt_fp,
               "--amass", qual_datasets_dir,
               "--save_as_video", outputs_dir, str(outputdirlvl),
               #"--test_results_directory", test_outputs_dir,
               "--model_config", model_config_fp,
               "--rotate_y", str(r)]
        subprocess.call(cmd)


def test_rotation_propagation(config):
    from data_wrangling import compute_joints

    amass_fp = "Datasets_n/ACCAD/SMPLpH G/ACCAD/Female1General_c3d/A1 - Stand_poses.npz"
    with np.load(amass_fp) as bdata:
        pose = bdata['poses'][0].reshape((-1, 3))[:22]
        rmats = Rotation.from_rotvec(pose).as_matrix()
        print("rotmats: {}".format(rmats.shape))
        d = compute_joints(config, rot_mats=rmats[np.newaxis], return_dict=True)
        print(d.keys())
        rmats_g = d['rot_mats_g'][0]
        # (rel rot) = (abs rot of parent) @ (abs rot)
        # -> (abs rot) = (abs rot of parent)^-1 @ (rel rot)
        assert(math.isclose(0.0, np.sum(rmats[0] @ rmats[1], rmats_g[1])))
        breakpoint()
        print("This statement prevents above breakpoint from escaping current context")

