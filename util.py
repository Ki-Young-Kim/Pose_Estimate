import torch
import numpy as np

from abc import ABC
import json
import os
import logging
import sys
import datetime
import time


dtype_np = np.float64
dtype_torch = torch.float64

torch.set_default_dtype(dtype_torch)


class ConfigBase(ABC, dict):
    def __init__(self):
        super().__init__()

    def load_from_file(self, path, append=False):
        if not append:
            self.clear()
            self._set_default_values()
        with open(path, encoding='utf8') as f:
            d = json.load(f)
            self.load_from_dict(d)
        self._verify()

    def load_from_dict(self, d):
        for k, v in d.items():
            self[k] = v
        self._verify()

    def _require(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key not in self:
                raise ValueError("Required config \"{}\" not found".format(key))

    def _set_default(self, key, default_v):
        self[key] = default_v if key not in self else self[key]

    def _set_default_values(self):
        """
        Set all defaults via self._set_defaults here
        """
        pass
    
    def _verify(self):
        """
            Verify that the config values are valid
        """
        pass


class Config(ConfigBase):
    def __init__(self, path):
        """
        'body_models' must be in the form:
        "body_models": { 
          "smpl": {
            "male": "x.npz",
            "female": "y.npz"
          },
          "smplh": {...},
          "smplx": {...}
        }
        """
        super().__init__()

        if os.path.isfile(path):
            self.load_from_file(path)

        self._require('body_data_root_directory')
        self._require('body_models')

        self._set_default_values()

    def _set_default_values(self):
        self._set_default('root_directory_suffixes', {})

        self._set_default('do_train_visualised', False)
        self._set_default('checkpoints_directory', "./checkpoints")
        self._set_default('use_tkagg', False)


lg = None
logger_initialised = False


def init_logger(print_to_screen=True, save_to_file=True, save_dir=None, save_path=None, **kw):
    global lg
    global logger_initialised

    if logger_initialised:
        raise RuntimeError("Logger was already initialised")
    lg = logging.getLogger()
    lg.setLevel(level=logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=kw.get('stream_handler_logging_level', logging.INFO))
    formatter = logging.Formatter(fmt='%(message)s')
    stream_handler.setFormatter(formatter)
    lg.addHandler(stream_handler)

    if save_to_file:
        if not save_path:
            fn = datetime.datetime.now().strftime("%Y%m%dT%H%M%S.log")
            if save_dir:
                save_path = os.path.join(save_dir, fn)
            else:
                save_path = os.path.join("logs", fn)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        file_handler = logging.FileHandler(save_path)
        file_handler.setLevel(level=kw.get('file_handler_logging_level', logging.INFO))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        lg.addHandler(file_handler)

    logger_initialised = True

    return lg


def get_logger():
    if not logger_initialised:
        raise RuntimeError("Logger has not been initialised yet")
    return lg


glg = get_logger


def get_device(cuda_index=0):
    return torch.device("cuda:{}".format(cuda_index) if torch.cuda.is_available()
            else "cpu")


def get_model_path(config, model_type, gender):
    gender = {'m': 'male', 'f': 'female', 'n': 'neutral'}.get(gender.lower(), gender)
    return config['body_models'][model_type.lower()][gender]


def load_smpl_model(model_path=None, config=None, as_class=False, **kw):
    from smpl_np import SMPLModel

    njoints = kw.get('njoints', None)
    if not model_path:
        get_logger().info("load_smpl_model: Model path not provided; returning male SMPL model.")
        model_path = get_model_path(config, 'smpl', 'male')
        njoints = 24

    t0 = time.time()
    if not as_class:
        mdl = np.load(model_path, allow_pickle=True, encoding='latin1')
    else:
        mdl = SMPLModel(model_path, k=kw.get('k', njoints - 1))
    glg().debug("Loaded SMPL model (class=%s) in %fs", as_class, time.time() - t0)

    return mdl


def get_root_dir(config, t):
    body_datas_root_dir = os.path.abspath(config['body_data_root_directory'])
    default_suffixes = {'body_data': '',
            'preprocessing': '_preprocessed',
            'animation': '_anim'}
    suffixes = config.get('root_directory_suffixes', {})
    suffix = suffixes.get(t, default_suffixes[t])
    return os.path.join(os.path.dirname(body_datas_root_dir),
            os.path.basename(body_datas_root_dir) + suffix)


def printerr(*a, **kw):
    print(*a, file=sys.stderr, **kw)


def get_dirsize(start_path, exclude_filetypes=None):
    """
        https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python#1392549
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            if exclude_filetypes:
                _, ext = os.path.splitext(f)
                ext = ext.lower()
                if any(ext in excl for excl in exclude_filetypes):
                    continue
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def print_pd_table_for_libreoffice(df, floatformat='{:.2f}'):
    """
    This function prints a pandas DataFrame so that it can be pasted into
    LibreOffice Writer and then be automatically converted to table by selecting
    the pasted text, then selecting Table -> Convert -> Text to Table from the top menu
    and setting tabs as delimiter.

    MAKE SURE TO Paste Special -> Unformatted Text SO THAT THE TABS AREN'T CONVERTED
    TO SPACES WHEN PASTED
    """
    if floatformat is not None:
        import pandas as pd
        pd.options.display.float_format = '{:.2f}'.format
    dfstr = df.to_string()
    while "  " in dfstr:
        dfstr = dfstr.replace("  ", " ")
    dfstr = dfstr.replace(" ", "\t")
    print(dfstr)
    return dfstr
