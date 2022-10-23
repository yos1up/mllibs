import os
import glob
import pandas as pd
import urllib.request
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np
import sys, traceback

_notmnist_label_names = list("ABCDEFGHIJ")


def get_error_message(sys_exc_info=None):
    ex, ms, tb = sys.exc_info() if sys_exc_info is None else sys_exc_info
    return '[Error]\n' + str(ex) + '\n' + str(ms)


def download_file(url, filepath, skip_if_exists=True):
    """  url (str),  filepath (str)"""
    if skip_if_exists and os.path.exists(filepath):
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    urllib.request.urlretrieve(url, filepath)


def extract_tar_gz(tar_gz_path):    
    with tarfile.open(tar_gz_path) as t:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t, os.path.dirname(tar_gz_path))


def load_notmnist(notmnist_dir):
    """ notmnist_dir (str): path for directory `notMNIST_*****` """
    Xs, ys = [], []
    for i, l in enumerate(_notmnist_label_names):
        for file in glob.glob(str(Path(notmnist_dir) / l / "*.png")):
            try:
                img = Image.open(file, 'r')
            except:
                print(get_error_message())
                continue
            ary = np.array(img)
            Xs.append(ary)
            ys.append(i)
    Xs, ys = np.asarray(Xs, dtype=np.float32) / 256, np.asarray(ys, dtype=np.int32)
    return Xs, ys


def get_notmnist(section="small"):
    """
    ----
    Args:
        section (str): "small" または "large"

    Returns:
        (Xs, ys)
            Xs (np.float32, (*, 28, 28))
            ys (np.int32, (*,))

            * は "small" では約 18723 になります．"large" では約 529114 になります．
            （稀に読み込み失敗する画像があり，それらは読み飛ばされます（エラーログが表示されます）．）
    注意点
        small の初回の読み込みは時間がかかります．2,3分かかります．
        large の初回の読み込みは非常に時間がかかります．1,2時間くらいかかります．
        初回の読み込みに成功すると，npz ファイルが生成され，次回からはこちらが読み込まれます．これは数秒程度です．
    """
    assert section in ["small", "large"]
    here = Path(os.path.dirname(os.path.abspath(__file__)))

    if os.path.exists(str(here / "data/notMNIST_{}.npz".format(section))):
        _ = np.load(str(here / "data/notMNIST_{}.npz".format(section)))
        Xs, ys = _['Xs'], _['ys']
    else:
        download_file(
            "http://yaroslavvb.com/upload/notMNIST/notMNIST_{}.tar.gz".format(section),
            str(here / "data/notMNIST_{}.tar.gz".format(section))
        )
        extract_tar_gz(str(here / "data/notMNIST_{}.tar.gz".format(section)))
        Xs, ys = load_notmnist(str(here / "data/notMNIST_{}/".format(section)))
        np.savez(str(here / "data/notMNIST_{}.npz".format(section)), Xs=Xs, ys=ys)

    return Xs, ys


def get_notmnist_labels():
    return _notmnist_label_names


if __name__ == '__main__':
    get_notmnist(section="small")
