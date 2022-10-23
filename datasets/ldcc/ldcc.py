import os
import glob
import pandas as pd
import urllib.request
import tarfile
from pathlib import Path


_ldcc_label_names = [
    'kaden-channel',
    'peachy',
    'sports-watch',
    'dokujo-tsushin',
    'livedoor-homme',
    'topic-news',
    'it-life-hack',
    'movie-enter',
    'smax'
]


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


def load_ldcc(text_dir):
    """ text_dir (str): path for directory `text` """
    titles, texts, labels = [], [], []
    for i, l in enumerate(_ldcc_label_names):
        for file in glob.glob(str(Path(text_dir) / l / "{}*.txt".format(l))):
            with open(file, 'r') as f:
                lines = f.read().split('\n')
            titles.append(lines[2])
            texts.append('\n'.join(lines[3:]))
            labels.append(i)
    return pd.DataFrame(
        columns=['title', 'text', 'title_and_text', 'label'],
        data={
            'title':titles, 'text':texts, 'label':labels,
            'title_and_text': [ti + '\n' + te for ti, te in zip(titles, texts)]
        })


def get_ldcc():
    here = Path(os.path.dirname(os.path.abspath(__file__)))
    download_file(
        "https://www.rondhuit.com/download/ldcc-20140209.tar.gz",
        str(here / "data/ldcc-20140209.tar.gz")
    )
    # TODO: 展開を毎回する必要はない．
    extract_tar_gz(str(here / "data/ldcc-20140209.tar.gz"))
    return load_ldcc(str(here / "data/text/"))


def get_ldcc_labels():
    return _ldcc_label_names

