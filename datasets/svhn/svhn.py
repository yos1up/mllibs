# 参考： https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn
import os
from pathlib import Path
from urllib import request
import tarfile
import numpy as np
from tqdm import tqdm
from chainer import datasets
import h5py
from pathlib import Path
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))

def get_svhn(fmt='TupleDataset', image_size=[70, 30], num=None):
    """
    SVHN データセット（多数桁のままの形式）を取得します．
    --------
    Args:
        fmt (str):
            返り値のフォーマット指定．Returns 参照．
        image_size ([w (int), h (int)])
            画像をリサイズする「横幅」と「縦幅」．すべてのデータの画像は，このサイズにリサイズされます．
        num (int or None, default None):
            最初の num データ分だけ読み込みを行うオプション，
    Returns:
        data_train, data_test:
            それぞれ訓練データとテストデータが返ります．各々のフォーマットは以下参照．
            - fmt == 'dict' の場合
                {
                    "xs": numpy.ndarray, shape==(*, 3, h, w), dtype==numpy.float32 (0-1 ranged)
                    "ys": list of str. (Example: ["42", "5", "3719", ...])
                    "digitStruct": 以下の要素からなる dict. 
                        "name": list of str: 各要素は，そのデータの画像ファイル名
                        "bbox": list of dict: 各要素は，そのデータの各桁のバウンディングボックス情報
                }
            - fmt == 'TupleDataset' の場合
                上記の xs と ys から作成した chainer.TupleDataset となります．
    """
    assert fmt in ['dict', 'TupleDataset']
    assert len(image_size) == 2
    if image_size[0] < image_size[1]:
        print('[WARNING] 横幅と縦幅の指定を逆にしていませんか？（image_size の第一要素が「横幅」，第二要素が「縦幅」です）')
    
    data_dir = Path(here) / "data"
    if not os.path.exists(str(data_dir / "train")):
        if not os.path.exists(str(data_dir / "train.tar.gz")):
            print("Downloading train.tar.gz...", end="")
            request.urlretrieve("http://ufldl.stanford.edu/housenumbers/train.tar.gz", str(data_dir / "train.tar.gz"))
            print("Done.")
        print("Extracting train.tar.gz...", end="")
        with tarfile.open(str(data_dir / "train.tar.gz")) as t:
            t.extractall(str(data_dir))
        print("Done.")
    if not os.path.exists(str(data_dir / "test")):
        if not os.path.exists(str(data_dir / "test.tar.gz")):
            print("Downloading test.tar.gz...", end="")
            request.urlretrieve("http://ufldl.stanford.edu/housenumbers/test.tar.gz", str(data_dir / "test.tar.gz"))
            print("Done.")
        print("Extracting test.tar.gz...", end="")
        with tarfile.open(str(data_dir / "test.tar.gz")) as t:
            t.extractall(str(data_dir))
        print("Done.")

    data_train = load_svhn(str(data_dir / "train"), fmt=fmt, image_size=image_size, num=num)
    data_test = load_svhn(str(data_dir / "test"), fmt=fmt, image_size=image_size, num=num)
    return data_train, data_test




def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])

def load_digitStruct(mat_filename, num=None):
    """
    Args:
        mat_fliename (str):
            digitStruct.mat ファイルのパス．

        num (int or None, default None):
            最初の num データ分だけ読み込みを行うオプション，
    """
    ret = {'name': [], 'bbox':[]}
    with h5py.File(mat_filename, 'r') as f:
        size = f['/digitStruct/name'].size
        if num is not None:
            size = min(size, num)
        for _i in tqdm(range(size)):
            ret['name'].append(get_name(_i, f))
            ret['bbox'].append(get_box_data(_i, f))
    return ret


def load_svhn(path, fmt='TupleDataset', image_size=[70, 30], num=None):
    assert fmt in ['TupleDataset', 'dict']
    print('Loading digitStruct.mat...')
    dsm = load_digitStruct(str(Path(path) / 'digitStruct.mat'), num=num)
    print('    Done.')
    print('Loading images...')
    xs = []
    for name in tqdm(dsm['name']):
        xs.append(np.array(Image.open(str(Path(path) / name)).resize(image_size)))
    print('    Done.')
    xs = np.asarray(xs, dtype=np.float32).transpose([0, 3, 1, 2]) / 256  # (batch, color, height, width)
    ys = [''.join(map(str, map(int, b['label']))) for b in dsm['bbox']]

    if fmt == 'dict':
        return {
            "xs": xs,
            "ys": ys,
            "digitStruct": dsm
        }
    elif fmt == 'TupleDataset':
        return datasets.TupleDataset(xs, ys)
    else:
        raise ValueError


if __name__ == '__main__':
    data_train, data_test = get_svhn(num=100)
    print("len(data_train) == {}".format(len(data_train)))
    print("len(data_test) == {}".format(len(data_test)))

