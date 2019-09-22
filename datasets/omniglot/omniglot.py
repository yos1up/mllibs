import glob
import numpy as np
from pathlib import Path
from PIL import Image
from chainer import datasets


def get_omniglot(section, resize=None, ndim=2, verbose=False):
    """
    ----
    Args:
        section (str):
            ['train-small1', 'train-small2', 'train', 'eval', 'all'] のいずれか
            'all' は 'train' + 'eval' です．
            'train-small1' や 'train-small2' は 'train' の部分集合です．
        resize (None or tuple):
            画像のリサイズ指定．例えば Finn+ 2017 では (28, 28)．None の場合は (105, 105)
        ndim (int):
            各画像データの array の ndim 指定．[1, 2, 3] のいずれか．
    Returns:
        (chainer.datasets.TupleDataset)
            クラスラベルは 0 始まりの連続した int32 です
    """

    # TODO: git clone

    # TODO: unzip image folders

    raise NotImplementedError('git clone して，zip を展開してから，load_omniglot を使ってください')


def load_omniglot(section, git_root_dir, resize=None, ndim=2, verbose=False):
    """
    ----
    Args:
        section (str):
            ['train-small1', 'train-small2', 'train', 'eval', 'all'] のいずれか
            'all' は 'train' + 'eval' です．
            'train-small1' や 'train-small2' は 'train' の部分集合です．
        git_root_dir (str or Path):
            omniglot の公式リポジトリを clone したディレクトリを指定．
            あらかじめ，"python" フォルダ内の zip ファイルを全て展開しておく必要があります．
        resize (None or tuple):
            画像のリサイズ指定．例えば Finn+ 2017 では (28, 28)．None の場合は (105, 105)
        ndim (int):
            各画像データの array の ndim 指定．[1, 2, 3] のいずれか．
    Returns:
        (chainer.datasets.TupleDataset)
            クラスラベルは 0 始まりの連続した int32 です
    """
    assert section in ['train-small1', 'train-small2', 'train', 'eval', 'all']
    assert ndim in [1, 2, 3]
    images_dir = Path(git_root_dir) / "python" / {
        "train-small1": "images_background_small1",
        "train-small2": "images_background_small2",
        "train": "images_background",
        "eval": "images_evaluation",
        "all": "images_*[dn]",
    }[section]

    class_dirs = sorted(glob.glob(str(images_dir / "*/*/")))
    if verbose:
        print('{} classes found.'.format(len(class_dirs)))

    Xs, ys = [], []
    for i, cl in enumerate(class_dirs):
        pngs = sorted(glob.glob(str(Path(cl) / "*.png")))
        for p in pngs:
            img = Image.open(p, 'r')
            if resize is not None:
                img = img.resize(resize, resample=Image.LANCZOS)

            # オリジナルは bool 形式かつ「白地に黒」
            # logical_not するのは「黒地に白」に色反転するため
            ary = np.logical_not(np.array(img))
            if ndim == 3:
                ary = ary[None, :, :]
            elif ndim == 1:
                ary = ary.reshape(-1)

            Xs.append(ary)
            ys.append(i)

    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.int32)
    return datasets.TupleDataset(Xs, ys)


if __name__ == '__main__':
    print('Loading dataset...')
    tds = load_omniglot('train', './data/omniglot/', resize=(28, 28), ndim=3)
    print('Done.')
    print(len(tds))
    print(tds[0][0].shape)
    print(tds[0][0].dtype)
    print(tds[0][1])
    print(tds[0][1].dtype)
