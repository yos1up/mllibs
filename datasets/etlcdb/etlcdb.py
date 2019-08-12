import os
from pathlib import Path
from urllib import request
import zipfile
import numpy as np
from chainer import datasets


def get_etl4c(ndim=3):
    """
    ひらがな 48 種のグレースケール画像データセット，ETL4C データセットを取得します．
    --------
    Args:
        ndim (int):
            これの値に応じて，入力画像の shape が以下のように変わります．
            1:(5328,),   2:(74, 72),   3:(1, 74, 72)
    Returns:
        (chainer.datasets.TupleDataset)
    """
    assert(1 <= ndim <= 3)
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = Path(here) / "data"
    if not os.path.exists(str(data_dir / "ETL4/ETL4C")):
        if not os.path.exists(str(data_dir / "ETL4.zip")):
            print("Downloading ETL4.zip...", end="")
            request.urlretrieve("http://etlcdb.db.aist.go.jp/etlcdb/data/ETL4.zip", str(data_dir / "ETL4.zip"))
            print("Done.")
        print("Extracting ETL4C...", end="")
        with zipfile.ZipFile(str(data_dir / "ETL4.zip")) as z:
            z.extract("ETL4/ETL4C", str(data_dir))
        print("Done.")

    img_offset = 288
    img_h = 74
    img_w = 72
    img_size = img_h * img_w // 2
    chunksize = img_offset + img_size
    xs, ys = [], []
    with open(str(data_dir / "ETL4/ETL4C"), "rb") as f:
        while True:
            s = f.read(chunksize)
            if s is None or len(s) < chunksize:
                break
            img = s[img_offset:img_offset + img_size]
            xs.append(np.array([[(b >> 4) & 15, b & 15] for b in img]).flatten().astype(np.float32)/16)
            decoded = _chr_jisx0201(s[9])
            ys.append(decoded)
    xs, ys = np.array(xs, dtype=np.float32), np.array(ys, dtype=str)

    if ndim == 2:
        xs = xs.reshape(-1, img_h, img_w)
    elif ndim == 3:
        xs = xs.reshape(-1, 1, img_h, img_w)

    label_to_id = {v: i for i, v in enumerate(get_etl4c_labels())}
    ys = np.array([label_to_id[y] for y in ys], dtype=np.int32)

    return datasets.TupleDataset(xs, ys)


def get_etl4c_labels():
    return list('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわゐゑをん')


def _chr_jisx0201(i):
    return "を*ゐ*ゑ******あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわん"[i-166]


if __name__ == '__main__':
    dataset = get_etl4c()
    print("Length of dataset:", len(dataset))
    print("shape of Input:", dataset[0][0].shape)

