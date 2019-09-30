# mllibs
いろいろな機械学習データセットのローダー

## 使い方

整備中ですが，大体この形で使えるようにすることを目指しています．
初回のみダウンロードや展開などが行われ，2回目からは高速に読み込めるようにします．
データの保存先は，各ソースが置かれている階層の `data` フォルダ内となります．

```
from datasets.[dataset_id].[dataset_id] import get_[dataset_id]
dataset = get_[dataset_id]()
```
