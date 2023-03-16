# 顔写真抽出モジュール

OpenCV を使って顔写真を生成する関数を提供します．

# Get started

```shell
pip install git+https://github.com/Yuu-Miino-NUE/crop_tools
```

# 使用例

```python
from crop_tools import crop_face, crop_face_dir

# For a single file
crop_face(input='input/test1.jpg', output='output/htu_test1.jpg', width=250, height=250, margin = 1.8, frame = True, classifier='alt')

# For all files in a directory
crop_face_dir(input_dir='input', output_dir='htu_output', verbose=True, width=200, height=200, margin=1.8, frame=True, classifier='default')
```

# 関数の仕様
英語ですが，[Github Pages](https://yuu-miino-nue.github.io/crop_tools/api/crop_tools.html) にまとめました．ご参照ください．
