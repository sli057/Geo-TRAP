## Installation

Geo-TRAP is implemented based on [MMAction2](https://github.com/open-mmlab/mmaction2.git). 
Therefore, we need to install the requirements of MMAction2.

We provide some tips related to MMAction2 installation in this file. For the original instruction, please see [here](https://github.com/open-mmlab/mmaction2/blob/master/docs/install.md).

<!-- TOC -->

- [Requirements](#requirements)
- [Install MMAction2 within Geo-TRAP](#install-mmaction2-within-geo-trap)
- [A from-scratch setup script](#a-from-scratch-setup-script)
- [Verification](#verification)

<!-- TOC -->

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) 1.1.1+
- Numpy
- ffmpeg (4.2 is preferred)
- [decord](https://github.com/dmlc/decord) (optional, 0.4.1+): Install CPU version by `pip install decord==0.4.1` and install GPU version from source
- [PyAV](https://github.com/mikeboers/PyAV) (optional): `conda install av -c conda-forge -y`
- [PyTurboJPEG](https://github.com/lilohuang/PyTurboJPEG) (optional): `pip install PyTurboJPEG`
- [denseflow](https://github.com/open-mmlab/denseflow) (optional): See [here](https://github.com/innerlee/setup) for simple install scripts.
- [moviepy](https://zulko.github.io/moviepy/) (optional): `pip install moviepy`. See [here](https://zulko.github.io/moviepy/install.html) for official installation. **Note**(according to [this issue](https://github.com/Zulko/moviepy/issues/693)) that:
    1. For Windows users, [ImageMagick](https://www.imagemagick.org/script/index.php) will not be automatically detected by MoviePy,
    there is a need to modify `moviepy/config_defaults.py` file by providing the path to the ImageMagick binary called `magick`, like `IMAGEMAGICK_BINARY = "C:\\Program Files\\ImageMagick_VERSION\\magick.exe"`
    2. For Linux users, there is a need to modify the `/etc/ImageMagick-6/policy.xml` file by commenting out
    `<policy domain="path" rights="none" pattern="@*" />` to `<!-- <policy domain="path" rights="none" pattern="@*" /> -->`, if [ImageMagick](https://www.imagemagick.org/script/index.php) is not detected by `moviepy`.
- [Pillow-SIMD](https://docs.fast.ai/performance.html#pillow-simd) (optional): Install it by the following scripts.
```shell
conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
conda install -y jpeg libtiff
```

**Note**:  You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

### Install MMAction2 within Geo-TRAP

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install PyTorch 1.5,
you need to install the prebuilt PyTorch with CUDA 10.1.

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g.2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.3.1.,
you need to install the prebuilt PyTorch with CUDA 9.2.

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.

c. Install mmcv, we recommend you to install the pre-build mmcv as below.

```shell
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
```

See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
Optionally you can choose to compile mmcv from source by the following command

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```

Or directly run

```shell
pip install mmcv-full
# alternative: pip install mmcv
```

**Important:** You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

d. Clone the GeoTRAP repository

```shell
https://github.com/sli057/Geo-TRAP.git
cd Geo-TRAP
```

d. Install build requirements and then install MMAction2.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

If you build MMAction2 on macOS, replace the last command with

```
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```


Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, MMAction2 is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. If you would like to use `PyAV`, you can install it with `conda install av -c conda-forge -y`.

5. Some dependencies are optional. Running `python setup.py develop` will only install the minimum runtime requirements.
To use optional dependencies like `decord`, either install them with `pip install -r requirements/optional.txt`
or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`,
valid keys for the `[optional]` field are `all`, `tests`, `build`, and `optional`) like `pip install -v -e .[tests,build]`.


### A from-scratch setup script

Here is a full script for setting up MMAction2 with conda and link the dataset path 
(supposing that your Jester dataset path is $JESTER_ROOT and your UCF-101 dataset path is $UCF101_ROOT).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install the latest mmcv or mmcv-full, here we take mmcv as example
pip install mmcv

# install mmaction2
git clone https://github.com/sli057/Geo-TRAP.git
cd Geo-TRAP
pip install -r requirements/build.txt
python setup.py develop

mkdir data
cd data
ln -s $UCF101_ROOT ucf101
ln -s $JESTER_ROOT jester
```

For more details on data preparation, you can refer to

* [preparing_ucf101](/tools/data/ucf101/README.md)
* [preparing_jester](/tools/data/jester/README.md)


### Verification

To verify whether MMAction2 and the required environment are installed correctly,
we can run sample python codes to initialize a recognizer and inference a demo video:

```python
import torch
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, device=device)
# inference the demo video
inference_recognizer(model, 'demo/demo.mp4', 'demo/label_map.txt')
```
