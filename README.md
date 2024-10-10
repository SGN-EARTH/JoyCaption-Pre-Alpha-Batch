# JoyCaption-Pre-Alpha-Batch

完全离线且本地的批量打标工具。原始项目：[Wi-zz/joy-caption-pre-alpha](https://huggingface.co/Wi-zz/joy-caption-pre-alpha) 。

所有代码均是把原始项目代码丢给 ChatGPT 并疯狂扯皮而来。~~有问题可以问 AI ，AI 不能解答的我也无法解答，OK？~~

一开始是折腾各种自动下载模型项目，即使模型和各类资源都已下载完成，运行时还是会请求云端。

。。。把 app.py 丢给 ChatGPT 后居然就行了，目前运行没遇到啥问题。~~面向 ChatGPT 编程~~

使用 venv 虚拟环境或 conda 的可以自行折腾。

## 目录结构

```
├─input
├─model
│  ├─Meta-Llama-3.1-8B
│  │      config.json
│  │      generation_config.json
│  │      model-00001-of-00004.safetensors
│  │      model-00002-of-00004.safetensors
│  │      model-00003-of-00004.safetensors
│  │      model-00004-of-00004.safetensors
│  │      model.safetensors.index.json
│  │      special_tokens_map.json
│  │      tokenizer.json
│  │      tokenizer_config.json
│  │      
│  ├─Meta-Llama-3.1-8B-bnb-4bit
│  │      config.json
│  │      generation_config.json
│  │      model.safetensors
│  │      special_tokens_map.json
│  │      tokenizer.json
│  │      tokenizer_config.json
│  │      
│  ├─siglip-so400m-patch14-384
│  │      config.json
│  │      model.safetensors
│  │      preprocessor_config.json
│  │      README.md
│  │      special_tokens_map.json
│  │      spiece.model
│  │      tokenizer_config.json
│  │      
│  └─wpkklhc6
│          config.yaml
│          image_adapter.pt
│          
└─python
│  ├─ ...
│  caption.py
│  requirements.txt
│  run.bat
│  run2.bat
```

## 获取模型

创建或打开 model 文件夹，使用 git clone 获取模型。

- wpkklhc6：

    ```
    # 82.0 MB
    
    git clone https://huggingface.co/Wi-zz/joy-caption-pre-alpha
    或者
    git clone https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha
    ```

    获取完成后进入 joy-caption-pre-alpha 文件夹把 wpkklhc6 目录移动到上一级目录。

- siglip-so400m-patch14-384

    ```
    # 3.27 GB
    git clone https://huggingface.co/google/siglip-so400m-patch14-384
    ```

- Meta-Llama-3.1-8B 或 Meta-Llama-3.1-8B-bnb-4bit

    ```
    # 5.31 GB
    git clone https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit
    
    # 14.9 GB
    git clone https://huggingface.co/unsloth/Meta-Llama-3.1-8B
    ```

## 安装

`requirements.txt` 使用 `pip freeze >> requirements.txt` 生成，所有包依赖指定了版本。

一、拉取本库

```
git clone https://github.com/SGN-EARTH/JoyCaption-Pre-Alpha-Batch.git
```

二、下载文件

```
https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
https://bootstrap.pypa.io/get-pip.py
```

三、开始安装

```
解压 python-3.11.9-embed-amd64.zip 到 python 目录，把 get-pip.py 丢 python 目录中，然后把 python 目录移动到 JoyCaption-Pre-Alpha-Batch 目录内；

运行 run.bat ，执行 python python\get-pip.py ；

编辑 python\python311._pth 把 import site 所在行开头的 # 符号去除，保存文件。
```

四、安装依赖

```
# torch
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# 其他依赖
python -m pip install -r requirements.txt
```

这个方式适合把很多 python 项目打包成所谓的一键包，~~拒绝引流，拒绝会员，拒绝认爹，直接白嫖~~，带着包就能跑路。

## 使用

把图片丢 input 目录中，运行 run2.bat 。

或者运行 run.bat 后执行 python caption.py -h 查看帮助。

根据自身硬件环境选择未量化或量化模型。

> 以下仅供参考。不同的图片处理时间差异很大。
>
> 
>
> 人物图，RTX2080Ti_22G：
>
> bs=4：Meta-Llama-3.1-8B 显存占用大概 19.1 GB 。Meta-Llama-3.1-8B-bnb-4bit 显存占用大概 10.9 GB 。
>
> bs=4：每张图 Meta-Llama-3.1-8B 一般在 35 秒左右，Meta-Llama-3.1-8B-bnb-4bit 一般在 25 秒。有大显存还是用未量化的吧。
>
> bs=8：使用 Meta-Llama-3.1-8B 每张图在 17 秒上下，显存占用最高 20.3 GB 。
>
> 
>
> 系统自带壁纸，RTX3060_12G：
>
> bs=8：使用 Meta-Llama-3.1-8B-bnb-4bit 每张图在 8 秒左右，显存占满。

## 自定义

如果需要修改使用量化或非量化模型，编辑 caption.py 把模型路径改成想要使用的模型所在的位置。

```
MODEL_PATH = Path("model/Meta-Llama-3.1-8B")
或者
MODEL_PATH = Path("model/Meta-Llama-3.1-8B-bnb-4bit")
```

图片描述相关参数

```
MAX_NEW_TOKENS = 1024
DO_SAMPLE = False
TOP_K = 40
TEMPERATURE = 0.25
```

run.bat 内容：

```
@echo off

cd /d %cd%
:: cd /d %~dp0

set DIR=%cd%

:: https://www.python.org/ftp/python/
:: https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
:: https://bootstrap.pypa.io/get-pip.py

set PATH=%DIR%\python;%DIR%\python\Scripts;%PATH%;
:: set PATH=%DIR%\git\bin;%DIR%\python;%DIR%\python\Scripts;%PATH%;
set PY_LIBS=%DIR%\python\Scripts\Lib;%DIR%\python\Scripts\Lib\site-packages
set PY_PIP=%DIR%\python\Scripts
set PIP_INSTALLER_LOCATION=%DIR%\python\get-pip.py

set HF_HOME=%DIR%\hf
:: set HF_ENDPOINT=https://hf-mirror.com
:: set HUGGINGFACE_HUB_DISABLE_CACHE=1

:: 安装 pip 后不可使用时，可尝试编辑 %DIR%\python\pythonXXX._pth 去掉 import site 的注释

:: python 脚本将当前目录添加到 sys.path
::      import os
::      import sys
::      sys.path.append(os.path.dirname(os.path.abspath(__file__)))

:: 包临时缓存路径
set PIP_CACHE_DIR=..\cache

:: 缓存。off 禁用，on 启用
:: set PIP_NO_CACHE_DIR=off

:: 包索引 URL
set PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
:: https://pypi.org/simple
:: https://mirrors.163.com/pypi/simple/
:: https://mirrors.cloud.tencent.com/pypi/simple
:: https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

:: python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
:: python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
:: python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: python -m pip install xformers===0.0.28.post1 --extra-index-url https://download.pytorch.org/whl/cu124

:: 额外包索引 URL
:: set PIP_EXTRA_INDEX_URL=https://pypi.org/simple

:: 请求包索引超时时间。单位：秒。
set PIP_TIMEOUT=10

:: 更详细的调试信息
:: set PIP_VERBOSE=1

cmd /k
```

