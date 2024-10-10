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
