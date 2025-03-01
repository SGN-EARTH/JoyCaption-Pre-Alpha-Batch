import torch

total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
print(f'\033[32mCUDA版本：{torch.version.cuda}\033[0m')
print(f'\033[32mPytorch版本：{torch.__version__}\033[0m')
print(f'\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m')
print(f'\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m')
if torch.cuda.get_device_capability()[0] >= 8:
    print(f'\033[32mBF16支持：是\033[0m')
    dtype = torch.bfloat16
else:
    print(f'\033[32mBF16支持：否\033[0m')
    dtype = torch.float16
