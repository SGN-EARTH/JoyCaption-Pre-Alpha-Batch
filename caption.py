import torch
import torch.amp.autocast_mode
import os
import sys
import logging
import warnings
import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from typing import List

# 常量定义
CLIP_PATH = Path("model/siglip-so400m-patch14-384")
VLM_PROMPT = "A descriptive caption for this image:\n"
MODEL_PATH = Path("model/Meta-Llama-3.1-8B-bnb-4bit")
# MODEL_PATH = Path("model/Meta-Llama-3.1-8B")
CHECKPOINT_PATH = Path("model/wpkklhc6")
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# 生成相关参数
MAX_NEW_TOKENS = 1024
DO_SAMPLE = False
TOP_K = 40
TEMPERATURE = 0.25

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
    
    def forward(self, vision_outputs: torch.Tensor):
        return self.linear2(self.activation(self.linear1(vision_outputs)))

def load_models():
    import time  # 导入时间模块
    total_start_time = time.time()  # 记录总开始时间

    # 加载 CLIP 模型
    clip_start_time = time.time()
    print("加载 CLIP")
    clip_processor = AutoProcessor.from_pretrained(str(CLIP_PATH), local_files_only=True)  # 本地加载
    clip_model = AutoModel.from_pretrained(str(CLIP_PATH), local_files_only=True).vision_model.eval().requires_grad_(False).to("cuda")
    clip_end_time = time.time()
    print(f"已加载 CLIP ，耗时: {clip_end_time - clip_start_time:.2f} 秒。\n")

    # 加载 tokenizer
    tokenizer_start_time = time.time()
    print("加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=False, local_files_only=True)  # 本地加载
    assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"tokenizer 类型错误: {type(tokenizer)}"
    tokenizer_end_time = time.time()
    print(f"已加载 tokenizer ，耗时: {tokenizer_end_time - tokenizer_start_time:.2f} 秒。\n")

    # 加载 LLM
    text_model_start_time = time.time()
    print("加载 LLM")
    text_model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True).eval()  # 本地加载
    text_model_end_time = time.time()
    print(f"已加载 LLM ，耗时: {text_model_end_time - text_model_start_time:.2f} 秒。\n")

    # 加载 adapter
    adapter_start_time = time.time()
    print("加载 adapter")
    image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
    image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu", weights_only=True))  # 加载本地权重
    image_adapter.eval().to("cuda")
    adapter_end_time = time.time()
    print(f"已加载 adapter ，耗时: {adapter_end_time - adapter_start_time:.2f} 秒。\n")

    total_end_time = time.time()
    print(f"所有模型和组件加载完成，总耗时: {total_end_time - total_start_time:.2f} 秒。\n")

    return clip_processor, clip_model, tokenizer, text_model, image_adapter

@torch.no_grad()
def stream_chat(input_images: List[Image.Image], batch_size: int, pbar: tqdm, models: tuple) -> List[str]:
    clip_processor, clip_model, tokenizer, text_model, image_adapter = models
    torch.cuda.empty_cache()
    all_captions = []

    for i in range(0, len(input_images), batch_size):
        batch = input_images[i:i+batch_size]
        
        try:
            images = clip_processor(images=batch, return_tensors='pt', padding=True).pixel_values.to('cuda')
        except ValueError as e:
            print(f"处理图片批次时出错: {e}")
            print("跳过此批次并继续处理...")
            continue

        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=images, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = image_adapter(image_features).to(dtype=torch.bfloat16)

        prompt = tokenizer.encode(VLM_PROMPT, return_tensors='pt')
        prompt_embeds = text_model.model.embed_tokens(prompt.to('cuda')).to(dtype=torch.bfloat16)
        embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64)).to(dtype=torch.bfloat16)

        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images,
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1).to(dtype=torch.bfloat16)

        input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).expand(embedded_images.shape[0], -1),
            torch.zeros((embedded_images.shape[0], embedded_images.shape[1]), dtype=torch.long),
            prompt.expand(embedded_images.shape[0], -1),
        ], dim=1).to('cuda')

        attention_mask = torch.ones_like(input_ids)

        generate_ids = text_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            top_k=TOP_K,
            temperature=TEMPERATURE,
        )

        generate_ids = generate_ids[:, input_ids.shape[1]:]

        for ids in generate_ids:
            caption = tokenizer.decode(ids[:-1] if ids[-1] == tokenizer.eos_token_id else ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            caption = caption.replace('<|end_of_text|>', '').replace('<|finetune_right_pad_id|>', '').strip()
            all_captions.append(caption)

        if pbar:
            pbar.update(len(batch))

    return all_captions

def process_directory(input_dir: Path, output_dir: Path, batch_size: int, models: tuple):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    images_to_process = [f for f in image_files if not (output_dir / f"{f.stem}.txt").exists()]

    if not images_to_process:
        print("没有新的图片需要处理。")
        return

    with tqdm(total=len(images_to_process), desc="处理图片", unit="it") as pbar:
        for i in range(0, len(images_to_process), batch_size):
            batch_files = images_to_process[i:i+batch_size]
            batch_images = [Image.open(f).convert('RGB') for f in batch_files]

            captions = stream_chat(batch_images, batch_size, pbar, models)
            
            for file, caption in zip(batch_files, captions):
                with open(output_dir / f"{file.stem}.txt", 'w', encoding='utf-8') as f:
                    f.write(caption)

            for img in batch_images:
                img.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="处理图片并生成描述。")
    parser.add_argument("input", nargs='+', help="输入图片文件或目录（可包含多个目录）")
    parser.add_argument("--output", help="输出目录（可选）")
    parser.add_argument("--bs", type=int, default=4, help="批处理大小（默认：4）")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_paths = [Path(input_path) for input_path in args.input]
    batch_size = args.bs
    models = load_models()

    for input_path in input_paths:
        if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
            output_path = input_path.with_suffix('.txt')
            print(f"处理单张图片: {input_path.name}")
            with tqdm(total=1, desc="处理图片", unit="it") as pbar:
                captions = stream_chat([Image.open(input_path).convert('RGB')], 1, pbar, models)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(captions[0])
            print(f"输出保存至 {output_path}")
        elif input_path.is_dir():
            output_path = Path(args.output) if args.output else input_path
            print(f"处理目录: {input_path}")
            print(f"输出目录: {output_path}")
            print(f"批处理大小: {batch_size}\n")
            process_directory(input_path, output_path, batch_size, models)
        else:
            print(f"无效输入: {input_path}")
            print("跳过...")

    if not input_paths:
        print("用法:")
        print("处理单张图片: python app.py [image_file] [--bs batch_size]")
        print("处理目录（相同输入输出目录）: python app.py [directory] [--bs batch_size]")
        print("处理目录（不同输入输出目录）: python app.py [directory] --output [output_directory] [--bs batch_size]")
        print("处理多个目录: python app.py [directory1] [directory2] ... [--output output_directory] [--bs batch_size]")
        sys.exit(1)

if __name__ == "__main__":
    main()
