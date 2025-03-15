# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if self.image_key in row_dict:
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            print("here: ", row_dict.pop(self.image_key), self.image_key)
            row_dict["multi_modal_data"] = {
                "image": [
                    process_image(image, self.max_pixels, self.min_pixels) for image in row_dict.pop(self.image_key)
                ]
            }
            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
        elif self.video_key in row_dict:
            prompt = prompt.replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
            video_inputs = []
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)

            model_inputs = self.processor(video_inputs, prompt, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                video_grid_thw=model_inputs["video_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)

        else:
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        return row_dict

    def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR,
                    return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
        if isinstance(ele["video"], str):
            video_reader_backend = get_video_reader_backend()
            try:
                video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
            except Exception as e:
                logger.warning(
                    f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
                video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

            nframes, _, height, width = video.shape
            min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
            total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
            max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels_supposed = ele.get("max_pixels", max_pixels)
            if max_pixels_supposed > max_pixels:
                logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
            max_pixels = min(max_pixels_supposed, max_pixels)
            if "resized_height" in ele and "resized_width" in ele:
                resized_height, resized_width = smart_resize(
                    ele["resized_height"],
                    ele["resized_width"],
                    factor=image_factor,
                )
            else:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=image_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
            video = transforms.functional.resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            if return_video_sample_fps:
                return video, sample_fps
            return video
        else:
            assert isinstance(ele["video"], (list, tuple))
            process_info = ele.copy()
            process_info.pop("type", None)
            process_info.pop("video", None)
            images = [
                fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
                for video_element in ele["video"]
            ]
            nframes = ceil_by_factor(len(images), FRAME_FACTOR)
            if len(images) < nframes:
                images.extend([images[-1]] * (nframes - len(images)))
            if return_video_sample_fps:
                return images, process_info.pop("fps", 2.0)
            return images
