# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from .model.conversation import SeparatorStyle, conv_templates
from .model.mm_utils import KeywordsStoppingCriteria, process_image, tokenizer_image_token
from .model import get_model_name_from_path, load_pretrained_model
from transformers import TextIteratorStreamer
from threading import Thread

class DescribeAnythingModel(nn.Module):
    def __init__(self, model_path, conv_mode, prompt_mode, **kwargs):
        super().__init__()
        
        self.model_path = model_path
        self.conv_mode = conv_mode
        self.prompt_mode = prompt_mode

        if isinstance(model_path, str):
            self.tokenizer, self.model, _, _ = load_pretrained_model(model_path, None, None, **kwargs)
            self.model_name = get_model_name_from_path(model_path)
        else:
            # model_path is actually a dict with model, tokenizer, and (optionally) model_name
            self.model = model_path["model"]
            self.tokenizer = model_path["tokenizer"]
            self.model_name = model_path.get("model_name", None)
        
        image_processor = self.model.vision_tower.image_processor
        self.model.config.image_processor = image_processor

    def get_prompt(self, qs):
        if DEFAULT_IMAGE_TOKEN not in qs:
            raise ValueError("no <image> tag found in input.")

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        return prompt, conv

    @staticmethod
    def mask_to_box(mask_np):
        mask_coords = np.argwhere(mask_np)
        y0, x0 = mask_coords.min(axis=0)
        y1, x1 = mask_coords.max(axis=0) + 1
        
        h = y1 - y0
        w = x1 - x0

        return x0, y0, w, h

    @classmethod
    def crop_image(cls, pil_img, mask_np, crop_mode, min_box_w=48, min_box_h=48):
        if crop_mode == "full":
            # no crop
            info = dict(mask_np=mask_np)
            return pil_img, info

        if crop_mode == "crop":
            # crop image and mask
            x0, y0, w, h = cls.mask_to_box(mask_np)
            img_np = np.asarray(pil_img)
            assert img_np.shape[:2] == mask_np.shape, f"image shape mismatches with mask shape: {img_np.shape}, {mask_np.shape}"
            cropped_mask_np = mask_np[y0:y0+h, x0:x0+w]
            cropped_img_np = img_np[y0:y0+h, x0:x0+w]
            cropped_pil_img = Image.fromarray(cropped_img_np)
        elif crop_mode == "context_crop":
            # crop image and mask
            x0, y0, w, h = cls.mask_to_box(mask_np)
            img_np = np.asarray(pil_img)
            assert img_np.shape[:2] == mask_np.shape, f"image shape mismatches with mask shape: {img_np.shape}, {mask_np.shape}"
            img_h, img_w = img_np.shape[:2]
            cropped_mask_np = mask_np[max(y0-h, 0):min(y0+2*h, img_h), max(x0-w, 0):min(x0+2*w, img_w)]
            cropped_img_np = img_np[max(y0-h, 0):min(y0+2*h, img_h), max(x0-w, 0):min(x0+2*w, img_w)]
            cropped_pil_img = Image.fromarray(cropped_img_np)
        elif crop_mode == "focal_crop":
            # crop image and mask
            x0, y0, w, h = cls.mask_to_box(mask_np)
            img_np = np.asarray(pil_img)
            assert img_np.shape[:2] == mask_np.shape, f"image shape mismatches with mask shape: {img_np.shape}, {mask_np.shape}"
            img_h, img_w = img_np.shape[:2]

            xc, yc = x0 + w/2, y0 + h/2
            # focal_crop: need to have at least min_box_w and min_box_h pixels, otherwise resizing to (384, 384) leads to artifacts that may be OOD
            w, h = max(w, min_box_w), max(h, min_box_h)
            x0, y0 = int(xc - w / 2), int(yc - h / 2)
            
            cropped_mask_np = mask_np[max(y0-h, 0):min(y0+2*h, img_h), max(x0-w, 0):min(x0+2*w, img_w)]
            cropped_img_np = img_np[max(y0-h, 0):min(y0+2*h, img_h), max(x0-w, 0):min(x0+2*w, img_w)]
            cropped_pil_img = Image.fromarray(cropped_img_np)
        elif crop_mode == "crop_mask":
            # crop image and mask
            x0, y0, w, h = cls.mask_to_box(mask_np)
            img_np = np.asarray(pil_img)
            assert img_np.shape[:2] == mask_np.shape, f"image shape mismatches with mask shape: {img_np.shape}, {mask_np.shape}"
            cropped_mask_np = mask_np[y0:y0+h, x0:x0+w]
            cropped_img_np = img_np[y0:y0+h, x0:x0+w]
            # Mask the image
            cropped_img_np = cropped_img_np * cropped_mask_np[..., None]
            cropped_pil_img = Image.fromarray(cropped_img_np)
        else:
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")

        info = dict(mask_np=cropped_mask_np)
        return cropped_pil_img, info

    def get_description(self, image_pil, mask_pil, query, streaming=False, temperature=0.2, top_p=0.5, num_beams=1, max_new_tokens=512, **kwargs):
        # kwargs is passed to generation_kwargs: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
        
        prompt, conv = self.get_prompt(query)
        if not isinstance(image_pil, (list, tuple)):
            assert not isinstance(mask_pil, (list, tuple)), "image_pil and mask_pil must be both list or tuple or not list or tuple."
            image_pils = [image_pil]
            mask_pils = [mask_pil]
        else:
            image_pils = image_pil
            mask_pils = mask_pil
        description = self.get_description_from_prompt(image_pils, mask_pils, prompt, conv, streaming=streaming, temperature=temperature, top_p=top_p, num_beams=num_beams, max_new_tokens=max_new_tokens, **kwargs)
        
        return description

    def get_image_tensor(self, image_pil, mask_pil, crop_mode, crop_mode2):
        # the pil has True/False (if the value is non-zero, then we treat it as True)
        mask_np = (np.asarray(mask_pil) > 0).astype(np.uint8)
        images_tensor, image_info = process_image(image_pil, self.model.config, None, pil_preprocess_fn=lambda pil_img: self.crop_image(image_pil, mask_np=mask_np, crop_mode=crop_mode))
        images_tensor = images_tensor[None].to(self.model.device, dtype=torch.float16)

        mask_np = image_info["mask_np"]
        mask_pil = Image.fromarray(mask_np * 255)
        
        masks_tensor = process_image(mask_pil, self.model.config, None)
        masks_tensor = masks_tensor[None].to(self.model.device, dtype=torch.float16)
        
        images_tensor = torch.cat((images_tensor, masks_tensor[:, :1, ...]), dim=1)

        if crop_mode2 is not None:
            images_tensor2, image_info2 = process_image(image_pil, self.model.config, None, pil_preprocess_fn=lambda pil_img: self.crop_image(pil_img, mask_np=mask_np, crop_mode=crop_mode2))
            images_tensor2 = images_tensor2[None].to(self.model.device, dtype=torch.float16)

            mask_np2 = image_info2["mask_np"]
            mask_pil2 = Image.fromarray(mask_np2 * 255)
            
            masks_tensor2 = process_image(mask_pil2, self.model.config, None)
            masks_tensor2 = masks_tensor2[None].to(self.model.device, dtype=torch.float16)

            images_tensor2 = torch.cat((images_tensor2, masks_tensor2[:, :1, ...]), dim=1)
        else:
            images_tensor2 = None
            
        return torch.cat((images_tensor, images_tensor2), dim=1) if images_tensor2 is not None else images_tensor
    
    def get_description_from_prompt(self, image_pils, mask_pils, prompt, conv, streaming=False, temperature=0.2, top_p=0.5, num_beams=1, max_new_tokens=512, **kwargs):
        if streaming:
            return self.get_description_from_prompt_iterator(image_pils, mask_pils, prompt, conv, streaming=True, temperature=temperature, top_p=top_p, num_beams=num_beams, max_new_tokens=max_new_tokens, **kwargs)
        else:
            # If streaming is False, there will be only one output
            output = self.get_description_from_prompt_iterator(image_pils, mask_pils, prompt, conv, streaming=False, temperature=temperature, top_p=top_p, num_beams=num_beams, max_new_tokens=max_new_tokens, **kwargs)
            return next(output)

    def get_description_from_prompt_iterator(self, image_pils, mask_pils, prompt, conv, streaming=False, temperature=0.2, top_p=0.5, num_beams=1, max_new_tokens=512, **kwargs):
        crop_mode, crop_mode2 = self.prompt_mode.split("+")
        assert crop_mode == "full", "Current prompt only supports first crop as full (non-cropped). If you need other specifications, please update the prompt."
        
        assert len(image_pils) == len(mask_pils), f"image_pils and mask_pils must have the same length. Got {len(image_pils)} and {len(mask_pils)}."
        image_tensors = [self.get_image_tensor(image_pil, mask_pil, crop_mode=crop_mode, crop_mode2=crop_mode2) for image_pil, mask_pil in zip(image_pils, mask_pils)]
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if streaming else None
        generation_kwargs = dict(
            input_ids=input_ids,
            images=image_tensors,
            do_sample=True if temperature > 0 else False,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            streamer=streamer,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        if streaming:
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                if stop_str in generated_text:
                    generated_text = generated_text[:generated_text.find(stop_str)]
                    break
                yield new_text
            
            thread.join()
        else:
            with torch.inference_mode():
                output_ids = self.model.generate(**generation_kwargs)

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            yield outputs
