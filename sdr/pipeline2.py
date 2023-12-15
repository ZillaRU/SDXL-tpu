import cv2
import inspect
import numpy as np
from transformers import CLIPTokenizer
from tqdm import tqdm

from .prompt_parser import parse_prompt_attention
from .scheduler import create_random_tensors, sample
from PIL import Image, ImageFilter, ImageOps
import PIL
from .npuengine import EngineOV
import time
from . import masking
from .utils import PromptChunk, resize_image, flatten, apply_overlay, WrapOutput
import torch
from .preprocess import HEDdetector
import math
from . import ultimate
import random
import os


opt_C = 4
opt_f = 8


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)


class StableDiffusionPipeline:
    def __init__(
            self,
            scheduler=None,
            tokenizer="openai/clip-vit-large-patch14",
            width=512,
            height=512,
            basic_model="abyssorangemix2NSFW-unet-2",
            controlnet_name=None,
            device_id=0,
            extra="",
    ):
        tokenizer = "./tokenizer"
        extra = ""
        self.device_id = device_id
        self.latent_shape = (4, height//8, width//8)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        vocab = self.tokenizer.get_vocab()
        self.tokenizer.comma_token = vocab.get(',</w>', None)

        self.scheduler = scheduler
        self.basemodel_name = basic_model
        st_time = time.time()
        self.text_encoder = EngineOV("./models/basic/{}/text_encoder_1684x_f32.bmodel".format(
            basic_model), device_id=self.device_id)
        print("====================== Load TE in ", time.time()-st_time)
        
        st_time = time.time()
        self.unet_pure = EngineOV("./models/basic/{}/unet_multize.bmodel".format(
            basic_model, extra), device_id=self.device_id)
        print("====================== Load UNET in ", time.time()-st_time)
        
        self.unet_lora = None
        if os.path.exists("./models/basic/{}/unet_multize_lora.bmodel"):
            st_time = time.time()
            self.unet_lora = EngineOV("./models/basic/{}/unet_multize_lora.bmodel".format(
                basic_model, extra), device_id=self.device_id)
            print("====================== Load UNET-lora in ", time.time()-st_time)
        
        st_time = time.time()
        self.vae_decoder = EngineOV("./models/basic/{}/{}vae_decoder_multize.bmodel".format(
            basic_model, extra), device_id=self.device_id)
        print("====================== Load VAE DE in ", time.time()-st_time)
        
        st_time = time.time()
        self.vae_encoder = EngineOV("./models/basic/{}/{}vae_encoder_multize.bmodel".format(
            basic_model, extra), device_id=self.device_id)
        print("====================== Load VAE EN in ", time.time()-st_time)
        
        self.controlnet = None
        # if controlnet_name:
        #     st_time = time.time()
        #     self.controlnet = EngineOV("./models/controlnet/{}.bmodel".format(
        #         controlnet_name), device_id=self.device_id)
        #     print("====================== Load CN in ", time.time()-st_time)
        # else:
        #     self.controlnet = None
        
        # st_time = time.time()
        # self.tile_contorlnet = EngineOV("./models/controlnet/tile_multize.bmodel", device_id=self.device_id)
        # print("====================== Load TILE in ", time.time()-st_time)

        self.unet = self.unet_pure
        self.tile_controlnet_name = "tile_multisize"
        self.controlnet_name = controlnet_name
        self.init_image_shape = (width, height)
        self._width = width
        self._height = height
        self.hed_model = None
        self.mlsd_model = None
        self.default_args()
        print(self.text_encoder, self.unet, self.vae_decoder,
              self.vae_encoder, self.controlnet)
    
    def set_lora(self, lora_state):
        if lora_state: # set to unet_lora
            if self.unet == self.unet_lora:
                return False
            else:
                self.unet = self.unet_lora
                return True
        else: # set to unet_pure
            if self.unet == self.unet_pure:
                return False
            else:
                self.unet = self.unet_pure
                return True

    def set_height_width(self, height, width):
        self._height = height
        self._width = width
        self.init_image_shape = (width, height)
        self.latent_shape = (4, height//8, width//8)

    def default_args(self):
        self.batch_size = 1
        self.handle_masked = False

    def _preprocess_mask(self, mask):
        if self.handle_masked:
            return mask
        h, w = mask.shape
        if h != self.init_image_shape[0] and w != self.init_image_shape[1]:
            mask = cv2.resize(
                mask,
                (self.init_image_shape[1], self.init_image_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        mask = cv2.resize(
            mask,
            (self.init_image_shape[1] // 8, self.init_image_shape[0] // 8),
            interpolation=cv2.INTER_NEAREST
        )
        mask = mask.astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)
        mask = 1 - mask
        return mask

    def _preprocess_image(self, image:Image):
        if image.mode != "RGB":
            image = image.convert("RGB") # RGBA or other -> RGB
        image = np.array(image)
        h, w = image.shape[:-1]
        if h != self.init_image_shape[1] or w != self.init_image_shape[0]:
            image = cv2.resize(
                image,
                (self.init_image_shape[0], self.init_image_shape[1]),
                interpolation=cv2.INTER_LANCZOS4
            )
        # normalize
        image = image.astype(np.float32) / 255.0
        image = 2.0 * image - 1.0
        # to batch
        image = image[None].transpose(0, 3, 1, 2)
        return image

    def _encode_image(self, init_image):
        moments = self.vae_encoder({
            "input.1": self._preprocess_image(init_image)
        })[0]
        mean, logvar = np.split(moments, 2, axis=1)
        std = np.exp(logvar * 0.5)
        latent = (mean + std * np.random.randn(*mean.shape)) * 0.18215
        return latent

    def _prepare_image(self, image, controlnet_args={}):
        print("do not use controlnet_args")
        width, height = self.init_image_shape
        if isinstance(image, Image.Image):
            image = image
        else:
            image = Image.fromarray(image)
        image = image.resize((width, height), PIL.Image.LANCZOS)  # RGB
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = image[None, :]
        return np.concatenate((image, image), axis=0)

    def _prepare_canny_image(self, image, controlnet_args={}):
        image = np.array(image)
        low_threshold = controlnet_args.get("low_threshold", 100)
        high_threshold = controlnet_args.get("high_threshold", 200)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    def _prepare_hed_image(self, image, controlnet_args={}):
        print("in hed preprocess, we do not use controlnet_args")
        image = np.array(image)
        if self.hed_model is None:
            self.hed_model = EngineOV(
                "./models/other/hed_fp16_dynamic.bmodel", device_id=self.device_id)
        hed = HEDdetector(self.hed_model)
        img = hed(image)
        image = img[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    def _before_upscale(self):
        self.controlnet, self.tile_contorlnet = self.tile_contorlnet, self.controlnet
        self.controlnet_name, self.tile_controlnet_name = self.tile_controlnet_name, self.controlnet_name

    def _after_upscale(self):
        self._before_upscale()

    def generate_zero_controlnet_data(self):
        res = []
        res.append(np.zeros((2, 320, self._height//8,
                   self._width//8)).astype(np.float32))
        res.append(np.zeros((2, 320, self._height//8,
                   self._width//8)).astype(np.float32))
        res.append(np.zeros((2, 320, self._height//8,
                   self._width//8)).astype(np.float32))
        res.append(np.zeros((2, 320, self._height//16,
                   self._width//16)).astype(np.float32))
        res.append(np.zeros((2, 640, self._height//16,
                   self._width//16)).astype(np.float32))
        res.append(np.zeros((2, 640, self._height//16,
                   self._width//16)).astype(np.float32))
        res.append(np.zeros((2, 640, self._height//32,
                   self._width//32)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//32,
                   self._width//32)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//32,
                   self._width//32)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//64,
                   self._width//64)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//64,
                   self._width//64)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//64,
                   self._width//64)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//64,
                   self._width//64)).astype(np.float32))
        return res

    def run_unet(self, latent, t, text_embedding, controlnet_img, controlnet_weight=1.0):
        if controlnet_img is not None:            
            controlnet_res = self.controlnet({"latent": latent.astype(np.float32),  # #### conditioning_scale=controlnet_conditioning_scale,
                                              "prompt_embeds": text_embedding,
                                              "image": controlnet_img,
                                              "t": t})
            if controlnet_weight != 1:
                for i in range(len(controlnet_res)):
                    controlnet_res[i] = controlnet_res[i] * controlnet_weight
        else:
            controlnet_res = self.generate_zero_controlnet_data()
        down_block_additional_residuals = controlnet_res[:-1]
        mid_block_additional_residual = controlnet_res[-1]
        res = self.unet([latent.astype(np.float32), t, text_embedding,
                        mid_block_additional_residual, *down_block_additional_residuals])
        return res

    def call_back_method(self):
        def callback(latent, t, text_embedding, cond_img=None, controlnet_weight=1.0):
            return self.run_unet(latent, t, text_embedding, controlnet_img=cond_img, controlnet_weight=controlnet_weight)
        return callback

    def encoder_with_resize(self, image, upscale=False):
        """
        Resizes the input image if it is not the ideal size and encodes it using the VAE encoder.

        Parameters:
            image (ndarray): The input image to be encoded.
            upscale (bool): If True, perform upscale resize on the image. Default is False.

        Returns:
            ndarray: The encoded latent representation of the input image.
        """
        self.target_upscale_size = image.shape[2:]
        if image.shape[2] != self.init_image_shape[0] or image.shape[3] != self.init_image_shape[1]:
            self.upscale_resize = upscale
            self.upscale_resize_h = image.shape[2]
            self.upscale_resize_w = image.shape[3]
            image = cv2.resize(image[0].transpose(
                1, 2, 0), (self.init_image_shape[0], self.init_image_shape[1]), interpolation=cv2.INTER_LANCZOS4)
            image = image.transpose(2, 0, 1)
            image = image[None, :]
        moments = self.vae_encoder({
            "input.1": image
        })[0]
        mean, logvar = np.split(moments, 2, axis=1)
        std = np.exp(logvar * 0.5)
        latent = (mean + std * np.random.randn(*mean.shape)) * 0.18215
        return latent

    def preprocess_controlnet_image(self, image:Image, controlnet_args={}) -> Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if "invert_image" in controlnet_args:
            if controlnet_args["invert_image"]:
                image = ImageOps.invert(image)
        if "rgbbgr_mode" in controlnet_args:
            if controlnet_args["rgbbgr_mode"]:
                image = image.convert("RGB")
        return image

    def handle_inpaint_image(self, all_seeds=[4187081955], all_subseeds=[4216381720]):
        crop_region = None
        image_mask = self.image_mask
        self.latent_mask = None
        if image_mask is not None:
            image_mask = image_mask.convert('L')
            if self.mask_blur > 0:  # mask + GaussianBlur
                image_mask = image_mask.filter(
                    ImageFilter.GaussianBlur(self.mask_blur))
            if self.inpaint_full_res:
                self.mask_for_overlay = image_mask
                mask = image_mask.convert('L')
                crop_region = masking.get_crop_region(
                    np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(
                    crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region
                mask = mask.crop(crop_region)
                image_mask = resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else:
                image_mask = resize_image(2, image_mask, self.width, self.height)
                np_mask = np.array(image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask
        imgs = []
        for img in self.init_images:

            image = flatten(img, "#ffffff")

            if crop_region is None:
                image = resize_image(2, image, self.width, self.height)

            if image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert(
                    "RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            # crop_region is not None if we are doing inpaint full res
            if crop_region is not None:
                image = image.crop(crop_region)
                image = resize_image(2, image, self.width, self.height)
            
            self.init_image = image
            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(
                imgs[0], axis=0).repeat(self.batch_size, axis=0)

        image = batch_images
        image = 2. * image - 1.0

        # 如果image不在合理的范围之内，我们需要做resize并对上下游都需要做resize
        self.init_latent = self.encoder_with_resize(image, upscale=True)  # img encoder的内容

        if image_mask is not None:  # 还是处理init_latent内容
            init_mask = latent_mask
            if self.upscale_resize:
                init_mask = init_mask.resize(
                    (self.init_image_shape[1], self.init_image_shape[0]), Image.LANCZOS)
            latmask = init_mask.convert('RGB').resize(
                (self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(
                np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))
            self.mask = 1.0 - latmask
            self.nmask = latmask
            self.handle_masked = True
        
        pil_res = self(prompt=self.prompt,
                   negative_prompt=self.negative_prompt,
                   init_image=self.init_image,
                   mask=self.mask,
                   strength=self.strength,
                   controlnet_img=self.controlnet_img,
                   num_inference_steps=self.num_inference_steps,
                   guidance_scale=self.guidance_scale,
                   seeds=self.seeds,
                   subseeds=self.subseeds,
                   using_paint=True,
                   subseed_strength=self.subseed_strength,
                   seed_resize_from_h=self.seed_resize_from_h,
                   seed_resize_from_w=self.seed_resize_from_w,
                   controlnet_args=self.controlnet_args,
                   controlnet_weight=self.controlnet_weight,
                   init_latents=self.init_latent)
        if self.upscale_resize:
            res = cv2.resize(np.array(pil_res), (self.upscale_resize_w, self.upscale_resize_h),
                             interpolation=cv2.INTER_LANCZOS4)

            pil_res = Image.fromarray(res)
        image = apply_overlay(pil_res, self.paste_to, 0, self.overlay_images)
        images = WrapOutput([image])
        return images

    def wrap_upscale(self, prompt,
                     negative_prompt=None,
                     init_image=None,
                     mask=None,
                     strength=0.5,
                     controlnet_img=None, # RGB
                     num_inference_steps=32,
                     guidance_scale=7.5,
                     seeds=[10],
                     subseeds=None,
                     subseed_strength=0.0,
                     seed_resize_from_h=0,
                     seed_resize_from_w=0,
                     controlnet_args={},
                     upscale_factor=2,
                     target_width=1024,
                     target_height=1024,
                     upscale_type="LINEAR",
                     tile_width=512,
                     tile_height=512,
                     mask_blur=8,
                     padding=32,
                     upscaler=None,
                     controlnet_weight=1.0,
                     seams_fix={},
                     seams_fix_enable=False):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.mask = mask
        self.controlnet_img = controlnet_img
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w
        self.controlnet_args = controlnet_args
        self.controlnet_weight = controlnet_weight
        self.upscale_factor = upscale_factor
        self.target_width = target_width
        self.target_height = target_height
        self.upscale_type = upscale_type
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.padding = padding
        self.upscaler = upscaler
        self.seams_fix = seams_fix
        self.seams_fix_enable = seams_fix_enable
        print("close to upscale_img")
        self._before_upscale()
        if upscale_factor <= 0:
            upscale_factor = None
        res = self.upscale_img(init_image, upscale_factor=upscale_factor, target_width=1024, target_height=1024, upscale_type=upscale_type,
                               tile_height=tile_height,
                               tile_width=tile_width,
                               mask_blur=mask_blur,
                               upscaler=upscaler,
                               padding=padding,
                               seams_fix={},
                               seams_fix_enable=False)
        self._after_upscale()
        # warp 接受所有参数 并调用 handle_inpaint_image 实现即可
        # args: prompt, negative_prompt, init_image, mask, strength, controlnet_img, num_inference_steps, guidance_scale, seeds, subseeds, subseed_strength, seed_resize_from_h, seed_resize_from_w, controlnet_args, controlnet_weight, upscale
        return res

    def upscale_img(self,
                    img,
                    upscale_factor=None,
                    target_width=1024,
                    target_height=1024,
                    upscale_type="linear",
                    tile_width=512,
                    tile_height=512,
                    mask_blur=8,
                    upscaler=None,
                    padding=32,
                    seams_fix={},
                    seams_fix_enable=False):
        # resize img into target_width and target_height
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        init_image_h, init_image_w = img.size
        if upscale_factor is not None:
            target_width = int(init_image_h * upscale_factor)
            target_height = int(init_image_w * upscale_factor)
        self.mask_blur = mask_blur
        # only record for myself
        self.up_target_width = target_width
        self.up_target_height = target_height
        self.upscale_type = upscale_type
        self.upscaler = upscaler
        # resize image into up_target_width and up_target_height
        img = img.resize((target_width, target_height), PIL.Image.LANCZOS)
        draw = ultimate.USDURedraw(tile_height=tile_height,
                                   tile_width=tile_width,
                                   mode=upscale_type,
                                   padding=padding)
        rows = math.ceil(target_height / tile_height)
        cols = math.ceil(target_width / tile_width)
        res = draw.start(self, img, rows, cols)
        return res
    
    def tokenize_line(self, line, enable_emphasis=True, comma_padding_backtrack=0):
        """
        this transforms a single prompt into a list of PromptChunk objects - as many as needed to
        represent the prompt.
        Returns the list and the total number of tokens in the prompt.
        # """

        if enable_emphasis:
            parsed = parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.tokenizer([text for text, _ in parsed],
                                   padding=False, # caution 此处不padding只截断
                                   max_length=self.tokenizer.model_max_length,
                                   truncation=False,
                                   add_special_tokens=False).input_ids

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk
            id_start = self.tokenizer.bos_token_id
            id_end = self.tokenizer.eos_token_id
            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += 75

            to_add = 75 - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [id_start] + chunk.tokens + [id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]
            # chunk.tokens = chunk.tokens+[id_end]+[id_end]
            # chunk.multipliers =chunk.multipliers + [1.0]+[1.0]


            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == self.tokenizer.comma_token: # N
                    last_comma = len(chunk.tokens)

                # this is when we are at the end of alloted 75 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
                # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
                elif comma_padding_backtrack != 0 and len(chunk.tokens) == 75 \
                    and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack: # N
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == 75: # self.chunk_length: # N
                    next_chunk()
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count


    def tokenizer_forward(self, texts):
        """
        Accepts an array of texts; Passes texts through transformers network to create a tensor with numerical representation of those texts.
        Returns a tensor with shape of (B, T, C), where B is length of the array; T is length, in tokens, of texts (including padding) - T will
        be a multiple of 77; and C is dimensionality of each token - for SD1 it's 768, for SD2 it's 1024, and for SDXL it's 1280.
        An example shape returned by this function can be: (2, 77, 768).
        For SDXL, instead of returning one tensor avobe, it returns a tuple with two: the other one with shape (B, 1280) with pooled values.
        Webui usually sends just one text at a time through this function - the only time when texts is an array with more than one elemenet
        is when you do prompt editing: "a picture of a [cat:dog:0.4] eating ice cream"
        """

        def process_texts(texts):
            """
            Accepts a list of texts and calls tokenize_line() on each, with cache. Returns the list of results and maximum
            length, in tokens, of all texts.
            """

            token_count = 0

            cache = {}
            batch_chunks = []
            for line in texts:
                if line in cache:
                    chunks = cache[line]
                else:
                    chunks, current_token_count = self.tokenize_line(line) # _, 7
                    token_count = max(current_token_count, token_count) # 7, 0

                    cache[line] = chunks

                batch_chunks.append(chunks)
            return batch_chunks, token_count
    
        def process_tokens(remade_batch_tokens, batch_multipliers):
            """
            sends one single prompt chunk to be encoded by transformers neural network.
            remade_batch_tokens is a batch of tokens - a list, where every element is a list of tokens; usually
            there are exactly 77 tokens in the list. batch_multipliers is the same but for multipliers instead of tokens.
            Multipliers are used to give more or less weight to the outputs of transformers network. Each multiplier
            corresponds to one token.
            """

            z = self.text_encoder({"tokens": np.array(remade_batch_tokens, dtype=np.int32)})[0] ## tokens shape [1,77]

            # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
            batch_multipliers = np.array(batch_multipliers)
            original_mean = np.mean(z)
            z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).repeat(z.shape[0], axis=1)
            new_mean = np.mean(z)
            z = z * (original_mean / new_mean)

            return z

        batch_chunks, token_count = process_texts(texts)

        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            _fixes = [x.fixes for x in batch_chunk]

            for fixes in _fixes:
                for _position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding
            z = process_tokens(tokens, multipliers)
            zs.append(z)

        return np.hstack(zs)


    def __call__(
            self,
            prompt,
            negative_prompt=None,
            init_image:Image=None,
            mask=None,
            strength=0.5,
            controlnet_img:Image=None,
            num_inference_steps=32,
            guidance_scale=7.5,
            seeds=[10],
            subseeds=None,
            subseed_strength=0.0,
            using_paint=False,
            seed_resize_from_h=0,
            seed_resize_from_w=0,
            controlnet_args={},
            controlnet_weight=1.0,
            use_controlnet=True,
            init_latents=None
    ):
        seed_torch(seeds[0])
        init_steps = num_inference_steps
        using_paint = mask is not None and using_paint  # mask 不在就没有paint
        if self.controlnet_name and controlnet_img is None and init_image is not None and use_controlnet:
            controlnet_img = init_image
        self.controlnet_args = {}

        text_embeddings = self.tokenizer_forward([prompt])
        if guidance_scale > 1.0 or negative_prompt is not None:
            if negative_prompt is None:
                negative_prompt = ""
        uncond_embeddings = self.tokenizer_forward([negative_prompt])
        # 确保shape正确，填充或截断到77
        if uncond_embeddings.shape[1] > 77:
            uncond_embeddings = uncond_embeddings[:, :77]
        elif uncond_embeddings.shape[1] < 77:
            uncond_embeddings = np.pad(uncond_embeddings, ((0, 0), (0, 77 - uncond_embeddings.shape[1])), mode='constant', constant_values=0)

        if text_embeddings.shape[1] > 77:
            text_embeddings = text_embeddings[:, :77]
        elif text_embeddings.shape[1] < 77:
            text_embeddings = np.pad(text_embeddings, ((0, 0), (0, 77 - text_embeddings.shape[1])), mode='constant', constant_values=0)
        text_embeddings = np.concatenate((uncond_embeddings, text_embeddings), axis=0)
        
        # controlnet image prepare
        if controlnet_img is not None: # PIL Image
            controlnet_img = self.preprocess_controlnet_image(controlnet_img)
            if self.controlnet_name == "hed_multisize":
                controlnet_img = self._prepare_hed_image(controlnet_img)
            elif self.controlnet_name == "canny_multisize":
                controlnet_img = self._prepare_canny_image(controlnet_img)
            elif self.controlnet_name in ["tile_multisize"]:
                controlnet_img = controlnet_img
            else:
                raise NotImplementedError()
            controlnet_img = self._prepare_image(controlnet_img)
        # initialize latent latent
        if init_image is None and init_latents is None:
            init_timestep = num_inference_steps
        else:
            init_latents = torch.from_numpy(self._encode_image(init_image))
            init_timestep = int(num_inference_steps * strength) + 1
            init_timestep = min(init_timestep, num_inference_steps)
            num_inference_steps = init_timestep
        
        # handle latents
        shape = self.latent_shape
        # import pdb;pdb.set_trace()
        # 这里是生成随机数的地方 直接看这个函数 而且这个函数没有任何依赖
        latents = create_random_tensors(shape, seeds, subseeds=subseeds, subseed_strength=subseed_strength,
                                        seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        
        if init_image is not None and mask is not None:
            mask = self._preprocess_mask(mask)
        else:
            mask = None
        
        # run scheduler
        model_partical_fn = self.call_back_method()

        latents = sample(num_inference_steps, latents, self.scheduler,
                         guidance_scale=guidance_scale,
                         text_embedding=text_embeddings,
                         cond_img=controlnet_img,
                         mask=mask,
                         init_latents_proper=init_latents,
                         using_paint=using_paint,
                         model_partical_fn=model_partical_fn,
                         controlnet_weight=controlnet_weight,
                         init_steps=init_steps,
                         strength=strength,)
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder({"input.1": latents.astype(np.float32)})[0]
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0)* 255).astype(np.uint8)  # RGB
        return Image.fromarray(image)
