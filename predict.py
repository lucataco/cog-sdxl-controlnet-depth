from cog import BasePredictor, Input, Path
import os
import time
import torch
import shutil
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image

CONTROL_MODEL = "diffusers/controlnet-depth-sdxl-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
FEATURE_NAME = "Intel/dpt-hybrid-midas"
CONTROL_CACHE = "control-cache"
VAE_CACHE = "vae-cache"
MODEL_CACHE = "sdxl-cache"
FEATURE_CACHE = "feature-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        t1 = time.time()
        print("Loading depth feature extractor")
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(FEATURE_CACHE).to("cuda")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(FEATURE_CACHE)
        print("Loading controlnet depth model")
        controlnet = ControlNetModel.from_pretrained(
            CONTROL_CACHE,
            use_safetensors=True,
            torch_dtype=torch.float16
        )
        print("Loading better VAE")
        better_vae = AutoencoderKL.from_pretrained(
            VAE_CACHE,
            torch_dtype=torch.float16
        )
        print("Loading sdxl")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            vae=better_vae,
            controlnet=controlnet,
            use_safetensors=True,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        self.pipe = pipe.to("cuda")
        t2 = time.time()
        print("Setup took: ", t2 - t1)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")
    
    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image for controlnet",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="spiderman lecture, photorealistic",
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps", ge=1, le=100, default=30
        ),
        condition_scale: float = Input(
            description="controlnet conditioning scale for generalization",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        seed: int = Input(
            description="Random seed. Set to 0 to randomize the seed", default=0
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed is None) or (seed <= 0):
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")

        image = self.load_image(image)
        depth_image = self.get_depth_map(image)

        images = self.pipe(
            prompt,
            image=depth_image,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=condition_scale,
            generator=generator
        ).images

        output_path = f"/tmp/output.png"
        images[0].save(output_path)

        return Path(output_path)
