# import math
# import torch

# from tqdm.notebook import tqdm
# from PIL import Image, ImageOps
# import torch
# from torchvision import transforms

# from diffusers import DDIMScheduler, DDIMInverseScheduler
# from Grounded_Instruct_Pix2Pix.external_mask_extractor import ExternalMaskExtractor
# from Grounded_Instruct_Pix2Pix.pipeline_stable_diffusion_grounded_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline

# from transformers import logging
# import random
# logging.set_verbosity_error()

# class GroundedInstructPixtoPix():
#     def __init__(self, num_timesteps: int = 100, device: str = 'cuda:0', image_guidance_scale: float = 1.5, 
#                  text_guidance_scale: float = 7.5, start_blending_at_tstep: int = 100,
#                  end_blending_at_tstep: int = 1, prompt: str = '', seed: int = 42, 
#                  verbose: bool = False):
        
#         self.num_timesteps = num_timesteps
#         self.device = device
#         self.image_guidance_scale = image_guidance_scale
#         self.text_guidance_scale = text_guidance_scale
#         self.prompt = prompt
#         self.seed = seed
#         self.verbose = verbose
#         self.blending_range = [start_blending_at_tstep,end_blending_at_tstep]
#         self.mask_extractor = ExternalMaskExtractor(device=self.device)
#         self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
#                                                                                 torch_dtype=torch.float16,
#                                                                                 safety_checker=None).to(self.device)
#         self.pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipeline.scheduler.config, set_alpha_to_zero=False) #noising

#         self.pipeline.scheduler.set_timesteps(self.num_timesteps) #denoising
#         self.pipeline.inverse_scheduler.set_timesteps(self.num_timesteps)


#     def load_pil_image(image_path, resolution=512):
#         image = Image.open(image_path).convert("RGB")
#         width, height = image.size
#         factor = resolution / max(width, height)
#         factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
#         width = int((width * factor) // 64) * 64
#         height = int((height * factor) // 64) * 64
#         image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
#         return image



#     def edit_image(self, image):
#         to_pil = transforms.ToPILImage()
#         image_pil = to_pil(image)
#         external_mask_pil, chosen_noun_phrase, clip_scores_dict = self.mask_extractor.get_external_mask(image_pil, self.prompt, verbose=self.verbose)
#         inv_results = self.pipeline.invert(self.prompt, image_pil, num_inference_steps=self.num_timesteps, inv_range=self.blending_range) #noising
#         generator = torch.Generator(self.device).manual_seed(self.seed) if self.seed is not None else torch.Generator(self.device)
#         edited_image = self.pipeline(self.prompt, src_mask=external_mask_pil, image=image_pil,
#                                 guidance_scale=self.text_guidance_scale, image_guidance_scale=self.image_guidance_scale,
#                                 num_inference_steps=self.num_timesteps, generator=generator).images[0] #denoising
#         to_tensor = transforms.ToTensor()
#         tensor_image = to_tensor(edited_image)
#         return tensor_image