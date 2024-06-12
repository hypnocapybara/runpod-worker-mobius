'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64
import concurrent.futures

import torch

from diffusers import StableDiffusion3Pipeline

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #


class ModelHandler:
    def __init__(self):
        self.base = None
        self.load_models()

    def load_base(self):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda", silence_dtype_warnings=True)
        return pipe

    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_base = executor.submit(self.load_base)

            self.base = future_base.result()


MODELS = ModelHandler()

# ---------------------------------- Helper ---------------------------------- #


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    # Generate latent image using pipe
    images = MODELS.base(
        prompt=job_input['prompt'],
        negative_prompt=job_input['negative_prompt'],
        height=job_input['height'],
        width=job_input['width'],
        num_inference_steps=job_input['num_inference_steps'],
        guidance_scale=job_input['guidance_scale'],
        denoising_end=job_input['high_noise_frac'],
        num_images_per_prompt=job_input['num_images'],
        generator=generator
    ).images

    image_urls = _save_and_upload_images(images, job['id'])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    return results


runpod.serverless.start({"handler": generate_image})
