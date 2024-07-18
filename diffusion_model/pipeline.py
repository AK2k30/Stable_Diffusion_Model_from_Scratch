import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler # type: ignore

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

"""
Generate an image using a diffusion model.

Args:
    prompt (str): The text prompt to use for image generation.
    uncond_prompt (str): The unconditional text prompt to use for image generation.
    input_image (PIL.Image.Image, optional): An input image to use as a starting point for the diffusion process.
    strength (float, optional): The strength of the input image, between 0 and 1. A value of 0 means the output image will be completely generated from the prompt, while a value of 1 means the output image will be a blend of the input image and the generated image.
    do_cfg (bool, optional): Whether to use conditional guidance during the diffusion process.
    cfg_scale (float, optional): The scale factor for conditional guidance.
    sampler_name (str, optional): The name of the diffusion sampler to use, currently only "ddpm" is supported.
    n_inference_steps (int, optional): The number of inference steps to perform during the diffusion process.
    models (dict, optional): A dictionary of pre-loaded models to use for the diffusion process.
    seed (int, optional): A seed value to use for the random number generator.
    device (torch.device, optional): The device to use for the diffusion process.
    idle_device (torch.device, optional): A device to use for idle operations.
    tokenizer (transformers.PreTrainedTokenizer, optional): A tokenizer to use for encoding the text prompts.

Returns:
    PIL.Image.Image: The generated image.
"""
def generate(prompt= str, uncond_prompt= str, input_image=None, strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name="ddpm", n_inference_steps=50, models={}, seed=None, device=None, idle_device=None, tokenizer=None):

    with torch.no_grad():
        if not(0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
           generator.manual_seed(seed) 
        
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image: 
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)

        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)

        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

"""
Rescales a tensor `x` from an old range to a new range, optionally clamping the values to the new range.

Args:
    x (torch.Tensor): The input tensor to be rescaled.
    old_range (tuple[float, float]): The minimum and maximum values of the old range.
    new_range (tuple[float, float]): The minimum and maximum values of the new range.
    clamp (bool, optional): If True, the rescaled values will be clamped to the new range. Defaults to False.

Returns:
    torch.Tensor: The rescaled tensor.
"""
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

"""
Computes a time embedding for a given timestep.

Args:
    timestep (torch.Tensor): A tensor of timesteps, with shape (batch_size,).

Returns:
    torch.Tensor: A tensor of time embeddings, with shape (batch_size, 160).
"""
def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
