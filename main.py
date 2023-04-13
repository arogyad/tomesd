import torch, tomesd
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

tomesd.apply_patch(pipe, ratio=0.5)

images = pipe("a photo of an astronaut riding a horse on mars").images

for (i, image) in enumerate(images):
    image.save( "image_" + str(i) + ".jpg")
