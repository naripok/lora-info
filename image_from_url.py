import numpy
import requests
import torch
from PIL import Image, ImageOps


class ImageFromURL:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"url": ("STRING", {"multiline": False})},
        }

    RETURN_NAMES = ("image",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fetch_image"
    OUTPUT_NODE = True
    CATEGORY = "jitcoder"

    def fetch_image(self, url):
        if not url:
            # Return an empty square transparent image
            # HWC format (batch_size, dim0, dim1, channels)
            return (torch.zeros((1, 1, 1, 3), dtype=torch.float32),)

        # Load the image from the url
        image = Image.open(requests.get(url, stream=True).raw)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = numpy.array(image).astype(numpy.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)
