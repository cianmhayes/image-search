from PIL import Image
from transformers import AutoProcessor, CLIPVisionModelWithProjection

from typing import List

import click
from flask import g


class ClipImageEncoder(object):
    def __init__(self):
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, ims:List[Image.Image]):
        inputs = self.processor(images=ims, return_tensors="pt")
        return self.model(**inputs).image_embeds.tolist()


def get_image_encoding_model() -> ClipImageEncoder:
    if 'image_encoder' not in g:
        g.image_encoder = ClipImageEncoder()

    return g.image_encoder

@click.command('init-image-encoding-model')
def init_image_encoding_model_command():
    get_image_encoding_model()
    click.echo('Initialized the encoding model.')

def init_app(app):
    app.cli.add_command(init_image_encoding_model_command)
