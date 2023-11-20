from transformers import AutoTokenizer, CLIPTextModelWithProjection
from typing import List

import click
from flask import g


class ClipTextEncoder(object):
    def __init__(self):
        self.model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def encode_text(self, t:List[str]):
        inputs = self.tokenizer(text=t, return_tensors="pt", padding=True)
        return self.model(**inputs).text_embeds.tolist()


def get_text_encoding_model() -> ClipTextEncoder:
    if 'text_encoder' not in g:
        g.text_encoder = ClipTextEncoder()

    return g.text_encoder

@click.command('init-encoding-model')
def init_encoding_model_command():
    get_text_encoding_model()
    click.echo('Initialized the encoding model.')

def init_app(app):
    app.cli.add_command(init_encoding_model_command)
