import json
import os
import io
from PIL import Image

from flask import Flask, request


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    from . import clip_image_encoding_model
    clip_image_encoding_model.init_app(app)
    
    # trigger any caching that needs to happen
    im = Image.new("RGB", (240, 240))
    clip_image_encoding_model.ClipImageEncoder().encode_image([im])

    # a simple page that says hello
    @app.post("/api/v1/encode-image")
    def encode_text():
        
        im = Image.open(io.BytesIO(request.data))
        encoding = clip_image_encoding_model.get_image_encoding_model().encode_image([im])
        return json.dumps(encoding, indent = 4)

    return app