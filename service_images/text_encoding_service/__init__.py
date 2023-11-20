import json
import os

from flask import Flask, request


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    from . import clip_text_encoding_model
    clip_text_encoding_model.init_app(app)

    # trigger any caching that needs to happen
    clip_text_encoding_model.ClipTextEncoder().encode_text(["text"])

    # a simple page that says hello
    @app.post("/api/v1/encode-text")
    def encode_text():
        encoding = clip_text_encoding_model.get_text_encoding_model().encode_text(request.json)
        return json.dumps(encoding, indent = 4)

    return app