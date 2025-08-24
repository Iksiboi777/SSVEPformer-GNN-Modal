# build_image.py

import modal

# Give the App a name. This name is part of how we will find the image later.
# app = modal.App("ssvepformer-base")

# This is the image definition. When we deploy this script,
# Modal will build this image and register it with the name 'ssvepformer-image'.
image = modal.Image.from_dockerfile("Dockerfile")

# This is a dummy function. We need at least one function in the app
# for the deployment to register the image.
# @app.function(image=image)
# def dummy():
#     pass