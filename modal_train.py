# modal_train.py (with Inspector)

import modal
import os
import subprocess # We'll use this to run simple commands

# We import the image definition from our factory.
from build_image import image

app = modal.App("ssvepformer-main")

# Get handles to our two "Memory Cards".
model_storage = modal.Volume.from_name("ssvepformer-model-storage", create_if_missing=True)
data_volume = modal.Volume.from_name("eeg-assets")

# =====================================================================================
# ## INSPECTOR FUNCTION ##
# This is a temporary, fast function to verify our setup.
# =====================================================================================
@app.function(
    image=image, # Use the same "Console" as our training job
    volumes={          # Plug in the same two "Memory Cards"
        "/root/project/models": model_storage,
        "/root/project": data_volume
    }
)
def inspect_environment():
    print("--- 🕵️‍♂️ INSPECTING CONTAINER ENVIRONMENT 🕵️‍♂️ ---")
    
    print("\n[1] Current Working Directory (from Dockerfile's WORKDIR):")
    subprocess.run(["pwd"])
    
    print("\n[2] Contents of /root/project (your code):")
    subprocess.run(["ls", "-l", "/root/project"])
    
    print("\n[3] Contents of /eeg_data (your eeg-assets volume):")
    subprocess.run(["ls", "-l", "/eeg_data"])
    
    print("\n[4] Contents of /root/project/models (your model_storage volume):")
    subprocess.run(["ls", "-l", "/root/project/models"])
    
    print("\n--- ✅ INSPECTION COMPLETE ---")

# =====================================================================================
# ## TRAINING FUNCTION ##
# This is your real training function. We won't run it yet.
# =====================================================================================
@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/root/project/models": model_storage,
        "/root/project": data_volume
    },
    timeout=7200
)
def train_ssvepformer():
    # ... (all your training logic is here, unchanged) ...
    pass # We can just pass for now since we aren't running it.

# =====================================================================================
# ## ENTRYPOINT ##
# This is what runs when you type `modal run`.
# =====================================================================================
@app.local_entrypoint()
def main():
    # Let's run our new inspector function first!
    print("🚀 Launching a quick inspection job...")
    inspect_environment.remote()
    
    # Once we're happy, we can comment the line above and uncomment the one below.
    # print("🚀 Launching the full training job...")
    # train_ssvepformer.remote()