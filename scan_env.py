# scan.py

import modal

# This name is a placeholder, it's not important for this test.
app = modal.App("environment-scanner")

# This is the object we are searching for.
VOLUME_NAME = "eeg-assets"

@app.function()
def find_volume():
    """
    This function will attempt to access the volume.
    It will succeed if the volume exists in the environment this function runs in.
    It will fail with NotFoundError if it does not.
    """
    try:
        # The .reload() method forces an API call to the backend
        # to check if the object actually exists.
        modal.Volume.from_name(VOLUME_NAME).reload()
        
        # If the line above does not raise an error, we have found it.
        print(f"\nSUCCESS: Volume '{VOLUME_NAME}' was found in this environment.")
        
    except modal.NotFoundError:
        print(f"\nFAILURE: Volume '{VOLUME_NAME}' was NOT found in this environment.")

@app.local_entrypoint()
def main():
    find_volume.remote()