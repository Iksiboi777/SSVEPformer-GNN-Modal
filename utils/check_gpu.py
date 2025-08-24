# check_gpu.py
import modal
import subprocess

app = modal.App("gpu-check")

@app.function(gpu="A100")
def check_my_gpu():
    print("--- GPU Check ---")
    # The 'nvidia-smi' command is the standard way to check NVIDIA GPU status.
    try:
        subprocess.run(["nvidia-smi"], check=True)
        print("\nSUCCESS: A100 GPU is allocated and accessible.")
    except FileNotFoundError:
        print("ERROR: nvidia-smi command not found. This might indicate a driver issue.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: nvidia-smi command failed with exit code {e.returncode}.")

@app.local_entrypoint()
def main():
    print("Submitting a quick job to check for A100 availability...")
    check_my_gpu.remote()