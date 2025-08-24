# modal_test.py

import modal
import os
from pathlib import Path

# --- 1. Define the Environment and Shared Volume (must be identical to the training script) ---
app = modal.App("ssvepformer-test")

# ssvep_image = (
#     modal.Image.micromamba(python_version="3.9")
#     # CORRECTED: Using the .micromamba_install method from the documentation
#     .micromamba_install(
#         [
#             "numpy=1.22.3",
#             "pandas=1.4.3",
#             "scipy=1.9.0",
#             "matplotlib=3.5.2",
#             "pyyaml=6.0.1",
#             "scikit-learn=1.1.1",
#             "pytorch=1.10.1",
#             "torchvision=0.11.2",
#             "torchaudio=0.10.1",
#             "cudatoolkit=11.3",
#         ],
#         channels=["pytorch", "nvidia", "conda-forge"],
#     )
#     .pip_install("einops==0.8.0")
#     .add_local_dir(".", remote_path="/root/project")  # Mounts the current directory
# )

ssvep_image = modal.Image.from_dockerfile("Dockerfile")

model_storage = modal.Volume.from_name("ssvepformer-model-storage")


# --- 2. Define the Remote Testing Function ---
@app.function(
    image=ssvep_image,
    gpu="A100", # Using a GPU can speed up testing as well
    volumes={"/root/project/models": model_storage},
    timeout=1800 # 30-minute timeout for testing
)
def test_ssvepformer():
    """A Modal function that runs the testing and ITR calculation."""
    import sys
    print("--- Running Testing on Modal GPU ---")
    
    # Best practice: reload the volume to make sure we have the latest changes
    # committed by the training script.
    model_storage.reload()
    
    # os.chdir("/root/project")
    # sys.path.append("/root/project")
    
    os.environ["MODELS_INPUT_DIR"] = "/root/project/models/benchmark_ssvepformer"    

    # Import and call your testing function
    from test_benchmark import run_testing
    mean_acc, mean_itr, std_acc, std_itr = run_testing()
    
    return mean_acc, mean_itr, std_acc, std_itr


# --- 3. Define the Local Entrypoint to Run the Job and Get Results ---
@app.local_entrypoint()
def main():
    print("--- Submitting remote testing job to Modal... ---")
    mean_acc, mean_itr, std_acc, std_itr = test_ssvepformer.remote()
    
    print("\n############################################################")
    print("            MODAL RUN COMPLETE - FINAL RESULTS            ")
    print("############################################################\n")
    print(f"Mean Accuracy across all 35 subjects: {mean_acc:.4f}± {std_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"Mean ITR across all 35 subjects: {mean_itr:.2f}± {std_itr:.4f} bits/min")
    print("\n############################################################")