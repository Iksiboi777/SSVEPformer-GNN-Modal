# app.py (Final, Docs-Compliant Version)

import modal
import os
import sys
import subprocess

# --- PART 1: COMMON SETUP ---
# This part is correct and remains the same.

from build_image import image as ssvep_image
app = modal.App("ssvepformer-main")

model_storage = modal.Volume.from_name("model-storage-volume", create_if_missing=True)
data_storage = modal.Volume.from_name("eeg-data", create_if_missing=True)
feature_storage = modal.Volume.from_name("feature-storage", create_if_missing=True)  # ## ADDED ##

# project_dir = "/root/mtgnet"
# sys.path.append(project_dir)
# --- PART 2: CORE REMOTE FUNCTIONS ---
# These are the functions that will run in the cloud. We define them once.

@app.function(
    image=ssvep_image,
    volumes={"/root/mtgnet/ssvep_models": model_storage, 
             "/root/mtgnet/data": data_storage,
             "/root/mtgnet/ssvep_features": feature_storage},  # ## ADDED ##
    timeout=7200
)
def inspect_environment():
    print("--- 🕵️‍♂️ INSPECTING CONTAINER ENVIRONMENT 🕵️‍♂️ ---")

    print("\n[1] Current Working Directory (from Dockerfile's WORKDIR):")
    subprocess.run(["pwd"])

    print("\n[2] Contents of /root/mtgnet (your code):")
    subprocess.run(["ls", "-lR", "/root/mtgnet"])

    print("\n[3] Contents of /root/mtgnet/data (your data volume):")
    subprocess.run(["ls", "-lR", "/root/mtgnet/data/"])

    print("\n[4] Contents of /root/mtgnet/ssvep_models (your models volume):")
    subprocess.run(["ls", "-lR", "/root/mtgnet/ssvep_models/"])

    print("\n[5] Contents of /root/mtgnet/ssvep_features (your features volume):")
    subprocess.run(["ls", "-lR", "/root/mtgnet/ssvep_features/"])

    print("\n--- ✅ INSPECTION COMPLETE ---")

@app.function(
    image=ssvep_image,
    gpu="A100",
    volumes={"/root/mtgnet/ssvep_models": model_storage, 
             "/root/mtgnet/data": data_storage,
             "/root/mtgnet/ssvep_features": feature_storage},  # ## ADDED ##
    timeout=7200
)
def train_ssvepformer():
    print("--- 🚀 Starting Training on Modal GPU ---")
    os.chdir("/root/mtgnet")
    sys.path.append("/root/mtgnet")

    # ## FIX ##: All environment variables now point to the correct mount paths.
    os.environ["MODELS_OUTPUT_DIR"] = "/root/mtgnet/ssvep_models/benchmark_models"
    os.environ["DATASET_INPUT_DIR"] = "/root/mtgnet/data/benchmark_data"
    os.environ["FEATURES_INPUT_DIR"] = "/root/mtgnet/ssvep_features/benchmark_features"
    
    from train_benchmark import run_training
    run_training()

    print("--- ✅ Training Complete ---")
    model_storage.commit()
    data_storage.commit()
    feature_storage.commit()

@app.function(
    image=ssvep_image,
    gpu="A100",
    volumes={"/root/mtgnet/ssvep_models": model_storage, 
             "/root/mtgnet/data": data_storage,
             "/root/mtgnet/ssvep_features": feature_storage},  # ## ADDED ##
    timeout=1800
)
def test_ssvepformer():
    print("--- 🧪 Starting Testing on Modal GPU ---")
    model_storage.reload()

    os.chdir("/root/mtgnet")
    sys.path.append("/root/mtgnet")

    os.environ["MODELS_INPUT_DIR"] = "/root/mtgnet/ssvep_models/benchmark_models"
    os.environ["DATASET_INPUT_DIR"] = "/root/mtgnets/data/benchmark_data"
    os.environ["FEATURES_INPUT_DIR"] = "/root/mtgnet/ssvep_features/benchmark_features"

    from test_benchmark import run_testing
    mean_acc, mean_itr, std_acc, std_itr = run_testing()
    return mean_acc, mean_itr, std_acc, std_itr

# --- PART 3: COMMAND-LINE ENTRYPOINTS ---
# We use @app.local_entrypoint to create our commands.

@app.local_entrypoint()
def inspect():
    """Runs the inspection function to check the environment."""
    print("🚀 Launching a quick inspection job...")
    inspect_environment.remote()

@app.local_entrypoint()
def train():
    """Runs the full training process."""
    print("--- Submitting remote training job to Modal... ---")
    train_ssvepformer.remote()
    print("--- Training job sent. Monitor the link above for progress. ---")

@app.local_entrypoint()
def test():
    """Runs the evaluation on the trained models."""
    print("--- Submitting remote testing job to Modal... ---")
    mean_acc, mean_itr, std_acc, std_itr = test_ssvepformer.remote()
    print("\n############################################################")
    print("           MODAL RUN COMPLETE - FINAL RESULTS           ")
    print("############################################################\n")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"Mean ITR: {mean_itr:.2f} ± {std_itr:.4f} bits/min")
    print("\n############################################################")