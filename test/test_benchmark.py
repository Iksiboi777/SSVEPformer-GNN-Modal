# test_benchmark.py

import os
import sys
import time
import json
from datetime import datetime
import numpy as np
import torch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

# Your project-specific modules
import SSVEPformer
import Constraint

# --- Model Hyperparameters (must match training script) ---
CHS_NUM = 9
MODEL_DEPTH = 2
ATTENTION_KERNEL_LENGTH = 31
MODEL_DROPOUT = 0.5

# --- Testing Parameters ---
# NOTE: Adjust this value to the duration of a single trial in your dataset (in seconds).
# This is required for the ITR calculation.
TRIAL_DURATION_SECONDS = 5.0 

def calculate_itr(n_classes, accuracy, time_per_trial):
    """Calculates Information Transfer Rate (ITR) in bits per minute."""
    if accuracy == 0:
        return 0.0
    if accuracy == 1:
        # Prevent log(0) for perfect accuracy by using a slightly smaller value
        accuracy = 1.0 - 1e-10
        
    term1 = np.log2(n_classes)
    term2 = accuracy * np.log2(accuracy)
    term3 = ((1 - accuracy) / (n_classes - 1)) * np.log2((1 - accuracy) / (n_classes - 1))
    
    bits_per_trial = term1 + term2 + (n_classes - 1) * term3
    itr_bits_per_minute = (bits_per_trial * 60) / time_per_trial
    return itr_bits_per_minute

def run_testing():
    """
    Main function to run the entire testing and evaluation pipeline.
    """
    # --- 1. Get Paths from Modal Environment ---
    FEATURES_DIR = os.getenv("FEATURES_INPUT_DIR", default='ssvep_features/benchmark_features')
    MODELS_INPUT_DIR = os.getenv("MODELS_INPUT_DIR", default='ssvep_models/benchmark_models')
    FEATURES_FILE = os.path.join(FEATURES_DIR, 'benchmark_ssvepformer_input_final.npz')
    # os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

    # --- 2. Path Verification Block ---
    print("\n----------------- PATH VERIFICATION (TEST) ------------------")
    if os.getenv("MODAL_IS_RUNNING"):
        print("✅ Running on Modal.")
        print("   Paths successfully loaded from cloud environment.")
    else:
        print("🖥️  Running on Local Machine.")
        print("   Using local fallback default paths.")
    print(f"   => Features Path: {FEATURES_DIR}")
    print(f"   => Models Path:   {MODELS_INPUT_DIR}")
    print("-----------------------------------------------------------\n")

    # --- 3. Data Loading ---
    print(f"--- Loading data from: {FEATURES_FILE} ---")
    data = np.load(FEATURES_FILE)
    features = data['features']
    labels = data['labels']
    subject_indices = data['subject_indices']

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).long()

    # --- 4. Cross-Validation Testing Loop ---
    logo = LeaveOneGroupOut()
    NUM_SUBJECTS = len(np.unique(subject_indices))
    NUM_CLASSES = len(np.unique(labels))

    all_accuracies = []
    all_itrs = []

    print("\n###########################################################")
    print("           Starting SSVEPformer Benchmark Testing (LOSO)           ")
    print("###########################################################\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold_num, (_, test_idx) in enumerate(logo.split(features, labels, subject_indices)):
        test_subject_id = fold_num + 1
        print(f"\n--- Testing Fold {test_subject_id}/{NUM_SUBJECTS} (Testing on Subject {test_subject_id}) ---")

        X_test, y_test = features[test_idx], labels[test_idx]
        
        # Load the corresponding trained model
        model_path = os.path.join(MODELS_INPUT_DIR, f'ssvepformer_fold_{test_subject_id}.pth')
        print(f"Loading model from: {model_path}")

        net = SSVEPformer.SSVEPformer(depth=MODEL_DEPTH, attention_kernal_length=ATTENTION_KERNEL_LENGTH, 
                                      chs_num=CHS_NUM, class_num=NUM_CLASSES, dropout=MODEL_DROPOUT)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net = net.to(device)
        net.eval()

        with torch.no_grad():
            outputs = net(X_test.to(device))
            _, predicted = torch.max(outputs.data, 1)
            
            y_test_np = y_test.cpu().numpy()
            predicted_np = predicted.cpu().numpy()

            acc = accuracy_score(y_test_np, predicted_np)
            itr = calculate_itr(NUM_CLASSES, acc, TRIAL_DURATION_SECONDS)
            
            all_accuracies.append(acc)
            all_itrs.append(itr)
            
            print(f"Subject {test_subject_id} Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"Subject {test_subject_id} ITR: {itr:.2f} bits/min")

    # --- 5. Calculate Final Metrics ---
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    mean_itr = np.mean(all_itrs)
    std_itr = np.std(all_itrs)

    # --- 6. Save Test Results to JSON ---
    results_data = {
        "testing_completed_at": datetime.now().isoformat(),
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "mean_itr_bits_per_min": round(mean_itr, 2),
        "std_itr": round(std_itr, 4),
        "parameters": {
            "num_subjects": NUM_SUBJECTS,
            "trial_duration_seconds_for_itr": TRIAL_DURATION_SECONDS
        }
    }
    results_path = os.path.join(MODELS_INPUT_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"\n--- Saved final test results to {results_path} ---")

    return mean_acc, mean_itr, std_acc, std_itr

if __name__ == "__main__":
    run_testing()