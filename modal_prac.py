# train_benchmark.py

import os
import sys
import time
import json
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut

# Your project-specific modules. These will be found because app.py 
# correctly sets the Python path inside the container.
import SSVEPformer
import Constraint

# --- Model Hyperparameters ---
CHS_NUM = 9
MODEL_DEPTH = 2
ATTENTION_KERNEL_LENGTH = 31
MODEL_DROPOUT = 0.5

# --- Training Parameters ---
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001

def run_training():
    """
    Main function to run the entire training pipeline.
    """
    # --- 1. Get Paths from Modal Environment ---
    # Reads paths from environment variables, with a fallback for local execution.
    FEATURES_DIR = os.getenv("FEATURES_INPUT_DIR", default='features/benchmark_features')
    MODELS_OUTPUT_DIR = os.getenv("MODELS_OUTPUT_DIR", default='models/benchmark_models')
    FEATURES_FILE = os.path.join(FEATURES_DIR, 'benchmark_ssvepformer_input_final.npz')
    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

    # --- 2. Path Verification Block ---
    print("\n----------------- PATH VERIFICATION (TRAIN) -----------------")
    if os.getenv("MODAL_IS_RUNNING"):
        print("✅ Running on Modal.")
        print("   Paths successfully loaded from cloud environment.")
    else:
        print("🖥️  Running on Local Machine.")
        print("   Using local fallback default paths.")
    print(f"   => Features Path: {FEATURES_DIR}")
    print(f"   => Models Path:   {MODELS_OUTPUT_DIR}")
    print("-----------------------------------------------------------\n")

    # --- 3. Data Loading ---
    print(f"--- Loading data from: {FEATURES_FILE} ---")
    data = np.load(FEATURES_FILE)
    features = data['features']
    labels = data['labels']
    subject_indices = data['subject_indices']
    
    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).long()

    # --- 4. Cross-Validation Training Loop ---
    logo = LeaveOneGroupOut()
    NUM_SUBJECTS = len(np.unique(subject_indices))
    NUM_CLASSES = len(np.unique(labels))
    
    print("\n############################################################")
    print("           Starting SSVEPformer Benchmark Training (LOSO)           ")
    print("############################################################\n")
    start_time = time.time()

    for fold_num, (train_idx, _) in enumerate(logo.split(features, labels, subject_indices)):
        test_subject_id = fold_num + 1
        print(f"\n--- Training Fold {test_subject_id}/{NUM_SUBJECTS} (Holding out Subject {test_subject_id}) ---")

        X_train, y_train = features[train_idx], labels[train_idx]
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        net = SSVEPformer.SSVEPformer(depth=MODEL_DEPTH, attention_kernal_length=ATTENTION_KERNEL_LENGTH, 
                                      chs_num=CHS_NUM, class_num=NUM_CLASSES, dropout=MODEL_DROPOUT)
        net.apply(Constraint.initialize_weights)
        net = net.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

        net.train()
        for epoch in range(EPOCHS):
            for data in train_loader:
                X, y = data
                X, y = X.to(device), y.to(device)
                
                y_hat = net(X)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model_save_path = os.path.join(MODELS_OUTPUT_DIR, f'ssvepformer_fold_{test_subject_id}.pth')
        torch.save(net.state_dict(), model_save_path)
        print(f"--- Saved trained model for Fold {test_subject_id} to {model_save_path} ---")

    end_time = time.time()
    total_training_time = end_time - start_time
    print("\n--- All Training Folds Complete ---")
    print(f"Total training time: {total_training_time:.2f} seconds")

    # --- 5. Save Training Summary to JSON ---
    summary_data = {
        "training_completed_at": datetime.now().isoformat(),
        "total_training_time_seconds": round(total_training_time, 2),
        "parameters": {
            "num_subjects": NUM_SUBJECTS,
            "epochs_per_fold": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        }
    }
    summary_path = os.path.join(MODELS_OUTPUT_DIR, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"--- Saved training summary to {summary_path} ---")

if __name__ == "__main__":
    run_training()