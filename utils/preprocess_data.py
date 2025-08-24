# preprocess_benchmark_data_final.py

import os
import numpy as np
import scipy.io

def preprocess_benchmark_data_final():
    """
    Loads and preprocesses the 2017 Benchmark Dataset ("Dataset 2") to
    PERFECTLY REPLICATE the input format of the SSVEPformer paper.
    """
    # --- 0. Configuration ---
    RAW_DATA_PATH = '../benchmark_data'
    CHANNEL_LOC_PATH = '64-channels.loc'
    OUTPUT_FILE_PATH = '../features/benchmark_ssvepformer_input_final.npz'
    os.makedirs('../features', exist_ok=True)

    # --- Parameters from the SSVEPformer Paper for Dataset 2 ---
    NUM_SUBJECTS = 35
    TARGET_CHANNELS = ['O1', 'Oz', 'O2', 'PO3', 'POz', 'PO4', 'PZ', 'PO5', 'PO6']
    
    ORIGINAL_FS = 250
    WINDOW_LENGTH_S = 1.0
    LATENCY_S = 0.64
    
    TARGET_FREQ_RESOLUTION = 0.2
    FREQ_RANGE_START = 8
    FREQ_RANGE_END = 64

    # --- Step 1: Channel Selection ---
    all_channel_names = np.loadtxt(CHANNEL_LOC_PATH, dtype=str, usecols=3)
    target_channel_indices = [np.where(all_channel_names == name)[0][0] for name in TARGET_CHANNELS]
    print(f"Selected {len(target_channel_indices)} channels: {TARGET_CHANNELS}")

    # --- Step 2: Load and Slice All Subject Data ---
    all_subjects_data, all_subjects_labels, all_subjects_indices = [], [], []
    start_sample = int(LATENCY_S * ORIGINAL_FS)
    end_sample = int((LATENCY_S + WINDOW_LENGTH_S) * ORIGINAL_FS)
    num_samples_in_window = end_sample - start_sample

    print(f"Loading and slicing all subjects to a {WINDOW_LENGTH_S}s window (samples {start_sample}:{end_sample})...")
    
    for subject_id in range(1, NUM_SUBJECTS + 1):
        filepath = os.path.join(RAW_DATA_PATH, f'S{subject_id}.mat')
        mat_data = scipy.io.loadmat(filepath)['data']
        
        data_sliced = mat_data[target_channel_indices, start_sample:end_sample, :, :]
        data_permuted = data_sliced.transpose(3, 2, 0, 1)
        data_reshaped = data_permuted.reshape(-1, len(target_channel_indices), num_samples_in_window)
        
        labels = np.tile(np.arange(40), 6)
        subject_indices = np.full(240, subject_id)
        
        all_subjects_data.append(data_reshaped)
        all_subjects_labels.append(labels)
        all_subjects_indices.append(subject_indices)

    final_time_domain_data = np.concatenate(all_subjects_data, axis=0)
    final_labels = np.concatenate(all_subjects_labels, axis=0)
    final_subject_indices = np.concatenate(all_subjects_indices, axis=0)
    
    # --- Step 3: FFT and Complex Spectrum Creation (Corrected) ---
    print("Applying FFT and creating complex spectrum input...")
    
    NFFT = round(ORIGINAL_FS / TARGET_FREQ_RESOLUTION)
    
    # Added amplitude normalization to match the author's code
    fft_result = np.fft.fft(final_time_domain_data, n=NFFT, axis=-1) / (num_samples_in_window / 2)
    
    freqs = np.fft.fftfreq(NFFT, 1.0/ORIGINAL_FS)
    freq_indices = np.where((freqs >= FREQ_RANGE_START) & (freqs < FREQ_RANGE_END))[0]
    
    real_part = np.real(fft_result[:, :, freq_indices])
    imag_part = np.imag(fft_result[:, :, freq_indices])
    
    # Concatenate along the last axis (-1) to match the author's code
    final_model_input = np.concatenate([real_part, imag_part], axis=-1)
    
    print(f"Created complex spectrum features. Final model input shape: {final_model_input.shape}")

    # --- 4. Save the final data ---
    print(f"Saving final preprocessed data to: {OUTPUT_FILE_PATH}")
    np.savez_compressed(
        OUTPUT_FILE_PATH,
        features=final_model_input,
        labels=final_labels,
        subject_indices=final_subject_indices
    )
    print("Preprocessing complete!")

if __name__ == '__main__':
    preprocess_benchmark_data_final()