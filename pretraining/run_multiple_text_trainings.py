"""
Wrapper script to run text-based T5 training for multiple balanced_number values sequentially.
This modifies the pre_train_flant5_text.py configuration and runs it multiple times.
"""

import subprocess
import sys
import os

PROPORTION = "IMBALANCED"  # Options: BALANCED, IMBALANCED

# Configuration imbalanced - list of (numbers, learning_rate) tuples
IMBALANCED_CONFIGS = [
    ([8], 5e-4),
    ([40, 100], 5e-5),
    ([200], 2e-5)
]

# Configuration balanced - list of (numbers, learning_rate) tuples
BALANCED_CONFIGS = [
    ([5, 20, 50], 5e-5),
    ([1, 3, 4], 5e-4),
    ([20, 50, 100], 2e-5)
]

REF_CONFIGS = IMBALANCED_CONFIGS if PROPORTION == "IMBALANCED" else BALANCED_CONFIGS

SCRIPT_PATH = "pretraining/pre_train_flant5_text.py"

def update_config(ref_num, learning_rate, proportion):
    """Update the balanced/imbalanced_number, learning_rate, and proportion in the training script"""
    with open(SCRIPT_PATH, 'r') as f:
        content = f.read()
    
    # Replace the configuration lines
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('PROPORTION ='):
            lines[i] = f'PROPORTION = "{proportion}"  # Options: BALANCED, IMBALANCED'
        elif proportion == "BALANCED" and line.strip().startswith('balanced_number ='):
            lines[i] = f'    balanced_number = {ref_num}'
        elif proportion == "IMBALANCED" and line.strip().startswith('imbalanced_number ='):
            lines[i] = f'    imbalanced_number = {ref_num}'
        elif line.startswith('LEARNING_RATE ='):
            lines[i] = f'LEARNING_RATE = {learning_rate} if USE_LORA else {learning_rate}  # Higher for LoRA'
    
    with open(SCRIPT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Updated {proportion.lower()}_number to {ref_num}")
    print(f"✓ Updated LEARNING_RATE to {learning_rate}")
    print(f"✓ Updated PROPORTION to {proportion}")


def main():
    print(f"\n{'='*80}")
    print(f"SEQUENTIAL TRAINING FOR {PROPORTION} CONFIGURATIONS")
    print(f"{'='*80}\n")
    
    total_runs = sum(len(numbers) for numbers, _ in REF_CONFIGS)
    current_run = 0
    
    for config_idx, (ref_numbers, learning_rate) in enumerate(REF_CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"CONFIG GROUP {config_idx}/{len(REF_CONFIGS)}: LR = {learning_rate}, Numbers = {ref_numbers}")
        print(f"{'='*80}\n")
        
        for ref_num in ref_numbers:
            current_run += 1
            print(f"\n{'='*80}")
            print(f"TRAINING RUN {current_run}/{total_runs}: {PROPORTION.lower()}_number = {ref_num}, LR = {learning_rate}")
            print(f"{'='*80}\n")
            
            # Update the script
            update_config(ref_num, learning_rate, PROPORTION)
            
            # Run the training script
            result = subprocess.run(
                [sys.executable, SCRIPT_PATH],
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                print(f"\n❌ Training failed for {PROPORTION.lower()}_number={ref_num}, LR={learning_rate}")
                print(f"Stopping sequential training.")
                sys.exit(1)
            
            print(f"\n✓ Completed training for {PROPORTION.lower()}_number={ref_num}, LR={learning_rate}")
    
    print(f"\n{'='*80}")
    print(f"ALL TRAINING RUNS COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
