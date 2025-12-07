"""
Wrapper script to run T5 training for multiple balanced_number values sequentially.
This modifies the pre_train_flant5.py configuration and runs it multiple times.
"""

import subprocess
import sys
import os

PROPORTION = "IMBALANCED"  # Options: BALANCED, IMBALANCED

# Configuration balanced
# BALANCED_NUMBERS = [5, 20, 50]  # 5e-5
# LEARNING_RATE = 5e-5  # Set the learning rate for all runs

# BALANCED_NUMBERS = [1,3,4]  # 5e-4
# LEARNING_RATE = 5e-4  # Set the learning rate for all runs

# BALANCED_NUMBERS = [20,50,100]  # 2e-5
# LEARNING_RATE = 2e-5  # Set the learning rate for all runs

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

SCRIPT_PATH = "pretraining/pre_train_flant5.py"

def update_config(ref_num, learning_rate, proportion):
    """Update the ref_number (balanced or imbalanced), learning_rate, and PROPORTION in the training script"""
    with open(SCRIPT_PATH, 'r') as f:
        content = f.read()
    
    # Replace the appropriate number variable, PROPORTION, and learning_rate lines
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('PROPORTION ='):
            lines[i] = f'PROPORTION = "{proportion}"  # Options: BALANCED, IMBALANCED'
        elif line.strip().startswith('balanced_number ='):
            if proportion == "BALANCED":
                lines[i] = f'    balanced_number = {ref_num}'
            else:
                lines[i] = f'    balanced_number = None'
        elif line.strip().startswith('imbalanced_number ='):
            if proportion == "IMBALANCED":
                lines[i] = f'    imbalanced_number = {ref_num}'
            else:
                lines[i] = f'    imbalanced_number = None'
        elif line.startswith('LEARNING_RATE ='):
            lines[i] = f'LEARNING_RATE = {learning_rate} if USE_LORA else {learning_rate}  # Higher for LoRA'
    
    with open(SCRIPT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    
    var_name = "balanced_number" if proportion == "BALANCED" else "imbalanced_number"
    print(f"✓ Updated PROPORTION to {proportion}")
    print(f"✓ Updated {var_name} to {ref_num}")
    print(f"✓ Updated LEARNING_RATE to {learning_rate}")


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
            print(f"TRAINING RUN {current_run}/{total_runs}: {PROPORTION}_number = {ref_num}, LR = {learning_rate}")
            print(f"{'='*80}\n")
            
            # Update the script
            update_config(ref_num, learning_rate, PROPORTION)
            
            # Run the training script
            result = subprocess.run(
                [sys.executable, SCRIPT_PATH],
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                print(f"\n❌ Training failed for balanced_number={ref_num}, LR={learning_rate}")
                print(f"Stopping sequential training.")
                sys.exit(1)
            
            print(f"\n✓ Completed training for balanced_number={ref_num}, LR={learning_rate}")
    
    print(f"\n{'='*80}")
    print(f"ALL TRAINING RUNS COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


