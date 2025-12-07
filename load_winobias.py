# from datasets import load_dataset

# def load_winobias(config="type1_pro"):
#     """
#     Load a specific WinoBias dataset configuration.
#     Options: 'type1_pro', 'type1_anti', 'type2_pro', 'type2_anti'
#     """
#     try:
#         dataset = load_dataset("wino_bias", config)
#         print(f"WinoBias ({config}) loaded successfully!")
#         print(f"Available splits: {dataset.keys()}")
#         return dataset
#     except Exception as e:
#         print(f"Error loading WinoBias ({config}): {e}")
#         return None

# if __name__ == "__main__":
#     for cfg in ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]:
#         print(f"\n{'='*30}\nLoading {cfg}\n{'='*30}")
#         ds = load_winobias(cfg)
#         if ds:
#             # print sample from whichever split exists
#             split = "validation" if "validation" in ds else "test"
#             print(f"\nSample from {split} split:")
#             print(ds[split][0])


import os
from datasets import load_dataset

def load_and_save_winobias(config="type1_pro", save_dir="data/winobias"):
    """
    Load and save a specific WinoBias dataset configuration locally.
    Options: 'type1_pro', 'type1_anti', 'type2_pro', 'type2_anti'
    """
    try:
        dataset = load_dataset("wino_bias", config)
        print(f"WinoBias ({config}) loaded successfully!")
        print(f"Available splits: {dataset.keys()}")

        # Save to local directory
        save_path = f"{save_dir}/{config}"
        os.makedirs(save_path, exist_ok=True)
        dataset.save_to_disk(save_path)
        print(f"✅ Saved to: {save_path}")

        return dataset

    except Exception as e:
        print(f"❌ Error loading WinoBias ({config}): {e}")
        return None


from datasets import load_from_disk
def load_saved_winobias(config="type1_pro", save_dir="data/winobias"):
    """
    Load a locally saved WinoBias dataset configuration.
    """
    try:
        load_path = f"{save_dir}/{config}"
        
        dataset = load_from_disk(load_path)
        print(f"WinoBias ({config}) loaded from disk successfully!")
        print(f"Available splits: {dataset.keys()}")
        return dataset

    except Exception as e:
        print(f"❌ Error loading WinoBias ({config}) from disk: {e}")
        return None
    
# dataset = load_from_disk("data/winobias/type1_pro")
# print(dataset["validation"][0])


if __name__ == "__main__":
    for cfg in ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]:
        print(f"\n{'='*30}\nLoading {cfg}\n{'='*30}")
        # ds = load_and_save_winobias(cfg)
        ds = load_saved_winobias(cfg)
        if ds:
            split = "validation" if "validation" in ds else "test"
            print(f"\nSample from {split} split:")
            print(ds[split][0])
        