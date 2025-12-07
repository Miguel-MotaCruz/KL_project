import pandas as pd


def join_val_test_overall_confusion(val_df_path, test_df_path, output_path):
    """
    Join validation and test results CSVs into a single CSV.
    """
    try:
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)
        
        '''
        Pronoun,Correct,Total,Accuracy
        Male, ... ,... ,...
        Female, ... ,... ,...
        '''

        combined_df = pd.DataFrame(columns=["Pronoun", "Correct", "Total", "Accuracy"])
        for pronoun in ["Male", "Female"]:
            val_row = val_df[val_df["Pronoun"] == pronoun].iloc[0]
            test_row = test_df[test_df["Pronoun"] == pronoun].iloc[0]
            
            correct = val_row["Correct"] + test_row["Correct"]
            total = val_row["Total"] + test_row["Total"]
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            combined_df = pd.concat([combined_df, pd.DataFrame({
                "Pronoun": [pronoun],
                "Correct": [correct],
                "Total": [total],
                "Accuracy": [f"{accuracy:.2f}%"]
            })], ignore_index=True)

        combined_df.to_csv(output_path, index=False)
                
        print(f"Joined validation and test results saved to {output_path}")
    except Exception as e:
        print(f"Error joining validation and test results: {e}")


def join_val_test_per_occupation_confusion(val_df_path, test_df_path, output_path):
    """
    Join validation and test per-occupation results CSVs into a single CSV.
    """
    '''occupations ='''
    occupations = ["driver", "supervisor", "janitor", "cook", "mover", "laborer", "construction worker",
        "chief", "developer", "carpenter", "manager", "lawyer", "farmer", "salesperson",
        "physician", "guard", "analyst", "mechanic", "sheriff", "ceo",
        "attendant", "cashier", "teacher", "nurse", "assistant", "secretary",
        "auditor", "cleaner", "receptionist", "clerk", "counselor", "designer",
        "hairdresser", "writer", "housekeeper", "baker", "accountant", "editor",
        "librarian", "tailor"]
    null_row = pd.DataFrame({
                "Occupation": ["occupation"],
                "Male_Correct": [0],
                "Male_Total": [0],
                "Female_Correct": [0],
                "Female_Total": [0],
            })
    try:
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)
        
        '''
        Occupation,Male_Correct,Male_Total,Female_Correct,Female_Total,Male_Accuracy,Female_Accuracy
        ..., ... 
        '''
        combined_df = pd.DataFrame(columns=["Occupation", "Male_Correct", "Male_Total", "Female_Correct", "Female_Total", "Male_Accuracy", "Female_Accuracy"])
        for occ in occupations:
            val_row = val_df[val_df["Occupation"] == occ].iloc[0] if occ in val_df["Occupation"].values else null_row
            test_row = test_df[test_df["Occupation"] == occ].iloc[0] if occ in test_df["Occupation"].values else null_row

            male_correct = val_row["Male_Correct"] + test_row["Male_Correct"]
            male_total = val_row["Male_Total"] + test_row["Male_Total"]
            female_correct = val_row["Female_Correct"] + test_row["Female_Correct"]
            female_total = val_row["Female_Total"] + test_row["Female_Total"]
            male_accuracy = (male_correct/male_total)*100 if male_total > 0 else 0
            female_accuracy = (female_correct/female_total)*100 if female_total > 0 else 0

            combined_df = pd.concat([combined_df, pd.DataFrame({
                "Occupation": [occ],
                "Male_Correct": [male_correct],
                "Male_Total": [male_total],
                "Female_Correct": [female_correct],
                "Female_Total": [female_total],
                "Male_Accuracy": [f"{male_accuracy:.2f}%"],
                "Female_Accuracy": [f"{female_accuracy:.2f}%"]
            })], ignore_index=True)

        combined_df.to_csv(output_path, index=False)
        print(f"Joined validation and test per-occupation results saved to {output_path}")
    except Exception as e:
        print(f"Error joining validation and test per-occupation results: {e}")


if __name__ == "__main__":
    dataset_configs = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]
    for cfgs in dataset_configs:
        val_path = f"results/winobias/val_tiiuae_Falcon-H1-1.5B-Deep-Instruct/{cfgs}/overall_confusion.csv"
        test_path = f"results/winobias/test_tiiuae_Falcon-H1-1.5B-Deep-Instruct/{cfgs}/overall_confusion.csv"
        output_path = f"results/winobias/tiiuae_Falcon-H1-1.5B-Deep-Instruct/{cfgs}/overall_confusion.csv"
        join_val_test_overall_confusion(val_path, test_path, output_path)
        val_occ_path = f"results/winobias/val_tiiuae_Falcon-H1-1.5B-Deep-Instruct/{cfgs}/per_occupation_confusion.csv"
        test_occ_path = f"results/winobias/test_tiiuae_Falcon-H1-1.5B-Deep-Instruct/{cfgs}/per_occupation_confusion.csv"
        output_occ_path = f"results/winobias/tiiuae_Falcon-H1-1.5B-Deep-Instruct/{cfgs}/per_occupation_confusion.csv"
        join_val_test_per_occupation_confusion(val_occ_path, test_occ_path, output_occ_path)
