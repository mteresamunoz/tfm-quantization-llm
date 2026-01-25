from datasets import Dataset, DatasetDict
import json
import random
import pandas as pd

def load_dataset(directory,format="ds"):

    data = []
    

    with open(directory, 'r') as f:
        for i, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")


    df = pd.DataFrame(data)  

    random.seed(42)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(0.8 * len(df))]
    dev_df = df.iloc[int(0.8 * len(df)):int(0.9 * len(df))]
    test_df = df.iloc[int(0.9 * len(df)):]

    train_ds = Dataset.from_pandas(train_df)
    dev_ds = Dataset.from_pandas(dev_df)
    test_ds = Dataset.from_pandas(test_df)

    dataset_dict = DatasetDict({
        "train": train_ds,
        "dev": dev_ds,
        "test": test_ds
    })

    if format == "ds":
        return dataset_dict
    elif format == "df":
        return train_df, dev_df, test_df
    else:
        print("Invalid format.")

