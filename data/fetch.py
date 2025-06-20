import polars as pl

'''
If using the notebook in VS Code :-
    Open your terminal and login to huggingface using the command :- huggingface-cli login
    Enter your generated token from Hugging Face.

If using the notebook in Colab :- 
    Create a new key in the Secrets tab.
    And then enter the token from Hugging Face, enabling 'Notebook access'.
'''

def load_data():
    '''
        Download data from Hugging Face Datasets 
    '''
    splits = {'train': 'data/train-00000-of-00001.parquet',
            'validation': 'data/validation-00000-of-00001.parquet',
            'test': 'data/test-00000-of-00001.parquet'}

    df = pl.read_parquet('hf://datasets/allenai/sciq/' + splits['train'])
    val_df = pl.read_parquet('hf://datasets/allenai/sciq/' + splits['validation'])
    test_df = pl.read_parquet('hf://datasets/allenai/sciq/' + splits['test'])
    print("Data fetched successfully!")
    return df, val_df, test_df

def view(frame, rows):
    print(frame.head(n=rows))
