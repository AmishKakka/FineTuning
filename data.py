import datasets

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
    ds = datasets.load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
    
    # Splitting the dataset into train and test
    ds = ds['train'].train_test_split(train_size=0.85)
    train_ds, test_ds = ds['train'], ds['test']
    print("Data fetched successfully!")
    return train_ds, test_ds