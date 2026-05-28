import datasets
from functools import partial
from tokenization import Tokenizer

''' 
If using the notebook in VS Code :-
    Open your terminal and login to huggingface using the command :- hf auth login
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
    ds = ds['train'].train_test_split(train_size=0.80) # type: ignore
    train_ds, test_ds = ds['train'], ds['test']
    print("Data fetched successfully!")
    return train_ds, test_ds


def embed_SFT_data(tokenizer, batch, max_length=1024):
    q_ids = tokenizer(
        text=batch["Question"],
        max_length=max_length//2,
        truncation=True,
        padding=False,
        add_special_tokens=True
    )
    a_ids = tokenizer(
        text=batch["Response"],
        max_length=max_length//2,
        truncation=True,
        padding=False,
        add_special_tokens=False
    )
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for q, a in zip(q_ids["input_ids"], a_ids["input_ids"]):
        combined_ids = q + a
        label = [-100] * len(q) + list(a)
        padding_length = max_length - len(combined_ids)

        if padding_length > 0:
            combined_ids = combined_ids + [tokenizer.pad_token_id]*padding_length
            label = label + [-100]*padding_length
            attn_mask = [1]*(max_length - padding_length) + [0]*padding_length
        else:
            combined_ids = combined_ids[:max_length]
            label = label[:max_length]
            attn_mask = [1]*max_length
        
        input_ids.append(combined_ids)
        labels.append(label)
        attention_mask.append(attn_mask)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


if __name__ == "__main__":
    # Instantiating tokenizer.
    tokenizer = Tokenizer()

    # Freezing the first argument off the function, will only change the value fo 2nd argument.
    embed_fn = partial(embed_SFT_data, tokenizer)

    # Loading data and tokenizing it.
    train_ds, test_ds = load_data()
    train = train_ds.map(embed_fn, batched=True, remove_columns=train_ds.column_names)
    print(train[0])