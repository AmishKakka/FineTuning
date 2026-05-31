from unsloth import FastLanguageModel

def Tokenizer():      
    '''
        Defining the Qwen 2.5-1.7B model tokenizer
    '''
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        dtype=None,
        load_in_4bit=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_spcial_tokens": ["<think>", "</think>"]})
    return tokenizer
    # print("Vocab length: ", len(tokenizer.get_vocab()))
    # print(tokenizer("amish kakka"))