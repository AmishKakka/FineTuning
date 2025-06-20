from transformers import AutoTokenizer
import torch

class T5Tokenizer():
    def __init__(self):      
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.tokenizer.add_tokens(['<think>', '</think>', '<answer>', '</answer>'])
        print(self.tokenizer("amish kakka"))

    def tokenizeRows(self, examples):
        model_inputs = self.tokenizer.batch_encode_plus(examples['questions'], padding='max_length', truncation=True, return_tensors='pt')
        reasoning = self.tokenizer.batch_encode_plus(examples['reasoning'], padding='max_length', truncation=True, return_tensors='pt')
        labels = self.tokenizer.batch_encode_plus(examples['answers'], padding='max_length', truncation=True, return_tensors='pt')
        model_inputs['input_ids'] = torch.tensor(model_inputs['input_ids'], dtype=torch.long)
        model_inputs['labels'] = torch.tensor(labels['input_ids'], dtype=torch.long)
        model_inputs['reasoning'] = torch.tensor(reasoning['input_ids'], dtype=torch.long)
        return model_inputs
