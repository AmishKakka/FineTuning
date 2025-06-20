import torch
import model
from data import tokenization, fetch, restructure
from datasets import Dataset, NamedSplit, DatasetDict
from rewards import ResponseLengthReward, ResponseStructureReward


def checkDevice():
    device = None
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Yes! MPS is available.")
        print(torch.mps.device_count())
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Yes! CUDA is available.")
        print(torch.cuda.device_count())
    return device

def createBatchedData(dataset: DatasetDict, type: str, batch_size: int):
    '''
        Here, we create batches of input_ids, attention_mask, and labels.
        Batch size = 8, for loading data along with the model on GPU

        1 long value = 8 bytes of memory
        512 long values = 1 tensor in our case
        3 such tensors at each input instance = 3 x 512 x 8 = 12,288 bytes

        For a single batch,
        8 instances = 8 x 12,288 = 98,304 bytes

        Memory for a single batch during training = 98.3 KB

        1 bfloat value = 2 bytes of memory
        512 bfloat values = 1 tensor in our case
        3 such tensors at each input instance = 3 x 512 x 2 = 3072 bytes

        For a single batch,
        8 instances = 8 x 3072 = 24,576 bytes

        Memory for a single batch during training = 24.5 KB
    '''
    DataLoader = torch.utils.data.DataLoader
    data = torch.utils.data.TensorDataset(torch.tensor(dataset[type]['input_ids']),
                                        torch.tensor(dataset[type]['attention_mask']),
                                        torch.tensor(dataset[type]['reasoning']),
                                        torch.tensor(dataset[type]['labels']))
    batched_data = DataLoader(dataset=data,
                            batch_size=batch_size)
    return batched_data


class T5Trainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-4)
        self.training_loss = []
        self.device = checkDevice()
        self.metrics = []
        print("Model, Tokenizer, and Optimizer intialized!")
        
    def RewardsForResponses(self, outputs, block):
        outputs = self.tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True)
        structure_rewards = ResponseStructureReward(outputs)
        length_rewards = ResponseLengthReward(outputs)
        total = [x1+x2 for x1, x2 in zip(structure_rewards, length_rewards)]
        print(outputs)
        rewards = [max(total[i:i+block]) for i in range(0, len(total), block)]
        return sum(rewards)/len(rewards)
       
    def SupervisedTraining(self, batched_train_data, epochs, model_dir=''):
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            epoch_losses = []
            train_iterator = iter(batched_train_data)

            for i, (input_id, attn_mask, target) in enumerate(train_iterator):
                labels = target.clone().detach()
                labels[labels == self.tokenizer.pad_token_type_id] = -100

                mps_input_ids = input_id.to(device=self.device)
                mps_attnMask_ids = attn_mask.to(device=self.device)
                mps_labels = labels.to(device=self.device)

                self.model.train()
                outputs = self.model(input_ids=mps_input_ids,
                                    attention_mask=mps_attnMask_ids,
                                    labels=mps_labels)
                loss = outputs.loss
                if i%50 == 0:
                    print(f"Batch {i} loss: ", loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss)
                print(f"Epoch loss: {sum(epoch_losses)/len(epoch_losses)}")
                if self.device == "mps":
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()
        torch.save(self.model.state_dict(), model_dir)
        print("Model saved.")
        return self.model


if __name__ == "__main__":
    # First we will collect data, transform it, and then load it.
    train_df, val, test = fetch.load_data()
    trainData = restructure.toPandas(train_df)
    valData = restructure.toPandas(val)
    testData = restructure.toPandas(test)
    
    train_dataset = Dataset.from_pandas(trainData, split=NamedSplit('train'))
    val_dataset = Dataset.from_pandas(valData, split=NamedSplit('validation'))
    test_dataset = Dataset.from_pandas(testData, split=NamedSplit('test'))

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    Tokenizer = tokenization.T5Tokenizer()
    tokenized_dataset = dataset_dict.map(Tokenizer.tokenizeRows, batched=True)
    
    batched_train = createBatchedData(tokenized_dataset, type="train", batch_size=8)
    print(" Batches for training data created.")
    