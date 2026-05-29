import torch
import model
from data import *
from tokenization import Tokenizer
from rewards import ResponseLengthReward, ResponseStructureReward
import evaluate
import numpy as np
from check_device import get_device


def createBatchedData(dataset, batch_size: int):
    '''
        Here, we create batches of input_ids, attention_mask, and labels.
        Batch size = 8, for loading data along with the model on GPU

        1 long value = 8 bytes of memory
        1024 long values = 1 tensor in our case
        3 such tensors at each input instance = 3 x 1024 x 8 = 24,576 bytes

        For a single batch,
        8 instances = 8 x 24,576 = 196,608 bytes

        Memory for a single batch during training = 196.6 KB
    '''
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    batched_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    return batched_data


class Trainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, eps=1e-4)
        self.training_loss = []
        self.device = get_device()
        self.metrics = []
        self.rouge = evaluate.load("rouge")
        self.model.to(device=self.device)
        print("Model, Tokenizer, and Optimizer intialized!")
    
    # def compute_metrics(self, eval_pred):
    #     predictions, labels = eval_pred
    #     decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    #     labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    #     decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

    #     result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    #     prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
    #     result["gen_len"] = np.mean(prediction_lens)
    #     return {k: round(v, 4) for k, v in result.items()}

    def RewardsForResponses(self, ground_truth, outputs, block):
        pass
       
    def SupervisedTraining(self, batched_train_data, epochs, model_dir=''):
        self.model.train()

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            epoch_losses    = []

            for i, batch in enumerate(batched_train_data):
                input_ids       = batch["input_ids"].to(device=self.device)
                attnMask_ids    = batch["attention_mask"].to(device=self.device)
                labels          = batch["labels"].to(device=self.device)

                outputs         = self.model(
                                    input_ids=input_ids,
                                    attention_mask=attnMask_ids,
                                    labels=labels)
                loss = outputs.loss
                if i%10 == 0:
                    print(f"Batch {i} loss: {loss.item():.4f}")

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_losses.append(loss.item())
            print(f"Epoch loss: {sum(epoch_losses)/len(epoch_losses)}")
           
            # Clearing cache
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        print("Model saved.")
        return self.model
    
    def RLTraining(self, batched_train_data, epochs, num_responses, model_dir=''):
        self.model.train()

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            epoch_losses = []

            for i, batch in enumerate(batched_train_data):
                input_ids       = batch["input_ids"].repeat_interleave(num_responses, dim=0).to(device=self.device)
                attnMask_ids    = batch["attention_mask"].repeat_interleave(num_responses, dim=0).to(device=self.device)
                reasoning       = batch["reasoning"]
                answers         = batch["answer"]

                # Generating 'num_responses' from the model for the input.
                # Calculating GRPO-style loss
                with torch.no_grad():
                    # This will output - (num_responses * batch_size) outputs. 
                    # For our case - (4 * 8) = 32 outputs  
                    multiple_outputs    = self.model.generate(input_ids=input_ids,
                                                        attention_mask=attnMask_ids,
                                                        top_p=0.9,
                                                        num_return_sequences=1,
                                                        do_sample=True,
                                                        cache_implementation='offloaded')
                    generated_outputs   = multiple_outputs[:, input_ids.shape(1):]
                    generated_responses = self.tokenizer.batch_decode(generated_responses, skip_special_tokens=True)

                rewards = self.RewardsForResponses(reasoning+answers, generated_responses, num_responses)

                loss = 0.0
                if i%10 == 0:
                    print(f"Batch {i} loss: ", loss)
                    print(f"Reward: {rewards}")

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
    
    def evaluate(self, test_data):
        self.model.eval()
        input_id, attn_mask, reasoning, target = next(iter(test_data))

        labels = torch.cat((reasoning, target), dim=1).clone().detach()
        labels[labels == self.tokenizer.pad_token_type_id] = -100
        mps1 = input_id.to(self.device)
        mps2 = attn_mask.to(self.device)

        return_sequences = 3
        outputs = self.model.generate(input_ids=mps1,
                                    attention_mask=mps2,
                                    temperature=0.7,
                                    top_p=0.9,
                                    num_return_sequences=return_sequences,
                                    do_sample=True,
                                    cache_implementation='offloaded')

        questions = self.tokenizer.batch_decode(input_id, skip_special_tokens=True)
        labels_to_decode = [[self.tokenizer.pad_token_type_id if x == -100 else x for x in label.tolist()] for label in labels]
        answers = self.tokenizer.batch_decode(labels_to_decode, skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True)

        i=0
        for (q, a) in zip(questions, answers):
            print("Question: ", q)
            print("Actual answer: ", a)
            print("Predicted outputs: ", outputs[i:i+return_sequences])
            i += return_sequences
        if self.device == "mps":
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()


#---------------------------------------------------------------------------------#
if __name__ == "__main__":
    #  Load the base model and the LoRA config
    baseModel = model.getBaseModel()
    loraModel = model.getLoRAmodel(baseModel, 
                                    r=6, 
                                    targetModules=["q_proj", "k_proj", "v_proj"])
    
    # Instantiating tokenizer.
    tokenizer = Tokenizer()
    tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(loraModel, tokenizer)

    # Freezing the first argument off the function, only changing the value of 2nd argument.
    embed_fn = partial(embed_SFT_data, tokenizer)

    # Loading data and tokenizing it.
    train_ds, test_ds = load_data()
    train = train_ds.map(embed_fn, batched=True, remove_columns=train_ds.column_names)
    
    batched_train = createBatchedData(train, batch_size=8)
    print(" Batches for training data created.")
    
    trainer.SupervisedTraining(batched_train, epochs=1, model_dir="./T5Model_ckpt1.pt")