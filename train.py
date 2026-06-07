import torch
import model
from data import *
from tokenization import Tokenizer
from unsloth import FastLanguageModel
from rewards import *
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from transformers import GenerationConfig
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
        self.model.to(device=self.device)
        print("Model, Tokenizer, and Optimizer intialized!")

    def load_SFT_model(self, model_dir: str):
        '''
            Load the SFT model we trained.
        '''
        SFT_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_dir,
            dtype = None,
            load_in_4bit = True,
        )
        return SFT_model, tokenizer
       
    def SupervisedTraining(self, batched_train_data, epochs: int, save_to: str):
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

        self.model.save_pretrained(save_to)
        # self.tokenizer.save_pretrained(model_dir)
        print("Model saved.")
        return self.model
    
    def RLTraining(self, train_data, epochs: int, num_responses: int, model_load_dir: str, save_to: str):
        # Load SFT trained model first
        trained_model, tokenizer = self.load_SFT_model(model_load_dir)

        # Defining the configs
        gen_config = GenerationConfig(
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        grpo_config = GRPOConfig(
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            num_generations=num_responses,
            num_train_epochs=epochs,
            output_dir=save_to,
            logging_steps=1,
            generation_kwargs=gen_config.to_dict(),
            max_completion_length=1024,
            max_prompt_length=1024
        )

        # Training model
        grpo_trainer = GRPOTrainer(
            model=trained_model,
            tokenizer=tokenizer, # type: ignore
            reward_funcs=[
                rouge_reward, 
                cot_reward,
                format_reward
            ],
            args=grpo_config,
            train_dataset=train_data
        )

        trainer_output = grpo_trainer.train()

        # Saving the model
        grpo_trainer.save_model(save_to)
        print("Model saved.")
        return trainer_output
    
    def evaluate(self, model, batched_test_data):
        model.eval()

        for i, batch in enumerate(batched_test_data):
            print(batch)
            input_ids     = batch["input_ids"].long().to(device=self.device)
            attnMask_ids  = batch["attention_mask"].long().to(device=self.device)
            batch_gt      = batch["gt_answer"]

            results = []
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attnMask_ids,
                    max_new_tokens=1024,
                    temperature=0.7)
            
            input_len = input_ids.shape[1]
            generated_texts = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            
            for text, gt in zip(generated_texts, batch_gt):
                reasoning, answer = parse_output(text)
                ans_score = rouge.compute(predictions=[answer], references=[gt], use_stemmer=True)
                results.append({"answer_rouge": ans_score["rougeL"]}) # pyright: ignore[reportOptionalSubscript]

            avg_ans = sum(r["answer_rouge"] for r in results) / len(results)
            print(f"\nAverage Answer ROUGE-L: {avg_ans:.4f}")
        if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return avg_ans

#---------------------------------------------------------------------------------#
if __name__ == "__main__":
    # Instantiating tokenizer.
    tokenizer = Tokenizer()

    #  Load the base model and the LoRA config
    baseModel = model.getBaseModel()
    baseModel.resize_token_embeddings(len(tokenizer))
    loraModel = model.getLoRAmodel(baseModel, 
                                    r=12, 
                                    targetModules=["q_proj", "k_proj", "v_proj"])

    trainer = Trainer(loraModel, tokenizer)

    # Freezing the first argument off the function, only changing the value of 2nd argument.
    embed_fn = partial(embed_SFT_data, tokenizer)

    # Loading data, tokenizing it, and creating batches.
    train_ds, test_ds   = load_data()
    train_sft_data      = train_ds.map(embed_fn, batched=True, remove_columns=train_ds.column_names)
    train_rl_data       = train_ds.map(format_for_grpo, batched=True, remove_columns=train_ds.column_names)

    test_embed_fn       = partial(embed_test_data, tokenizer)
    test_mapped         = test_ds.map(test_embed_fn, batched=True, remove_columns=test_ds.column_names)

    
    batched_train = createBatchedData(train_sft_data, batch_size=8)
    print("Batches for training data created...")

    test_mapped.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    batched_test_data   = torch.utils.data.DataLoader(
        test_mapped, # pyright: ignore[reportArgumentType]
        batch_size=4,
        shuffle=True,
        pin_memory=True
    )
    print("Batches for test data created...")
    
    
    # Training the model on just the 'Responses' column
    trainer.SupervisedTraining(batched_train, epochs=1, save_to="./SFT_model_ckpt.pt")

    # Training the dat on 'CoT' along with 'Responses'
    RL_training_output = trainer.RLTraining(train_rl_data, epochs=1, num_responses=4, model_load_dir="./SFT_model_ckpt.pt", save_to="./RL_model_ckpt.pt")