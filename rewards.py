import re
import evaluate 
import numpy as np
rouge = evaluate.load("rouge")

def parse_output(generated_text):
	if isinstance(generated_text, list):
			generated_text = generated_text[-1]['content']

	if "<think>" in generated_text and "</think>" in generated_text:
			reasoning = generated_text.split("<think>")[1].split("</think>")[0].strip()
			answer    = generated_text.split("</think>")[1].strip()
	else:
			reasoning = ""
			answer    = generated_text.strip()
	return reasoning, answer

def rouge_reward(prompts, completions, answer, **kwargs):
	scores = []
	for generated, gt in zip(completions, answer):
			reasoning, final_answer = parse_output(generated)

			# Score final answer against ground truth Response
			answer_score = rouge.compute(
					predictions=[final_answer],
					references=[gt],
					use_stemmer=True
			)["rougeL"] # type: ignore

			scores.append(answer_score)
	return scores

def format_reward(prompts, completions, **kwargs):
	scores = []
	for text_item in completions:
			text = text_item[-1]['content'] if isinstance(text_item, list) else text_item
			score = 0.0
			if "<think>" in text:    
				score += 0.3
			if "</think>" in text:
				score += 0.3
			_, answer = parse_output(text)
			if len(answer.split()) > 5:  
				score += 0.4
			scores.append(score)
	return scores

def cot_reward(prompts, completions, cot, **kwargs):
	scores = []
	for generated, gt_cot in zip(completions, cot):
			reasoning, _ = parse_output(generated)

			if not reasoning:
					scores.append(0.0)
					continue

			cot_score = rouge.compute(
					predictions=[reasoning],
					references=[gt_cot],
					use_stemmer=True
			)["rougeL"] # type: ignore

			scores.append(cot_score)
	return scores