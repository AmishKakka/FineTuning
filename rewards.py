import re
import evaluate 
rouge = evaluate.load("rouge")


def parse_output(completion):
    pattern = r"<think>(.*?)</think>(.*)"

    text = completion[0]["content"] if isinstance(completion, list) else completion
    match = re.search(pattern, text, re.DOTALL)

    if match:
        reasoning = match.group(1).strip()
        answer    = match.group(2).strip()
    else:
        reasoning = ""
        answer    = text.strip()
    return reasoning, answer


def rouge_reward(completions, **kwargs):
    answers = kwargs.get("answer", [])
    scores  = []
    for completion, gt in zip(completions, answers):
        _, final_answer = parse_output(completion)
        score = rouge.compute(
            predictions=[final_answer],
            references=[gt],
            use_stemmer=True
        )["rougeL"] # type: ignore
        scores.append(score)
    return scores


def cot_reward(completions, **kwargs):
    cots   = kwargs.get("cot", [])
    scores = []
    for completion, gt_cot in zip(completions, cots):
        text = completion[0]["content"] if isinstance(completion, list) else completion

        has_open  = "<think>" in text
        has_close = "</think>" in text
        print(f"  <think>:{has_open}  </think>:{has_close}  len:{len(text.split())}")

        reasoning, _ = parse_output(completion)
        if not reasoning:
            scores.append(0.0)
            continue
        score = rouge.compute(
            predictions=[reasoning],
            references=[gt_cot],
            use_stemmer=True
        )["rougeL"] # type: ignore
        scores.append(score)
    return scores


def format_reward(completions, **kwargs):
    scores = []
    for completion in completions:
        text  = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0
        if "<think>" in text:   
          score += 0.3
        if "</think>" in text:  
          score += 0.3
        
        _, answer = parse_output(completion)
        if len(answer.split()) > 5: 
          score += 0.4
        scores.append(score)
    return scores