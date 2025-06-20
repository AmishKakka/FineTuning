import re

def ResponseStructureReward(completions):
  rewards = []
  for completion in completions:
    match1 = re.search(r"^<think>.*?</think>.*?", completion)
    match2 = re.search(r"^<answer>.*?</answer>.*?", completion)
    if match1 and match2:
        rewards.append(1.0)
    elif match1 or match2:
        rewards.append(0.5)
    else:
        rewards.append(0.25)
  return rewards
# ResponseStructureReward(["The energy is sun. Correct answer: sun"])
# ResponseStructureReward(["<think>The nu#$%^&*([]/[;'/mber of molecular orbi875676rbtals </> produced is the same as the number. <answer>energy </answer>"])


def ResponseLengthReward(completions):
  rewards = []
  for completion in completions:
    words = completion.split(" ")
    if len(words) > 100:
      if len(words) > 200 and len(words) < 400:
        rewards.append(1.0)
      elif len(words) > 400:
        rewards.append(0.5)
      else:
        rewards.append(0.25)
    else:
      rewards.append(0.05)
  return rewards
