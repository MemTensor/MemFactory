import re
def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125

    return reward

def correctness_reward(prompts, responses, answers):
    
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0]}", f"\n参考答案:\n{answers[0]}")
    for index, item in enumerate(responses):
        print(f"模型答案- {index}\n", item)
        
    rewards = [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answers)]
    print("正确性奖励： ", rewards)
    return rewards

def digit_reward(prompts, responses, answers):
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]

def hard_format_reward(prompts, responses, answers):
    pattern = r'<think>\s*([\s\S]+?)\s*</think>\s*<answer>\s*([\s\S]+?)\s*</answer>'
    matches = [re.match(pattern, response) for response in responses]
    rewards = [0.5 if match else 0.0 for match in matches]
    print("格式奖励： ", rewards)
    return rewards

def mark_reward(prompts, responses, answers):
    return [mark_num(response) for response in responses]