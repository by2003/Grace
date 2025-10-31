import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import openai
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_jsonl, dump_jsonl, make_needed_dir
import copy
from utils.utils import CodexTokenizer, CodeGenTokenizer, StarCoderTokenizer
import json
import requests
import ollama

device = "cuda"


def build_retrieval_prompt(case, tokenizer, max_num_tokens, max_top_k):
    context_max_tokens = max_num_tokens // 2
    comment = "Given following context: \n"
    context = ";".join(case["retrieved_snippets"])
    before = "and your need to complete following: \n"
    b = case["all_code"]
    context_prompt = comment + context + before+ b + "in one line:"
    return context_prompt

def build_prompt(case, tokenizer, max_num_tokens, max_top_k=10, mode='retrieval'):
    prompt = ""
    prompt = build_retrieval_prompt(case, tokenizer, max_num_tokens, max_top_k)
    return prompt

def parser_args():
    parser = argparse.ArgumentParser(description="Generate response from llm")
    parser.add_argument('--model', default='qwen-2.5-coder', type=str)
    parser.add_argument('--mode', default='retrieval', type=str, choices=['infile', 'retrieval'])
    parser.add_argument('--max_top_k', default=10, type=int)
    parser.add_argument('--max_new_tokens', default=100, type=int)

    return parser.parse_args()


def main(args, input_cases, responses_save_name):
    # set model and tokenizer
    if args.model == 'gpt-3.5-turbo':

        model = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url="https://api.chatanywhere.tech/v1/chat/completions")

        tokenizer = CodexTokenizer()
        max_num_tokens = 4096
    elif args.model == 'starcoder':
        model = AutoModelForCausalLM.from_pretrained(f"./models_cache/{args.model}/")
        tokenizer_raw = AutoTokenizer.from_pretrained(f"./models_cache/{args.model}-tokenizer/", trust_remote_code=True)
        tokenizer = StarCoderTokenizer(tokenizer_raw)
        max_num_tokens = 8192
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer_raw.eos_token_id,
            temperature=0,
            pad_token_id=tokenizer_raw.pad_token_id,
        )
    elif args.model in ['codegen2-16b', 'codegen2-7b', 'codegen2-1b']:
        model = AutoModelForCausalLM.from_pretrained(f"{args.model}/")
        tokenizer_raw = AutoTokenizer.from_pretrained(f"./models_cache/{args.model}-tokenizer/", trust_remote_code=True)
        tokenizer = CodeGenTokenizer(tokenizer_raw)
        max_num_tokens = 2048
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer_raw.eos_token_id,
            temperature=0,
            pad_token_id=tokenizer_raw.pad_token_id,
        )
    print('Model loading finished')

    responses = []
    max_num_tokens = 4096
    tokenizer = CodexTokenizer()
    max_prompt_tokens = max_num_tokens - args.max_new_tokens
    with tqdm(total=len(input_cases)) as pbar:
        for case in input_cases:
            pbar.set_description(f'Processing...')
            prompt = build_prompt(case, tokenizer, max_prompt_tokens, max_top_k=args.max_top_k,mode=args.mode)  

            if args.model == 'gpt-3.5-turbo':
                url = "https://api.chatanywhere.tech/v1/chat/completions"
                OPENAI_KEY = os.getenv("OPENAI_API_KEY")
                payload = json.dumps({
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
                headers = {
                    'Authorization': f'Bearer {OPENAI_KEY}',
                    'Content-Type': 'application/json'
                }
                try:
                    response = requests.request("POST", url, headers=headers, data=payload)
                    response_dict = json.loads(response.text)
                    generated_content = response_dict['choices'][0]['message']['content']
                    response = generated_content
                except Exception as e:
                    path = f"generation_results/gpt-3.5-turbo/cross_file_first.gen_res111.jsonl"
                    make_needed_dir(path)
                    dump_jsonl(responses, path)
                    return
            elif args.model == "starcoder":
                prompt_ids = tokenizer_raw(prompt, return_tensors="pt").to(device)
                response_ids = model.generate(prompt_ids['input_ids'],
                                              generation_config=generation_config,
                                              attention_mask=prompt_ids['attention_mask'])
                response = tokenizer.decode(response_ids[0])
                prompt_lines = prompt.splitlines(keepends=True)
                n_prompt_lines = len(prompt_lines)
                response_lines = response.splitlines(keepends=True)
                response = "".join(response_lines[n_prompt_lines:])
            elif args.model in ['codegen2-16b', 'codegen2-7b', 'codegen2-1b']:
                prompt_ids = tokenizer_raw(prompt, return_tensors="pt").to(device)
                response_ids = model.generate(prompt_ids['input_ids'],
                                              generation_config=generation_config,
                                              attention_mask=prompt_ids['attention_mask'])
                response = tokenizer.decode(response_ids[0])
                prompt_lines = prompt.splitlines(keepends=True)
                n_prompt_lines = len(prompt_lines)
                response_lines = response.splitlines(keepends=True)
                response = "".join(response_lines[n_prompt_lines:])
            elif args.model == 'qwen-2.5-coder':
                try:
                    response = ollama.chat(model='qwen2.5-coder:32b', messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                        },
                    ],
                    options={
                             'num_predict': 100  # 限制最多生成 100 个 token
                             }
                    )
                    response=response['message']['content']
                except Exception as e:
                    path=f"dataset/dataset_repoeval_updated/repoeval_to_repobench/generate_result/cross_file_random_gen_res111.jsonl"
                    make_needed_dir(path)
                    dump_jsonl(responses,path)
                    return
            case_res = copy.deepcopy(case)
            case_res['generate_response'] = response
            responses.append(copy.deepcopy(case_res))
            pbar.update(1)

    dump_jsonl(responses, responses_save_name)


if __name__ == "__main__":
    args = parser_args()

    input_cases = load_jsonl("dataset/dataset_repoeval_updated/repoeval_to_repobench/grace_serach/line_python_search_results.jsonl")
    #input_cases=input_cases[1400:]

    print('Input loading finished')

    responses_save_name = f"dataset/dataset_repoeval_updated/repoeval_to_repobench/generate_result/cross_file_random_gen_res.jsonl"
    make_needed_dir(responses_save_name)
    main(args, input_cases, responses_save_name)


