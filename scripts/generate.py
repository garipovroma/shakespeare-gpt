import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

def generate(model,
             tokenizer,
             prompt,
             out_promts=1,
             prompt_words=64,
             top_p=0.8,
             temperature=0.2,
):
    model.eval()
    generated_texts = []
    with torch.no_grad():
        for _ in tqdm(range(out_promts), desc="Generating text", total=out_promts):
            generated_tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
            generated_text = torch.tensor([]).unsqueeze(0)
            ended = False
            
            for _ in tqdm(range(prompt_words), desc="Generating tokens", total=prompt_words):
                outputs = model(generated_tokens)
                logits = outputs[0][:, -1, :]
                probs = torch.softmax(logits / (temperature if temperature > 0 else 1.0), dim=-1).squeeze()
                
                sorted_prob, sorted_indices = torch.sort(probs, descending=True)
                cumulative_prob = torch.cumsum(sorted_prob, dim=-1)
                top_p_indices = cumulative_prob > top_p
                
                top_p_indices[1:] = top_p_indices[:-1].clone()
                top_p_indices[0] = 0
                top_p_indices = sorted_indices[~top_p_indices]
                probs[top_p_indices] = 0
                probs /= torch.sum(probs)

                next_token_id = torch.multinomial(probs, 1)
                generated_tokens = torch.cat((generated_tokens, next_token_id.unsqueeze(-1)), dim=-1)
                generated_text = torch.cat((generated_text, next_token_id.unsqueeze(-1)), dim=-1)

                if next_token_id[0] is tokenizer.eos_token_id:
                    output_list = list(generated_text.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_texts.append(output_text)
                    ended = True
                    break
            
            if not ended:
                output_list = list(generated_text.squeeze().numpy())
                generated_texts.append(prompt + tokenizer.decode(output_list))
                
    return generated_texts

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--load_mode", type=str, default="huggingface", choices=["huggingface", "torch"], help="where from to load the model and tokenizer")
    argparser.add_argument("--model", type=str, required=True, help="path to the model or the model name on huggingface")
    argparser.add_argument("--tokenizer", type=str, required=True, help="path to the tokenizer or the tokenizer name on huggingface")
    argparser.add_argument("--device", type=str, default="cpu", help="device to run the model on")
    argparser.add_argument("--prompt", type=str, required=True, help="prompt to generate from")
    argparser.add_argument("--out_prompts", type=int, default=1, help="number of prompts to generate")
    argparser.add_argument("--result_length", type=int, default=500, help="approximate length of the generated text")
    argparser.add_argument("--top_p", type=float, default=0.8, help="top p parameter for nucleus sampling")
    argparser.add_argument("--temperature", type=float, default=0.2, help="temperature parameter for softmax")
    argparser.add_argument("--seed", type=int, default=0, help="seed for random number generators")
    args = argparser.parse_args()

    torch.manual_seed(args.seed)

    if args.load_mode == "huggingface":
        model = GPT2LMHeadModel.from_pretrained(args.model)
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    elif args.load_mode == "torch":
        model = torch.load(args.model)
        tokenizer = torch.load(args.tokenizer)
    
    model.to(args.device)

    AVG_WORD_LENGTH = 4

    generated_texts = generate(model, tokenizer, args.prompt, args.out_prompts, args.result_length // AVG_WORD_LENGTH, args.top_p, args.temperature)

    for i, text in enumerate(generated_texts):
        print(f"{i}:\n{text}\n")
        print(f"Number of words: {len(text.split())}")
        print(f"Length of the text: {len(text)}")
        print("\n\n")


