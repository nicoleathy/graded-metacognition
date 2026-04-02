import argparse
import pickle

from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("model1", type=str)
parser.add_argument("model2", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--output-path", "-o", type=str, default="weight_change.pkl")
args = parser.parse_args()


def main(args):
    print(f"[+] Loading model1: {args.model1}")
    model1 = AutoModelForCausalLM.from_pretrained(args.model1)
    print(f"[+] Loading model2: {args.model2}")
    model2 = AutoModelForCausalLM.from_pretrained(args.model2)

    print("[+] Extracting weight change...")
    delta = {}
    for name, param in model1.named_parameters():
        delta[name] = param.data - model2.get_parameter(name).data

    with open(args.output_path, "wb") as f:
        pickle.dump(delta, f)
    print(f"[+] Saved weight change to {args.output_path}")


if __name__ == "__main__":
    main(args)
