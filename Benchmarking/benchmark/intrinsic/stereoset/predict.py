import argparse
import json
import os

import transformers

from benchmark.intrinsic.stereoset.runner import StereoSetRunner

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark.")

# Argument: directory to store results and data
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)

# Argument: specify the type of model to evaluate
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertForMaskedLM",
    help="Model to evalute (e.g., BertForMaskedLM, RobertaForMaskedLM). Typically, these correspond to a HuggingFace "
    "class.",
)

# Argument: specify the model's name or path
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)

# Argument: set the batch size
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
)

# Argument: set the random seed
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="RNG seed. Used for logging in experiment ID.",
)

# Argument: path name for results
parser.add_argument(
    "--path_name",
    action="store",
    type=str,
    default="results",
    help="Path name.",
)

parser.add_argument(
    "--cache_dir",
    action="store",
    type=str,
    default=None,
    help="Path to store cached HuggingFace models",
)


if __name__ == "__main__":
    args = parser.parse_args()

    if "rob" in args.model_name_or_path:  # roberta case
        args.model = "RobertaForMaskedLM"

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - seed: {args.seed}")

    model = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )

    # Run the benchmark and collect the results
    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/stereoset/test.json",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=False,
    )
    results = runner()

    # Save the evaluation results to a JSON file
    with open(f"{args.persistent_dir}/stereoset/for_github_test_batch_2.json", "w") as f:
        json.dump(results, f, indent=2)
