import os
import json
import argparse
import jsonlines
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI


def is_explicitly_view_dependent(tokens):
    """
    Check if the given tokens explicitly depend on the view.

    :param tokens: List of tokens
    :return: Boolean indicating if the tokens are view-dependent
    """
    target_words = {
        "front",
        "behind",
        "back",
        "right",
        "left",
        "facing",
        "leftmost",
        "rightmost",
        "looking",
        "across",
    }
    return len(set(tokens).intersection(target_words)) > 0


def decode_stimulus_string(s):
    """
    Decode the stimulus string into its components.

    :param s: The stimulus string
    :return: Tuple containing scene_id, instance_label, n_objects, target_id, distractor_ids
    """
    parts = s.split("-", maxsplit=4)
    if len(parts) == 4:
        scene_id, instance_label, n_objects, target_id = parts
        distractor_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractor_ids = parts

    instance_label = instance_label.replace("_", " ")
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractor_ids = [int(i) for i in distractor_ids.split("-") if i]

    assert len(distractor_ids) == n_objects - 1

    return scene_id, instance_label, n_objects, target_id, distractor_ids


def load_ref_data(anno_file, scan_id_file):
    """
    Load reference data from the annotation file.

    :param anno_file: Path to the annotation file
    :param scan_id_file: Path to the scan ID file
    :return: List of reference data
    """
    split_scan_ids = set(x.strip() for x in open(scan_id_file, "r"))
    ref_data = []
    with jsonlines.open(anno_file, "r") as f:
        for item in f:
            if item["scan_id"] in split_scan_ids:
                ref_data.append(item)
    return ref_data


def process_reference_item(ref, program_prompt, client, args):
    """
    Process each reference item and generate new data for it.

    :param ref: A single reference item
    :param program_prompt: The prompt to be used for the OpenAI model
    :param client: The OpenAI client
    :param args: Parsed command-line arguments
    :return: Processed data for the reference item
    """
    caption = ref["utterance"]
    print("-" * 20)
    print(caption)
    print(ref["scan_id"])

    # Determine context and view dependence
    hardness = decode_stimulus_string(ref["stimulus_id"])[2]
    easy_context_mask = hardness <= 2
    view_dep_mask = is_explicitly_view_dependent(ref["tokens"])

    input_prompt = f"Query: {caption}"

    # Interact with OpenAI API
    messages = [
        {"role": "system", "content": program_prompt},
        {"role": "user", "content": [{"type": "text", "text": input_prompt}]},
    ]
    chat_response = client.chat.completions.create(
        model=args.model_name, messages=messages
    )
    answer = chat_response.choices[0].message.content

    try:
        answer = answer.replace("'", '"')
        answer = json.loads(answer)
        print(answer)
        target = answer["Target"]
        anchor = answer["Anchor"]
        print(f"Target: {target}, Anchor: {anchor}")
    except:
        print(answer)
        target = ""
        anchor = ""
        print("!!! Warning, Error in answer")

    return {
        "scan_id": ref["scan_id"],
        "target_id": ref["target_id"],
        "caption": ref["utterance"],
        "parsed_query": answer,
        "easy": easy_context_mask,
        "view_dep": view_dep_mask,
    }


def save_processed_data(new_data, save_dir, scan_id):
    """
    Save the processed data for a scan to a file.

    :param new_data: The new data to be saved
    :param save_dir: Directory where the file should be saved
    :param scan_id: The scan ID for which the data is being saved
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/{scan_id}.json", "w") as f:
        json.dump(new_data, f, indent=4)
    print(f"Saved scan {scan_id} data to {save_dir}/{scan_id}.json")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process rooms for object detection.")
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="your_openai_api_key",
        help="OpenAI API key.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/root/Qwen2-VL/SeeGround/data/nr3d/query",
        help="Directory to save data.",
    )
    parser.add_argument(
        "--openai_api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI API base URL.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2-VL-72B-Instruct",
        help="Model name for OpenAI API.",
    )
    parser.add_argument(
        "--anno_file",
        type=str,
        default="/remote-home/rongli/data/nr3d.jsonl",
        help="Path to the annotation file.",
    )
    parser.add_argument(
        "--scan_id_file",
        type=str,
        default="/remote-home/rongli/data/scannetv2_val.txt",
        help="Path to the scan ID file.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="SeeGround/prompts/parsing_query.txt",
        help="Path to the prompt file.",
    )

    args = parser.parse_args()

    # Load reference data
    ref_data = load_ref_data(args.anno_file, args.scan_id_file)

    # Load prompt
    with open(args.prompt_file, "r") as f:
        program_prompt = f.read()

    # Group data by scan_id
    grouped_data = defaultdict(list)
    for ref in ref_data:
        grouped_data[ref["scan_id"]].append(ref)

    # Sort scan_ids
    sorted_scan_ids = sorted(grouped_data.keys())

    # Process each scan_id and generate new data
    for scan_id in sorted_scan_ids:
        if os.path.exists(f"{args.save_dir}/{scan_id}.json"):
            print(f"Skipping {scan_id}. Already exists.")
            continue

        entries = grouped_data[scan_id]
        new_data = []

        # Process each reference item
        for ref in tqdm(entries):
            new_data.append(
                process_reference_item(
                    ref,
                    program_prompt,
                    OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_base),
                    args,
                )
            )

        # Save processed data
        save_processed_data(new_data, args.save_dir, scan_id)
