# LLM-Generated Supervision Notes

This note records the generation and filtering details for the CC3M-derived edit supervision used by the DeCIR experiments.

## Model

- Forward edit and target caption generator: `glm-4.7-flash` via the ZhipuAI batch `/v4/chat/completions` API.
- Generation setting in `data/generate_batch.py`: reasoning/thinking enabled, `temperature=0.7`.
- Reverse edit parser accepts `reverse_instruction` and Chinese `反向指令` keys in `data/parse_reverse_batch_output.py`.

## Forward Prompt

System prompt:

```text
You are helping to create a multi-modal dataset for Composed Image Retrieval (CIR).
The dataset requires pairs of source and target image captions, along with a single, concise instruction describing how the source image is transformed into the target image.

Task:
1. Input: A source image caption will be provided.
2. Brainstorming: Identify the key elements in the source caption (objects, actions, setting) and propose one significant, plausible modification.
3. Modification Instruction: Write a clear, succinct directive describing the intended change.
4. Modified Caption: Apply the change to the original caption, preserving all other details.

Important Constraints:
1. The modification instruction must focus on a single, significant change (e.g., changing an object's color, location, or action).
2. The modified caption must only incorporate this one change and remain otherwise consistent with the original caption.
3. The instruction and modified caption should be coherent and plausible.

Output Requirements:
Output exactly three items in valid JSON format with the following keys:
{
    "brainstorming": "Briefly explain the original elements and the proposed change.",
    "instruction": "A short statement of the exact change to be made.",
    "modified_caption": "The new caption reflecting the transformation."
}
```

User prompt template:

```text
Input: {source_caption}
```

## Filtering

Forward responses are parsed by removing Markdown code fences, accepting either a JSON object or a list of JSON objects that can be merged into one object, and keeping rows with both `instruction` and `modified_caption`.

The retrieval-cleaned merged file applies these filters:

- non-empty instruction and modified caption;
- at least 3 instruction words and 3 caption words;
- no `#`, no ellipsis in the modified caption, and no `stock photo` caption;
- instruction/modified-caption Jaccard overlap at most 0.5;
- duplicate `(instruction, modified_caption)` pairs removed.

## Counts

- `data/cc3m_new.jsonl`: 727,573 raw parsed forward generations.
- `data/cc3m_new.cleaned.jsonl`: 667,229 kept after first-stage cleaning, 91.71%.
- `data/cc3m_cir_dataset_cleaned_v1mid_v2__merged_with_cc3m_new.jsonl`: 2,725,236 merged candidate rows.
- `data/cc3m_cir_dataset_cleaned_v1mid_v2__merged_with_cc3m_new.retrieval_clean_v2.jsonl`: 2,598,571 kept, 95.35%.
- `data/cc3m_cir_dataset_cleaned_v1mid_v2_with_reverse.jsonl`: 1,277,145 rows with reverse supervision.

## Examples

The five appendix examples below cover color change, object replacement, action/posture change, background/scene change, and style/material change.

```jsonl
{"category":"color change","id":"000020618","instruction":"Change the color of the umbrella from red to blue.","modified_caption":"An open blue umbrella with a shadow.","reverse_instruction":"Change the color of the umbrella from blue to red"}
{"category":"object replacement","id":"000042412","instruction":"Replace the broom with a shovel for clearing snow.","modified_caption":"A worker clears snow off the sidewalk with a shovel.","reverse_instruction":"Replace the shovel with a broom for clearing snow."}
{"category":"action/posture change","id":"000050032","instruction":"Change the dog's posture from standing to sitting.","modified_caption":"A dog sits on a balcony looking down.","reverse_instruction":"Change the dog's posture from sitting to standing"}
{"category":"background/scene change","id":"000076808","instruction":"Change the river to a lake.","modified_caption":"A young man struggles to row through a lake.","reverse_instruction":"Change the lake to a river"}
{"category":"style/material change","id":"000057407","instruction":"Change the material of the statue from stone to gold.","modified_caption":"A gold statue of a builder at the shrine.","reverse_instruction":"Change the material of the statue to stone"}
```
