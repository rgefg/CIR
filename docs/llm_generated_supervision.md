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

```jsonl
{"id":"000003390","instruction":"Change the man's attire to a business suit.","modified_caption":"A man in a business suit conducts business during an event.","reverse_instruction":"Change the man's attire to non-business suit attire"}
{"id":"000007547","instruction":"Change the rehearsal setting to take place at night.","modified_caption":"Dancers rehearse for a performance at night.","reverse_instruction":"Change the rehearsal setting to day time."}
{"id":"000020829","instruction":"Change the time of day from 'a week' to 'on a sunny afternoon'.","modified_caption":"actor arrives on a sunny afternoon to a screening","reverse_instruction":"Change the time of day to 'a week'."}
{"id":"000015554","instruction":"Change the color of the brick wall from red to blue.","modified_caption":"a blue brick wall from a building","reverse_instruction":"Change the color of the brick wall to red"}
{"id":"000018986","instruction":"Change the color of the patterned piece from blue and white to green and black.","modified_caption":"In the sea, she showcased her flat stomach in a green and black patterned piece as she took to the sea for a paddle.","reverse_instruction":"Change the color of the patterned piece from green and black to blue and white"}
```
