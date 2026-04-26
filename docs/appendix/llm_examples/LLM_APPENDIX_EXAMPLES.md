# Cleaner LLM-Generated Supervision Examples

Selected from `data/cc3m_cir_dataset_cleaned_v1mid_v2_with_reverse.jsonl` to replace noisy appendix examples. The `source_evidence` field is a concise paraphrase of the source elements described in the LLM `brainstorming` field, not an additional hand-written training target.

| Category | ID | Source evidence | Instruction | Modified caption | Reverse instruction |
| --- | --- | --- | --- | --- | --- |
| color change | `000020618` | open red umbrella casting a shadow | Change the color of the umbrella from red to blue. | An open blue umbrella with a shadow. | Change the color of the umbrella from blue to red |
| object replacement | `000042412` | worker clearing snow off the sidewalk with a broom | Replace the broom with a shovel for clearing snow. | A worker clears snow off the sidewalk with a shovel. | Replace the shovel with a broom for clearing snow. |
| action/posture change | `000050032` | dog standing on a balcony and looking down | Change the dog's posture from standing to sitting. | A dog sits on a balcony looking down. | Change the dog's posture from sitting to standing |
| background/scene change | `000076808` | young man struggling to row through a river | Change the river to a lake. | A young man struggles to row through a lake. | Change the lake to a river |
| style/material change | `000057407` | statue of a builder at a shrine, originally stone | Change the material of the statue from stone to gold. | A gold statue of a builder at the shrine. | Change the material of the statue to stone |

## Exact JSONL Rows

```jsonl
{"category": "color change", "id": "000020618", "instruction": "Change the color of the umbrella from red to blue.", "modified_caption": "An open blue umbrella with a shadow.", "reverse_instruction": "Change the color of the umbrella from blue to red"}
{"category": "object replacement", "id": "000042412", "instruction": "Replace the broom with a shovel for clearing snow.", "modified_caption": "A worker clears snow off the sidewalk with a shovel.", "reverse_instruction": "Replace the shovel with a broom for clearing snow."}
{"category": "action/posture change", "id": "000050032", "instruction": "Change the dog's posture from standing to sitting.", "modified_caption": "A dog sits on a balcony looking down.", "reverse_instruction": "Change the dog's posture from sitting to standing"}
{"category": "background/scene change", "id": "000076808", "instruction": "Change the river to a lake.", "modified_caption": "A young man struggles to row through a lake.", "reverse_instruction": "Change the lake to a river"}
{"category": "style/material change", "id": "000057407", "instruction": "Change the material of the statue from stone to gold.", "modified_caption": "A gold statue of a builder at the shrine.", "reverse_instruction": "Change the material of the statue to stone"}
```

## Selection Notes

- `000020618` (color change): Single visual attribute edit; caption preserves object and shadow context.
- `000042412` (object replacement): Replaces only the tool while keeping worker, action, and sidewalk setting.
- `000050032` (action/posture change): Changes posture only; object, balcony, and gaze remain aligned.
- `000076808` (background/scene change): Changes the scene element from river to lake without changing subject or action.
- `000057407` (style/material change): Clean material edit from stone to gold with the statue subject preserved.
