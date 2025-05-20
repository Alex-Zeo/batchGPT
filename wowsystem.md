# WowSystem JSON Schema

OpenAI responses must conform to the JSON schema below. This schema is also loaded at runtime by `postprocessor.py` for validation purposes.

```json
{
  "title": "WowResponse",
  "type": "object",
  "properties": {
    "summary": {
      "title": "Summary",
      "type": "string"
    },
    "key_points": {
      "title": "Key Points",
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  },
  "required": ["summary", "key_points"]
}
```
