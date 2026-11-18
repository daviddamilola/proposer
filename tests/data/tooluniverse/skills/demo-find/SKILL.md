---
name: demo-find
description: Find where a dataset is published.
---

# Finding data

## When to use
Whenever a dataset is named without a URL.

## Search strategy
Search with `Zenodo_search_records`, then list files with `Zenodo_get_record_files`.
If nothing matches, try `biostudies_search`. Never use `python_code_executor`.

```python
import pandas  # code for a model that can run it
```

### Ranking
Prefer versioned records.

## Reporting
Say what you found.
