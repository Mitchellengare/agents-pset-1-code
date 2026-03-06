# Problem Set 1 — Part II: Written Answers (Problems 13 and 15a)

---

## Problem 13: `complete(P)` Pseudocode

```text
function complete(P):

    prompt = render_chat_template(
        system="You are an expert programmer. Write a Python solution.",
        user=P.description + P.starter_code
    )

    tokens = tokenizer.encode(prompt)

    output = []

    while True:
        next_token = model.sample(tokens + output)
        output.append(next_token)

        if next_token == EOS:
            break

        if decoded(output).endswith("```"):
            break

        if len(output) >= MAX_TOKENS:
            break

    return decode(output)
```

The non-max-token stopping condition here is detecting a **closing code fence (` ``` `)**.
Since the prompt instructs the model to output code in a code block, this signals the completion is finished.

---

## Problem 15: Code Extraction

### (a) Deterministic Extraction Rule

```text
function extract_code(response):

    matches = regex.findall("```python ... ```")

    if none found:
        return ∅

    if one found:
        return that block

    if multiple found:
        return the last one
```

---

i used claude to help format