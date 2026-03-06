# Problem Set 1 --- Part II: Written Answers (Problems 11--21)

------------------------------------------------------------------------

# Problem 11: SFT Objective and Its Limitations

### (a) SFT Objective

In supervised fine-tuning (SFT), the model is trained to maximize the
probability of a reference solution given a problem.

$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{P \sim D}[\log \pi_\theta(Y_{ref}(P) \mid P)]
$$

If the solution is made of tokens $y_1, ..., y_T$, the loss becomes:

$$
\mathcal{L}_{SFT}(\theta) =
-\mathbb{E}_{P \sim D}
\left[
\sum_{t=1}^{T}
\log \pi_\theta(y_t \mid P, y_{<t})
\right]
$$

This means the model is trained to predict the **next token of the
reference solution** at every step.

------------------------------------------------------------------------

### (b) When Can SFT Reduce Probability of Other Correct Solutions?

SFT only increases the probability of **one reference solution** per
problem.

Because probabilities must sum to 1, increasing the probability of the
reference solution reduces probability elsewhere. If there are other
correct solutions that look different (for example, recursion vs
iteration or different algorithm choices), the model may assign them
lower probability.

This is common in code generation because many different implementations
can be correct.

------------------------------------------------------------------------

# Problem 12: Binary vs Fractional Reward

### (a) What Does Fractional Reward Provide?

Fractional reward $R_{fraction}=k/n$ gives **partial credit** when a
solution passes some tests.

This helps learning early in training. Even if the model fails the full
problem, passing a few tests still provides a signal about what it did
right.

Binary reward gives **0 unless all tests pass**, so the model receives
no learning signal until it fully solves the problem.

------------------------------------------------------------------------

### (b) Why Prefer Binary Reward?

Binary reward matches how programming problems are evaluated in
practice: a solution either passes all tests or it does not.

Fractional rewards may encourage shortcuts, such as solutions that pass
simple cases but fail edge cases. Binary reward avoids this by rewarding
only fully correct solutions.

------------------------------------------------------------------------

# Problem 13: `complete(P)` Pseudocode

``` text
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

The non-max-token stopping condition here is detecting a **closing code
fence (```` ``` ````)**.\
Since the prompt instructs the model to output code in a code block,
this signals the completion is finished.

------------------------------------------------------------------------

# Problem 14: Prompt Renderers

### (a) What Goes Wrong With the Wrong Renderer?

Different models expect different prompt formats. For example:

-   Llama uses `[INST] ... [/INST]`
-   Qwen uses `<|im_start|>`

If the wrong format is used, the model may not understand where the user
message ends or where it should start responding.

------------------------------------------------------------------------

### (b) Possible Symptom

The model might:

-   repeat the prompt in its output\
-   print template tokens like `[INST]`\
-   generate short or irrelevant responses

because it does not recognize the prompt structure.

------------------------------------------------------------------------

# Problem 15: Code Extraction

### (a) Deterministic Extraction Rule

``` text
function extract_code(response):

    matches = regex.findall("```python ... ```")

    if none found:
        return ∅

    if one found:
        return that block

    if multiple found:
        return the last one
```

Returning the last code block is useful because models sometimes
generate a first attempt and then correct themselves.

------------------------------------------------------------------------

### (b) Why Extraction Is Part of the Environment

Extraction directly affects reward. If correct code cannot be extracted,
the model receives zero reward.

Making extraction part of the environment ensures the rule is
deterministic and consistent, so the reward signal is reliable.

------------------------------------------------------------------------

# Problem 16: Why `exec()` on the Host Is Unsafe

### Accidental Failure

A generated program might contain an infinite loop or heavy computation
that freezes the training process.

Since thousands of programs are executed during training, even a few
problematic ones could stall the pipeline.

------------------------------------------------------------------------

### Adversarial Failure

A completion could contain commands such as:

``` python
import os
os.system("rm -rf /")
```

Running code directly on the host would allow the model to execute
arbitrary commands. Using Docker or another sandbox prevents this.

------------------------------------------------------------------------

# Problem 17: Shaped Reward

Reward is defined as:

$$
r = r_{format} + r_{correct}
$$

### (a) Why Allow Negative Format Reward?

If formatting is wrong, a small negative reward (e.g., −0.1) discourages
malformed outputs even when the solution itself is incorrect.

Without this penalty, the model would not learn about formatting until
it also solved the problem correctly.

------------------------------------------------------------------------

### (b) Argument Against Format Reward

Formatting is not the true objective --- passing tests is.

Adding a formatting reward introduces a hyperparameter and may encourage
the model to produce nicely formatted but incorrect code.

An alternative is to simply give zero reward when extraction fails.

------------------------------------------------------------------------

# Problem 18: Policy Gradient

We want to show:

$$
\nabla_\theta \mathbb{E}_{y \sim \pi_\theta(\cdot | P)}[R(y)]
=
\mathbb{E}
\left[
R(y)
\sum_{t=1}^{T}
\nabla_\theta
\log
\pi_\theta(y_t | P, y_{<t})
\right]
$$

Starting from the expectation:

$$
\nabla_\theta \mathbb{E}[R(y)]
=
\nabla_\theta
\sum_y
\pi_\theta(y|P)R(y)
$$

$$
=
\sum_y
R(y)
\nabla_\theta \pi_\theta(y|P)
$$

Using the log-derivative trick:

$$
\nabla_\theta \pi_\theta(y|P)
=
\pi_\theta(y|P)
\nabla_\theta \log \pi_\theta(y|P)
$$

Substitute:

$$
=
\sum_y
R(y)
\pi_\theta(y|P)
\nabla_\theta
\log \pi_\theta(y|P)
$$

$$
=
\mathbb{E}
\left[
R(y)
\nabla_\theta
\log \pi_\theta(y|P)
\right]
$$

Since generation is autoregressive:

$$
\log \pi_\theta(y|P)
=
\sum_{t=1}^{T}
\log \pi_\theta(y_t | P, y_{<t})
$$

Therefore:

$$
\nabla_\theta \log \pi_\theta(y|P)
=
\sum_{t=1}^{T}
\nabla_\theta
\log
\pi_\theta(y_t | P, y_{<t})
$$

which gives the final result.

------------------------------------------------------------------------

# Problem 19: Baseline and Advantage

### (a) Why Subtracting a Baseline Does Not Change the Expected Gradient

If we replace $R(y)$ with $A(y)=R(y)-b(P)$,

the extra term becomes:

$$
b(P)E[\nabla_\theta \log \pi_\theta(y|P)]
$$

but

$$
E[\nabla_\theta \log \pi_\theta(y|P)] = 0
$$

so the expected gradient remains unchanged.

------------------------------------------------------------------------

### (b) Why This Helps

Subtracting a baseline reduces **variance**.

Instead of reinforcing all sampled solutions equally, the model only
reinforces solutions that perform **better than average**, which leads
to more stable updates.

------------------------------------------------------------------------

# Problem 20: GRPO Advantage

### (a) Show That Advantages Sum to Zero

$$
A_g = R_g - \bar{R}
$$

Summing:

$$
\sum_g A_g =
\sum_g R_g - G\bar{R}
$$

Since

$$
\bar{R}=\frac{1}{G}\sum_g R_g
$$

the terms cancel, giving

$$
\sum_g A_g = 0
$$

------------------------------------------------------------------------

### (b) Interpretation

The update depends only on **relative performance within the group**.

Solutions better than the group average are reinforced, and worse ones
are suppressed.

If all rewards are equal, the update is zero.

------------------------------------------------------------------------

# Problem 21: Degenerate Groups

### (a) When All Rewards Are Equal

If all rewards are the same value $c$, then

$$
\bar{R}=c
$$

so

$$
A_g = 0
$$

for every sample.

------------------------------------------------------------------------

### (b) Implication

With zero advantages, the gradient update is zero and no learning
occurs.

This is expected because the model receives no information about which
solutions are better.

In practice these groups should be skipped, which is why the
implementation checks for this case.


i used claude to help format