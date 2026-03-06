import os
import datetime
import logging
import chz
import datasets
import tinker
from typing import cast
from transformers import AutoTokenizer
import asyncio

# Import utilities
from tinker_utils.checkpoint import save_checkpoint, get_last_checkpoint
from tinker_utils.cli import check_log_dir
from tinker_utils.data import build_question
from tinker_utils.env import CodeEnv
from tinker_utils.lcb import normalize_tests
from tinker_utils.log import setup_logging
from tinker_utils.renderers import get_renderer, Message


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = os.path.join(
        os.path.expanduser("~/code-rl-logs"),
        datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    )
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    batch_size: int = 128
    group_size: int = 8
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 10   # 0 = disabled
    eval_every: int = 10   # -1 = disabled
    max_tokens: int = 24576
    format_coef: float = 0.1
    reward_timeout: int = 6
    temperature: float = 1.0
    max_steps: int = -1    # -1 = unlimited
    wandb_project: str | None = None
    resume: bool = False


# Sampling helpers
def _sample_sync(
    sampling_client: tinker.SamplingClient,
    prompt: tinker.ModelInput,
    sampling_params: tinker.types.SamplingParams,
) -> tinker.SampledSequence:
    """
    Call sampling_client.sample() synchronously and return the first SampledSequence.

    sample() returns a ConcurrentFuture; .result() blocks the calling thread.
    Always run inside loop.run_in_executor so it never blocks the event loop.
    """
    response: tinker.types.SampleResponse = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
        include_prompt_logprobs=False,
        topk_prompt_logprobs=0,
    ).result()
    return response.sequences[0]


async def gather_samples(
    loop: asyncio.AbstractEventLoop,
    sampling_client: tinker.SamplingClient,
    prompts: list[tinker.ModelInput],
    sampling_params: tinker.types.SamplingParams,
    count_per_prompt: int,
) -> list[list[tinker.SampledSequence]]:
    """
    Concurrently sample `count_per_prompt` completions for each prompt.
    Returns all_samples[i] = list of SampledSequence for prompts[i].
    """
    all_tasks = []
    index_map: list[int] = []

    for i, prompt in enumerate(prompts):
        for _ in range(count_per_prompt):
            task = loop.run_in_executor(
                None, _sample_sync, sampling_client, prompt, sampling_params
            )
            all_tasks.append(task)
            index_map.append(i)

    flat_results = await asyncio.gather(*all_tasks)

    all_samples: list[list] = [[] for _ in prompts]
    for result, prompt_idx in zip(flat_results, index_map):
        all_samples[prompt_idx].append(result)

    return all_samples


# Main
def main(config: Config):
    check_log_dir(config.log_path, "resume" if config.resume else "ask")
    ml_logger = setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
    )

    logger.info("Loading dataset...")
    train_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset(
                    "agentica-org/DeepCoder-Preview-Dataset",
                    name=name,
                    split="train",
                ),
            )
            for name in ("primeintellect", "taco", "lcbv5")
        ]
    )

    test_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset(
                    "agentica-org/DeepCoder-Preview-Dataset",
                    name=name,
                    split="test",
                ),
            )
            for name in ("codeforces", "lcbv5")
        ]
    )

    logger.info(f"Loading tokenizer for {config.model_name}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=token)
    renderer = get_renderer("qwen3_instruct", tokenizer)

    # ------------------------------------------------------------------
    # Initialise Tinker: create a LoRA training run on the remote server.
    # This registers the model and returns a TrainingClient bound to it.
    # ------------------------------------------------------------------
    logger.info("Creating Tinker LoRA training run...")
    service = tinker.ServiceClient(base_url=config.base_url)

    training_client: tinker.TrainingClient = service.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    # Optionally restore from checkpoint
    start_step = 0
    if config.resume:
        last_checkpoint = get_last_checkpoint(config.log_path)
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            training_client.load_state_with_optimizer(
                last_checkpoint["state_path"]
            ).result()
            start_step = last_checkpoint.get("step", 0) + 1

    # ------------------------------------------------------------------
    # Get an initial SamplingClient from the just-created training weights.
    # We refresh this after every checkpoint save.
    # ------------------------------------------------------------------
    logger.info("Exporting initial weights for sampling...")
    sampling_client: tinker.SamplingClient = (
        training_client.save_weights_and_get_sampling_client()
    )

    adam_params = tinker.types.AdamParams(
        learning_rate=config.learning_rate,
        beta1=0.9,
        beta2=0.999,
    )

  
    # Training loop
    logger.info("Starting training loop...")
    step = start_step
    total_steps = (
        len(train_dataset) // config.batch_size
        if config.max_steps == -1
        else config.max_steps
    )

    train_iter = iter(train_dataset)

    while config.max_steps == -1 or step < total_steps:
        try:
            batch_examples = []
            for _ in range(config.batch_size):
                try:
                    batch_examples.append(next(train_iter))
                except StopIteration:
                    if config.max_steps == -1:
                        train_iter = iter(train_dataset)
                        batch_examples.append(next(train_iter))
                    else:
                        break

            if not batch_examples:
                break

            logger.info(
                f"Step {step}/{total_steps}: Processing batch of "
                f"{len(batch_examples)} examples"
            )

            asyncio.run(
                run_training_step(
                    training_client=training_client,
                    sampling_client=sampling_client,
                    batch_examples=batch_examples,
                    renderer=renderer,
                    config=config,
                    ml_logger=ml_logger,
                    step=step,
                    adam_params=adam_params,
                )
            )

            # Save checkpoint and refresh sampling weights
            if config.save_every > 0 and (step + 1) % config.save_every == 0:
                logger.info(f"Saving checkpoint at step {step}")
                save_checkpoint(
                    training_client=training_client,
                    name=f"step_{step}",
                    log_path=config.log_path,
                    loop_state={"step": step},
                    kind="both",
                )
                # Refresh sampler with latest weights
                sampling_client = (
                    training_client.save_weights_and_get_sampling_client()
                )

            # Evaluate
            if config.eval_every > 0 and (step + 1) % config.eval_every == 0:
                logger.info(f"Evaluating at step {step}")
                asyncio.run(
                    run_evaluation(
                        sampling_client=sampling_client,
                        test_dataset=test_dataset,
                        renderer=renderer,
                        config=config,
                        ml_logger=ml_logger,
                        step=step,
                    )
                )

            step += 1

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in training step {step}: {e}", exc_info=True)
            continue

    # Final checkpoint + eval
    logger.info("Saving final checkpoint...")
    save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        loop_state={"step": step},
        kind="both",
    )

    logger.info("Running final evaluation...")
    final_sampling_client = training_client.save_weights_and_get_sampling_client()
    asyncio.run(
        run_evaluation(
            sampling_client=final_sampling_client,
            test_dataset=test_dataset,
            renderer=renderer,
            config=config,
            ml_logger=ml_logger,
            step=step,
        )
    )

    ml_logger.close()
    logger.info("Training complete!")

# Training step
async def run_training_step(
    training_client: tinker.TrainingClient,
    sampling_client: tinker.SamplingClient,
    batch_examples: list,
    renderer,
    config: Config,
    ml_logger,
    step: int,
    adam_params: tinker.types.AdamParams,
) -> None:
    """Execute one GRPO training step over a batch of examples."""

    envs: list[CodeEnv] = []
    prompts: list[tinker.ModelInput] = []

    for example in batch_examples:
        question = build_question(example)
        if not question:
            continue

        tests = normalize_tests(
            example.get("tests") or example.get("test_list") or [],
            example.get("metadata", {}),
        )
        if not tests:
            continue

        env = CodeEnv(
            problem=question,
            tests=tests,
            renderer=renderer,
            format_coef=config.format_coef,
            reward_timeout=config.reward_timeout,
        )
        envs.append(env)

        messages = [Message(role="user", content=question)]
        prompt = renderer.build_generation_prompt(messages, role="assistant")
        prompts.append(prompt)

    if not envs:
        logger.warning("No valid examples in batch, skipping")
        return

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    logger.info(f"Starting gather_samples for {len(prompts)} prompts x {config.group_size} samples...")
    loop = asyncio.get_event_loop()
    all_samples = await gather_samples(
        loop, sampling_client, prompts, sampling_params, config.group_size
    )
    logger.info(f"gather_samples done. Starting env.step for {len(prompts)} groups...")

    all_datums: list[tinker.types.Datum] = []
    metrics: dict = {
        "total_groups": 0,
        "skipped_groups": 0,
        "format_score_sum": 0.0,
        "correct_score_sum": 0.0,
        "reward_sum": 0.0,
        "num_samples": 0,
    }

    for i, (prompt, env, group_samples) in enumerate(zip(prompts, envs, all_samples)):
        logger.info(f"Running env.step for group {i+1}/{len(prompts)}...")
        # env.step takes Action = list[int] (raw completion token ids)
        reward_tasks = [env.step(sample.tokens) for sample in group_samples]
        step_results = await asyncio.gather(*reward_tasks)
        logger.info(f"Group {i+1} done, rewards: {[r.reward for r in step_results]}")

        rewards = [r.reward for r in step_results]
        advantages = compute_advantages(rewards)

        if should_skip(advantages):
            metrics["skipped_groups"] += 1
            continue

        metrics["total_groups"] += 1

        prompt_tokens = prompt.to_ints()

        for sample, advantage, result in zip(group_samples, advantages, step_results):
            datum = make_datum(
                prompt_tokens=prompt_tokens,
                completion_tokens=sample.tokens,
                logprobs=sample.logprobs or [],
                advantage=advantage,
            )
            all_datums.append(datum)

            metrics["format_score_sum"] += result.metrics.get("format", 0.0)
            metrics["correct_score_sum"] += result.metrics.get("correct", 0.0)
            metrics["reward_sum"] += result.reward
            metrics["num_samples"] += 1

    if all_datums:
        train_step(training_client, all_datums, adam_params)

        n = max(metrics["num_samples"], 1)
        avg_metrics = {
            "train/format_score": metrics["format_score_sum"] / n,
            "train/correct_score": metrics["correct_score_sum"] / n,
            "train/avg_reward": metrics["reward_sum"] / n,
            "train/total_groups": metrics["total_groups"],
            "train/skipped_groups": metrics["skipped_groups"],
            "train/num_samples": metrics["num_samples"],
        }
        ml_logger.log_metrics(avg_metrics, step=step)
        logger.info(f"Step {step} metrics: {avg_metrics}")
    else:
        logger.warning(
            f"Step {step}: no valid datums after reward filtering, skipping update"
        )


# Evaluation


async def run_evaluation(
    sampling_client: tinker.SamplingClient,
    test_dataset,
    renderer,
    config: Config,
    ml_logger,
    step: int,
    num_eval_examples: int = 50,
) -> None:
    """Evaluate the model on the test set using greedy decoding."""

    eval_examples = list(
        test_dataset.select(range(min(num_eval_examples, len(test_dataset))))
    )

    envs: list[CodeEnv] = []
    prompts: list[tinker.ModelInput] = []

    for example in eval_examples:
        question = build_question(example)
        if not question:
            continue

        tests = normalize_tests(
            example.get("tests") or example.get("test_list") or [],
            example.get("metadata", {}),
        )
        if not tests:
            continue

        env = CodeEnv(
            problem=question,
            tests=tests,
            renderer=renderer,
            format_coef=config.format_coef,
            reward_timeout=config.reward_timeout,
        )
        envs.append(env)

        messages = [Message(role="user", content=question)]
        prompt = renderer.build_generation_prompt(messages, role="assistant")
        prompts.append(prompt)

    if not envs:
        logger.warning("No valid eval examples")
        return

    # Greedy decoding for eval
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=0.0,
    )

    loop = asyncio.get_event_loop()
    all_samples = await gather_samples(
        loop, sampling_client, prompts, sampling_params, count_per_prompt=1
    )
    samples = [group[0] for group in all_samples]

    reward_tasks = [env.step(sample.tokens) for env, sample in zip(envs, samples)]
    results = await asyncio.gather(*reward_tasks)

    format_scores = [r.metrics.get("format", 0.0) for r in results]
    correct_scores = [r.metrics.get("correct", 0.0) for r in results]
    rewards = [r.reward for r in results]

    n = len(results)
    eval_metrics = {
        "eval/format_score": sum(format_scores) / n,
        "eval/correct_score": sum(correct_scores) / n,
        "eval/avg_reward": sum(rewards) / n,
        "eval/pass_rate": sum(1 for s in correct_scores if s > 0) / n,
    }

    ml_logger.log_metrics(eval_metrics, step=step)
    logger.info(f"Eval metrics at step {step}: {eval_metrics}")


# Helper functions

def should_skip(advantages: list[float]) -> bool:
    """Return True if all advantages are essentially zero (degenerate group)."""
    return all(abs(a) < 1e-9 for a in advantages)


def compute_advantages(rewards: list[float]) -> list[float]:
    """
    Compute GRPO group-relative advantages: Ai = Ri - mean(R).
    Per the pset spec, we use mean-subtraction only (no std normalisation).
    """
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def make_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    logprobs: list[float],
    advantage: float,
) -> tinker.types.Datum:
    """
    Construct a Tinker Datum for importance-sampling (GRPO) training.

    The server requires target_tokens, logprobs, and advantages to all have
    the same length as the full input sequence (prompt + completion).
    Prompt positions use dummy values (0 tokens, 0.0 logprob, 0.0 advantage)
    since the loss is only applied to the completion portion.
    """
    full_tokens = prompt_tokens + completion_tokens
    total_len = len(full_tokens)
    prompt_len = len(prompt_tokens)
    comp_len = len(completion_tokens)

    # Align logprobs to completion length
    if len(logprobs) >= comp_len:
        comp_logprobs = list(logprobs[:comp_len])
    else:
        comp_logprobs = list(logprobs) + [0.0] * (comp_len - len(logprobs))

    # Pad with prompt-length prefix of dummy values
    all_target_tokens = [0] * prompt_len + completion_tokens
    all_logprobs      = [0.0] * prompt_len + comp_logprobs
    all_advantages    = [0.0] * prompt_len + [advantage] * comp_len

    assert len(all_target_tokens) == total_len
    assert len(all_logprobs) == total_len
    assert len(all_advantages) == total_len

    return tinker.types.Datum(
        model_input=tinker.ModelInput.from_ints(full_tokens),
        loss_fn_inputs={
            "target_tokens": tinker.types.TensorData(
                data=all_target_tokens,
                dtype="int64",
                shape=[total_len],
            ),
            "logprobs": tinker.types.TensorData(
                data=all_logprobs,
                dtype="float32",
                shape=[total_len],
            ),
            "advantages": tinker.types.TensorData(
                data=all_advantages,
                dtype="float32",
                shape=[total_len],
            ),
        },
    )


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams,
) -> None:
    """
    One gradient update:
      1. forward_backward() with importance_sampling loss — computes gradients
      2. optim_step()                                     — applies Adam update

    Both futures are submitted before blocking so the server can pipeline them.
    """
    fwd_bwd_future = training_client.forward_backward(
        data=datums,
        loss_fn="importance_sampling",
    )
    optim_future = training_client.optim_step(adam_params=adam_params)

    fwd_bwd_result = fwd_bwd_future.result()
    optim_future.result()

    logger.debug(f"train_step server metrics: {fwd_bwd_result.metrics}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
