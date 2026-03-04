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


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_sync(sampling_client: tinker.SamplingClient, prompt: tinker.ModelInput,
                 sampling_params: tinker.types.SamplingParams):
    """
    Call sampling_client.sample() synchronously and return the first SampleResult.

    sampling_client.sample() returns a concurrent.futures.Future; calling .result()
    blocks the calling thread until the result is ready.  We always run this inside
    loop.run_in_executor so it never blocks the asyncio event loop.
    """
    future = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
        include_prompt_logprobs=False,
        topk_prompt_logprobs=0,
    )
    results = future.result()
    # sample() with num_samples=1 returns a list/sequence with one SampleResult.
    if isinstance(results, (list, tuple)):
        return results[0]
    return results


async def gather_samples(
    loop: asyncio.AbstractEventLoop,
    sampling_client: tinker.SamplingClient,
    prompts: list[tinker.ModelInput],
    sampling_params: tinker.types.SamplingParams,
    count_per_prompt: int,
) -> list[list]:
    """
    Concurrently sample `count_per_prompt` completions for each prompt.

    Returns a list-of-lists: all_samples[i] is a list of `count_per_prompt`
    SampleResult objects for prompts[i].
    """
    all_tasks = []
    index_map: list[int] = []

    for i, prompt in enumerate(prompts):
        for _ in range(count_per_prompt):
            task = loop.run_in_executor(
                None,
                _sample_sync,
                sampling_client,
                prompt,
                sampling_params,
            )
            all_tasks.append(task)
            index_map.append(i)

    flat_results = await asyncio.gather(*all_tasks)

    # Reconstruct per-prompt groupings
    all_samples: list[list] = [[] for _ in prompts]
    for result, prompt_idx in zip(flat_results, index_map):
        all_samples[prompt_idx].append(result)

    return all_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: Config):
    # Setup logging
    check_log_dir(config.log_path, "resume" if config.resume else "ask")
    ml_logger = setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
    )

    # Load dataset
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

    # Initialize tokenizer and renderer
    logger.info(f"Loading tokenizer for {config.model_name}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=token)
    renderer = get_renderer("qwen3_instruct", tokenizer)

    # Initialize Tinker client
    logger.info("Initializing Tinker training client...")
    service = tinker.ServiceClient(base_url=config.base_url)

    # Resolve the internal holder that owns the async loop managed by Tinker.
    # Fall back to `service` itself if none of the known attribute names match.
    holder = service
    for attr_name in ("_client", "client", "_holder", "holder"):
        candidate = getattr(service, attr_name, None)
        if candidate is not None:
            holder = candidate
            logger.debug(f"Using Tinker holder from attribute '{attr_name}'")
            break

    model_id = tinker.ModelID(config.model_name)

    training_client = tinker.TrainingClient(
        holder=holder,
        model_seq_id=0,
        model_id=model_id,
    )

    sampling_client = tinker.SamplingClient(
        holder=holder,
        sampling_session_id="train",
    )

    # Adam optimizer parameters
    adam_params = tinker.types.AdamParams(
        learning_rate=config.learning_rate,
        beta1=0.9,
        beta2=0.999,
    )

    # Check for checkpoint to resume from
    start_step = 0
    if config.resume:
        last_checkpoint = get_last_checkpoint(config.log_path)
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            training_client.load_state(last_checkpoint["state_path"]).result()
            start_step = last_checkpoint.get("step", 0) + 1

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
            # Collect batch
            batch_examples = []
            for _ in range(config.batch_size):
                try:
                    batch_examples.append(next(train_iter))
                except StopIteration:
                    if config.max_steps == -1:
                        # Restart iterator for unlimited training
                        train_iter = iter(train_dataset)
                        batch_examples.append(next(train_iter))
                    else:
                        break

            if not batch_examples:
                break

            logger.info(
                f"Step {step}/{total_steps}: Processing batch of {len(batch_examples)} examples"
            )

            asyncio.run(
                run_training_step(
                    training_client=training_client,
                    sampling_client=sampling_client,
                    batch_examples=batch_examples,
                    renderer=renderer,
                    tokenizer=tokenizer,
                    adam_params=adam_params,
                    config=config,
                    ml_logger=ml_logger,
                    step=step,
                )
            )

            # Save checkpoint
            if config.save_every > 0 and (step + 1) % config.save_every == 0:
                logger.info(f"Saving checkpoint at step {step}")
                save_checkpoint(
                    training_client=training_client,
                    name=f"step_{step}",
                    log_path=config.log_path,
                    loop_state={"step": step},
                    kind="both",
                )

            # Evaluate
            if config.eval_every > 0 and (step + 1) % config.eval_every == 0:
                logger.info(f"Evaluating at step {step}")
                asyncio.run(
                    run_evaluation(
                        training_client=training_client,
                        sampling_client=sampling_client,
                        test_dataset=test_dataset,
                        renderer=renderer,
                        tokenizer=tokenizer,
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

    # Final checkpoint
    logger.info("Saving final checkpoint...")
    save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        loop_state={"step": step},
        kind="both",
    )

    # Final evaluation
    logger.info("Running final evaluation...")
    asyncio.run(
        run_evaluation(
            training_client=training_client,
            sampling_client=sampling_client,
            test_dataset=test_dataset,
            renderer=renderer,
            tokenizer=tokenizer,
            config=config,
            ml_logger=ml_logger,
            step=step,
        )
    )

    ml_logger.close()
    logger.info("Training complete!")


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

async def run_training_step(
    training_client: tinker.TrainingClient,
    sampling_client: tinker.SamplingClient,
    batch_examples: list,
    renderer,
    tokenizer,
    adam_params: tinker.types.AdamParams,
    config: Config,
    ml_logger,
    step: int,
) -> None:
    """Execute one training step with a batch of examples."""

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

    # Sample group_size completions per prompt, all concurrently.
    loop = asyncio.get_event_loop()
    all_samples = await gather_samples(
        loop, sampling_client, prompts, sampling_params, config.group_size
    )

    # Process rewards and build training datums
    all_datums: list[tinker.types.Datum] = []
    metrics = {
        "total_groups": 0,
        "skipped_groups": 0,
        "format_score_sum": 0.0,
        "correct_score_sum": 0.0,
        "reward_sum": 0.0,
        "num_samples": 0,
    }

    for prompt, env, group_samples in zip(prompts, envs, all_samples):
        # env.step expects Action = list[int] (the raw token ids from the sampler)
        reward_tasks = [env.step(sample.tokens) for sample in group_samples]
        step_results = await asyncio.gather(*reward_tasks)

        rewards = [r.reward for r in step_results]
        advantages = compute_advantages(rewards)

        if should_skip(advantages):
            metrics["skipped_groups"] += 1
            continue

        metrics["total_groups"] += 1

        # Compute prompt length once per group (all samples share the same prompt)
        prompt_token_count = sum(chunk.length for chunk in prompt.chunks)

        for sample, advantage, result in zip(group_samples, advantages, step_results):
            datum = make_datum(
                tokens=sample.tokens,
                logprobs=sample.logprobs,
                ob_len=prompt_token_count,
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def run_evaluation(
    training_client: tinker.TrainingClient,
    sampling_client: tinker.SamplingClient,
    test_dataset,
    renderer,
    tokenizer,
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

    # One sample per prompt
    loop = asyncio.get_event_loop()
    all_samples = await gather_samples(
        loop, sampling_client, prompts, sampling_params, count_per_prompt=1
    )
    # Flatten: each inner list has exactly one element
    samples = [group[0] for group in all_samples]

    # env.step expects Action = list[int] tokens
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def should_skip(advantages: list[float]) -> bool:
    """Return True if all advantages are essentially zero (degenerate group)."""
    return all(abs(a) < 1e-9 for a in advantages)


def compute_advantages(rewards: list[float]) -> list[float]:
    """Compute GRPO-style advantages by subtracting the group mean reward."""
    mean_reward = sum(rewards) / len(rewards)
    return [r - mean_reward for r in rewards]


def make_datum(
    tokens: list[int],
    logprobs: list[float],
    ob_len: int,
    advantage: float,
) -> tinker.types.Datum:
    """
    Construct a Tinker training Datum.

    Args:
        tokens:    Sampled completion token IDs.
        logprobs:  Per-token log-probabilities from the behaviour policy.
        ob_len:    Number of tokens in the prompt (used to mask the loss).
        advantage: Scalar advantage (reward − group baseline).

    Returns:
        A tinker.types.Datum ready to be passed to TrainingClient.train().
    """
    return tinker.types.Datum(
        tokens=tokens,
        logprobs=logprobs,
        ob_len=ob_len,
        advantage=advantage,
    )


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams,
) -> None:
    """
    Send datums to Tinker's training service and block until the update completes.
    TrainingClient.train() returns a concurrent.futures.Future.
    """
    training_client.train(datums=datums, adam_params=adam_params).result()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
