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
    save_every: int = 10  # 0 = disabled
    eval_every: int = 10 # -1 = disabled
    max_tokens: int = 24576
    format_coef: float = 0.1
    reward_timeout: int = 6
    temperature: float = 1.0
    max_steps: int = -1  # -1 = unlimited
    wandb_project: str | None = None
    resume: bool = False


def main(config: Config):
    # Setup logging
    check_log_dir(config.log_path, "resume" if config.resume else "ask")
    ml_logger = setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config
    )
    
    # Load dataset
    logger.info("Loading dataset...")
    train_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="train")
            ) for name in ("primeintellect", "taco", "lcbv5")
        ]
    )

    test_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="test")
            ) for name in ("codeforces", "lcbv5")
        ]
    )
    
    # Initialize tokenizer and renderer
    logger.info(f"Loading tokenizer for {config.model_name}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=token)
    renderer = get_renderer("qwen3_instruct", tokenizer)
    
    # Initialize Tinker client
    logger.info("Initializing Tinker training client...")
    service = tinker.ServiceClient(
        base_url=config.base_url,
    )
    # Grab the internal holder used by the generated clients
    holder = getattr(service, "_client", None) or getattr(service, "client", None) or getattr(service, "_holder", None)
    if holder is None:
        # fallback (some SDKs pass the service itself as holder)
        holder = service

    model_id = tinker.ModelID(config.model_name)

    training_client = tinker.TrainingClient(
        holder=holder,
        model_seq_id=0,
        model_id=model_id,
    )
    sampling_session_id = os.path.basename(config.log_path)
    sampling_client = tinker.SamplingClient(
        holder,
        model_id=model_id,
)
    
    # Check for checkpoint to resume from
    start_step = 0
    if config.resume:
        last_checkpoint = get_last_checkpoint(config.log_path)
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            training_client.load_state(last_checkpoint["state_path"]).result()
            start_step = last_checkpoint.get("step", 0) + 1
    
    # Adam optimizer parameters
    adam_params = tinker.types.AdamParams(
        learning_rate=config.learning_rate,
        beta1=0.9,
        beta2=0.999,
    )
    
    # Training loop
    logger.info("Starting training loop...")
    step = start_step
    total_steps = len(train_dataset) // config.batch_size if config.max_steps == -1 else config.max_steps
    
    train_iter = iter(train_dataset)
    
    while step < total_steps or config.max_steps == -1:
        try:
            # Collect batch
            batch_examples = []
            for _ in range(config.batch_size):
                try:
                    example = next(train_iter)
                    batch_examples.append(example)
                except StopIteration:
                    if config.max_steps == -1:
                        train_iter = iter(train_dataset)
                        example = next(train_iter)
                        batch_examples.append(example)
                    else:
                        break
            
            if not batch_examples:
                break
            
            # Process batch
            logger.info(f"Step {step}/{total_steps}: Processing batch of {len(batch_examples)} examples")
            
            # Run training step
            asyncio.run(run_training_step(
                training_client=training_client,
                sampling_client=sampling_client,
                batch_examples=batch_examples,
                renderer=renderer,
                tokenizer=tokenizer,
                adam_params=adam_params,
                config=config,
                ml_logger=ml_logger,
                step=step
            ))
            
            # Save checkpoint
            if config.save_every > 0 and (step + 1) % config.save_every == 0:
                logger.info(f"Saving checkpoint at step {step}")
                save_checkpoint(
                    training_client=training_client,
                    name=f"step_{step}",
                    log_path=config.log_path,
                    loop_state={"step": step},
                    kind="both"
                )
            
            # Evaluate
            if config.eval_every > 0 and (step + 1) % config.eval_every == 0:
                logger.info(f"Evaluating at step {step}")
                asyncio.run(run_evaluation(
                    training_client=training_client,
                    sampling_client=sampling_client,
                    test_dataset=test_dataset,
                    renderer=renderer,
                    tokenizer=tokenizer,
                    config=config,
                    ml_logger=ml_logger,
                    step=step
                ))
            
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
        kind="both"
    )
    
    # Final evaluation
    logger.info("Running final evaluation...")
    asyncio.run(run_evaluation(
        training_client=training_client,
        sampling_client=sampling_client,
        test_dataset=test_dataset,
        renderer=renderer,
        tokenizer=tokenizer,
        config=config,
        ml_logger=ml_logger,
        step=step
    ))
    
    ml_logger.close()
    logger.info("Training complete!")


async def run_training_step(
    training_client: tinker.TrainingClient,
    sampling_client: tinker.SamplingClient,
    batch_examples: list,
    renderer,
    tokenizer,
    adam_params: tinker.types.AdamParams,
    config: Config,
    ml_logger,
    step: int
):
    """Execute one training step with a batch of examples."""
    
    # Prepare environments and prompts
    envs = []
    prompts = []
    
    for example in batch_examples:
        # Build question
        question = build_question(example)
        if not question:
            continue
        
        # Normalize tests
        tests = normalize_tests(
            example.get("tests") or example.get("test_list") or [],
            example.get("metadata", {})
        )
        if not tests:
            continue
        
        # Create environment
        env = CodeEnv(
            problem=question,
            tests=tests,
            renderer=renderer,
            format_coef=config.format_coef,
            reward_timeout=config.reward_timeout
        )
        envs.append(env)
        
        # Build prompt
        messages = [Message(role="user", content=question)]
        prompt = renderer.build_generation_prompt(messages, role="assistant")
        prompts.append(prompt)
    
    if not envs:
        logger.warning("No valid examples in batch, skipping")
        return
    
    # Sample multiple completions per prompt (group_size)
    all_samples = []
    
    for prompt in prompts:
        # Sample group_size completions for this prompt
        sample_tasks = [
            sampling_client.sample(
                observation=prompt,
                stop_condition=renderer.get_stop_sequences(),
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                return_logprobs=True
            )
            for _ in range(config.group_size)
        ]
        
        group_samples = await asyncio.gather(*sample_tasks)
        all_samples.append(group_samples)
    
    # Process rewards
    all_datums = []
    metrics = {
        "total_groups": 0,
        "skipped_groups": 0,
        "format_score_sum": 0.0,
        "correct_score_sum": 0.0,
        "reward_sum": 0.0,
        "num_samples": 0
    }
    
    for env, group_samples in zip(envs, all_samples):
        # Execute code and get rewards
        reward_tasks = [env.step(sample.completion) for sample in group_samples]
        step_results = await asyncio.gather(*reward_tasks)
        
        rewards = [result.reward for result in step_results]
        
        # Compute advantages
        advantages = compute_advantages(rewards)
        
        # Skip degenerate groups
        if should_skip(advantages):
            metrics["skipped_groups"] += 1
            continue
        
        metrics["total_groups"] += 1
        
        # Create training datums
        for sample, advantage, result in zip(group_samples, advantages, step_results):
            datum = make_datum(
                tokens=sample.completion,
                logprobs=sample.logprobs,
                ob_len=len(sample.observation.tokens),
                advantage=advantage
            )
            all_datums.append(datum)
            
            # Update metrics
            metrics["format_score_sum"] += result.metrics.get("format", 0.0)
            metrics["correct_score_sum"] += result.metrics.get("correct", 0.0)
            metrics["reward_sum"] += result.reward
            metrics["num_samples"] += 1
    
    # Train if we have valid datums
    if all_datums:
        train_step(training_client, all_datums, adam_params)
        
        # Log metrics
        avg_metrics = {
            "train/format_score": metrics["format_score_sum"] / max(metrics["num_samples"], 1),
            "train/correct_score": metrics["correct_score_sum"] / max(metrics["num_samples"], 1),
            "train/avg_reward": metrics["reward_sum"] / max(metrics["num_samples"], 1),
            "train/total_groups": metrics["total_groups"],
            "train/skipped_groups": metrics["skipped_groups"],
            "train/num_samples": metrics["num_samples"]
        }
        
        ml_logger.log_metrics(avg_metrics, step=step)
        logger.info(f"Step {step} metrics: {avg_metrics}")


async def run_evaluation(
    training_client: tinker.TrainingClient,
    sampling_client: tinker.SamplingClient,
    test_dataset,
    renderer,
    tokenizer,
    config: Config,
    ml_logger,
    step: int,
    num_eval_examples: int = 50
):
    """Evaluate the model on test set."""
    
    eval_examples = list(test_dataset.select(range(min(num_eval_examples, len(test_dataset)))))
    
    envs = []
    prompts = []
    
    for example in eval_examples:
        question = build_question(example)
        if not question:
            continue
        
        tests = normalize_tests(
            example.get("tests") or example.get("test_list") or [],
            example.get("metadata", {})
        )
        if not tests:
            continue
        
        env = CodeEnv(
            problem=question,
            tests=tests,
            renderer=renderer,
            format_coef=config.format_coef,
            reward_timeout=config.reward_timeout
        )
        envs.append(env)
        
        messages = [Message(role="user", content=question)]
        prompt = renderer.build_generation_prompt(messages, role="assistant")
        prompts.append(prompt)
    
    if not envs:
        logger.warning("No valid eval examples")
        return
    
    # Sample one completion per prompt
    sample_tasks = [
        sampling_client.sample(
            observation=prompt,
            stop_condition=renderer.get_stop_sequences(),
            max_tokens=config.max_tokens,
            temperature=0.0,  # Greedy for eval
            return_logprobs=False
        )
        for prompt in prompts
    ]
    
    samples = await asyncio.gather(*sample_tasks)
    
    # Get rewards
    reward_tasks = [env.step(sample.completion) for env, sample in zip(envs, samples)]
    results = await asyncio.gather(*reward_tasks)
    
    # Compute metrics
    format_scores = [r.metrics.get("format", 0.0) for r in results]
    correct_scores = [r.metrics.get("correct", 0.0) for r in results]
    rewards = [r.reward for r in results]
    
    eval_metrics = {
        "eval/format_score": sum(format_scores) / len(format_scores),
        "eval/correct_score": sum(correct_scores) / len(correct_scores),
        "eval/avg_reward": sum(rewards) / len(rewards),
        "eval/pass_rate": sum(1 for s in correct_scores if s > 0) / len(correct_scores)
    }
    
    ml_logger.log_metrics(eval_metrics, step=step)
    logger.info(f"Eval metrics: {eval_metrics}")



########################################################################
# Helper functions
########################################################################
def should_skip(advantages: list[float]) -> bool:
    # raise NotImplementedError("This function needs to be implemented.")
    return all(abs(a) < 1e-9 for a in advantages)



def compute_advantages(rewards: list[float]) -> list[float]:
    # raise NotImplementedError("This function needs to be implemented.")
    mean_reward = sum(rewards) / len(rewards)
    return [r - mean_reward for r in rewards]



def make_datum(
    tokens: list[int],
    logprobs: list[float],
    ob_len: int,
    advantage: float
) -> tinker.types.Datum:
    # raise NotImplementedError("This function needs to be implemented.")
    """
    Make a training datapoint for Tinker.
    
    A Datum contains:
    - tokens: The sampled completion tokens
    - logprobs: Log probabilities from the old policy
    - ob_len: Length of the observation (prompt), used to mask loss
    - advantage: The advantage value for importance sampling
    
    Args:
        tokens: Sampled completion token IDs
        logprobs: Log probabilities for each token
        ob_len: Length of observation (prompt) in tokens
        advantage: Advantage value (reward - baseline)
        
    Returns:
        Tinker Datum object
    """
    return tinker.types.Datum(
        tokens=tokens,
        logprobs=logprobs,
        ob_len=ob_len,
        advantage=advantage
    )


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams
) -> None:
    # raise NotImplementedError("This function needs to be implemented.")
    """
    Run one training step.
    
    Sends the datums to Tinker's training service, which will:
    1. Compute importance sampling ratios from logprobs
    2. Calculate policy gradient loss with advantages
    3. Apply Adam optimizer update
    
    Args:
        training_client: Tinker training client
        datums: List of training datums
        adam_params: Adam optimizer parameters
    """
    training_client.train(datums=datums, adam_params=adam_params).result()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
