import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langfuse import Langfuse, get_client
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# import metrics
from ragas.metrics import (
    AnswerAccuracy,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    NoiseSensitivity,
)
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM
from ragas.run_config import RunConfig

load_dotenv()

embedding_model = os.getenv("EMBEDDING_MODEL")
llm_model = os.getenv("CEREBRAS_LLM_MODEL")

Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

# Metrics
metrics = [
    Faithfulness(),
]

# Get the configured client instance
langfuse = get_client()


# util function to init Ragas Metrics
def init_ragas_metrics(metrics, llm, embedding):
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)


# Utility function to convert Langfuse dataset to RAGAS format
def convert_langfuse_to_ragas_dataset(langfuse_dataset):
    """
    Convert Langfuse dataset to list of evaluation items

    Args:
        langfuse_dataset: Langfuse dataset object (llm_evaluation_dataset format)

    Returns:
        list of dicts with question, answer, contexts
    """
    evaluation_items = []

    for item in langfuse_dataset.items:
        try:
            # Extract question from input
            if "args" in item.input and len(item.input["args"]) > 0:
                question = item.input["args"][0]
            else:
                continue

            # Extract answer and contexts from expected output
            if item.expected_output:
                answer = item.expected_output.get("answer", "")
                sources = item.expected_output.get("sources", [])

                # Convert sources to contexts (list of strings)
                contexts = [
                    source["content"] for source in sources if "content" in source
                ]

                if answer and contexts:
                    evaluation_items.append(
                        {
                            "question": question,
                            "answer": answer,
                            "contexts": contexts,
                        }
                    )

        except Exception as e:
            print(f"Error processing item: {e}")
            continue

    return evaluation_items


# Score a single item with RAGAS
async def score_with_ragas(query, chunks, answer):
    """
    Calculate RAGAS scores for a single question-context-answer tuple

    Args:
        query: The question
        chunks: List of context strings
        answer: The generated answer

    Returns:
        dict: Scores for each metric
    """
    scores = {}
    for m in metrics:
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=chunks,
            response=answer,
        )
        print(f"  Calculating {m.name}...", end=" ")
        try:
            scores[m.name] = await m.single_turn_ascore(sample)
            print(f"✓ {scores[m.name]:.4f}")
        except Exception as e:
            print(f"✗ Error: {e}")
            scores[m.name] = None
    return scores


# Evaluate all items and push scores to Langfuse
async def evaluate_and_push_scores(evaluation_items):
    """
    Evaluate all items and push scores to Langfuse as traces

    Args:
        evaluation_items: List of dicts with question, answer, contexts

    Returns:
        list of all scores
    """
    all_scores = []

    print(f"\n{'=' * 60}")
    print(f"Evaluating {len(evaluation_items)} items with RAGAS")
    print(f"{'=' * 60}\n")

    for idx, item in enumerate(evaluation_items, 1):
        question = item["question"]
        contexts = item["contexts"]
        answer = item["answer"]

        print(f"[{idx}/{len(evaluation_items)}] Evaluating: {question[:60]}...")

        # Start a new trace for this evaluation
        with langfuse.start_as_current_observation(as_type="span", name="rag") as trace:
            # Store trace_id for later use
            trace_id = trace.trace_id

            # Add retrieval span
            with trace.start_as_current_observation(
                as_type="span",
                name="retrieval",
                input={"question": question},
                output={"contexts": contexts},
            ):
                pass

            # Add generation span
            with trace.start_as_current_observation(
                as_type="span",
                name="generation",
                input={"question": question, "contexts": contexts},
                output={"answer": answer},
            ):
                pass

            # Compute scores for the question, context, answer tuple
            ragas_scores = await score_with_ragas(question, contexts, answer)

            # Send the scores to Langfuse
            for m in metrics:
                if ragas_scores[m.name] is not None:
                    langfuse.create_score(
                        name=m.name, value=ragas_scores[m.name], trace_id=trace_id
                    )

            all_scores.append(
                {
                    "question": question,
                    "answer": answer,
                    "trace_id": trace_id,
                    **ragas_scores,
                }
            )

            print(f"  ✓ Scores pushed to trace: {trace_id}\n")

    # Ensure all data is sent to Langfuse
    langfuse.flush()

    return all_scores


# Main execution
async def main():
    print("=" * 60)
    print("RAG EVALUATION PIPELINE WITH LANGFUSE TRACES")
    print("=" * 60)

    # Load dataset from Langfuse
    print("\n[1/4] Loading dataset from Langfuse...")
    llm_eval_dataset = langfuse.get_dataset(name="llm_evaluation_dataset")
    print(f"✓ Loaded dataset with {len(llm_eval_dataset.items)} items")

    # Convert to evaluation items
    print("\n[2/4] Converting dataset...")
    evaluation_items = convert_langfuse_to_ragas_dataset(llm_eval_dataset)
    print(f"✓ Prepared {len(evaluation_items)} items for evaluation")

    if len(evaluation_items) == 0:
        print("✗ No valid data found. Exiting.")
        return

    # Setup embedding and LLM
    print("\n[3/4] Initializing embedding and LLM models...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    llm = ChatCerebras(model=llm_model, temperature=0)

    # Initialize metrics
    print("Initializing RAGAS metrics...")
    init_ragas_metrics(
        metrics,
        llm=LangchainLLMWrapper(llm),
        embedding=LangchainEmbeddingsWrapper(embeddings),
    )
    print(f"✓ Initialized {len(metrics)} metrics")

    # Evaluate all items and push scores
    print(f"\n[4/4] Running RAGAS evaluation and pushing to Langfuse...")
    all_scores = await evaluate_and_push_scores(evaluation_items)

    # Display summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"✓ Evaluated {len(all_scores)} items")
    print(f"✓ All scores pushed to Langfuse as traces")

    # Calculate and display average scores
    print("\n" + "=" * 60)
    print("AVERAGE SCORES")
    print("=" * 60)

    for metric in metrics:
        metric_scores = [
            s[metric.name] for s in all_scores if s.get(metric.name) is not None
        ]
        if metric_scores:
            avg_score = sum(metric_scores) / len(metric_scores)
            print(f"{metric.name:40s}: {avg_score:.4f}")

    # Save results to CSV
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    import pandas as pd

    df = pd.DataFrame(all_scores)
    output_file = "ragas_evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Results saved to '{output_file}'")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
