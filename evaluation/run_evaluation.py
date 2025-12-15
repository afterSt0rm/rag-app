"""
Run Evaluation Script for RAG Pipeline

This script provides multiple ways to evaluate your RAG system:
1. From Langfuse dataset (existing evaluation.py functionality)
2. From custom test dataset (JSON/CSV)
3. Against live queries from a collection
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from evaluation.integrated_evaluation import RAGEvaluator
from langfuse import Langfuse, get_client
from query.query_pipeline import get_query_pipeline

load_dotenv()

# Initialize Langfuse
Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

langfuse = get_client()


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


def load_test_dataset_from_json(filepath: str) -> List[Dict]:
    """
    Load test dataset from JSON file.

    Expected format:
    [
        {
            "question": "What is...",
            "answer": "The answer is...",
            "contexts": ["context1", "context2"],
            "reference_answer": "Ground truth..." (optional)
        },
        ...
    ]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of evaluation items")

    return data


def load_test_dataset_from_csv(filepath: str) -> List[Dict]:
    """
    Load test dataset from CSV file.

    Expected columns:
    - question
    - answer
    - contexts (JSON string or comma-separated)
    - reference_answer (optional)
    """
    df = pd.read_csv(filepath)

    required_columns = ["question", "answer", "contexts"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")

    evaluation_items = []

    for _, row in df.iterrows():
        item = {
            "question": row["question"],
            "answer": row["answer"],
        }

        # Parse contexts (try JSON first, then comma-separated)
        contexts = row["contexts"]
        if isinstance(contexts, str):
            try:
                item["contexts"] = json.loads(contexts)
            except json.JSONDecodeError:
                item["contexts"] = [c.strip() for c in contexts.split(",")]
        else:
            item["contexts"] = [contexts]

        # Add reference answer if available
        if "reference_answer" in df.columns and pd.notna(row["reference_answer"]):
            item["reference_answer"] = row["reference_answer"]

        evaluation_items.append(item)

    return evaluation_items


async def evaluate_from_langfuse_dataset(
    dataset_name: str = "llm_evaluation_dataset",
) -> None:
    """
    Evaluate RAG pipeline using a Langfuse dataset.

    Args:
        dataset_name: Name of the Langfuse dataset
    """
    print("=" * 60)
    print("RAG EVALUATION FROM LANGFUSE DATASET")
    print("=" * 60)

    # Load dataset from Langfuse
    print(f"\n[1/3] Loading dataset '{dataset_name}' from Langfuse...")
    try:
        llm_eval_dataset = langfuse.get_dataset(name=dataset_name)
        print(f"✓ Loaded dataset with {len(llm_eval_dataset.items)} items")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Convert to evaluation items
    print("\n[2/3] Converting dataset...")
    evaluation_items = convert_langfuse_to_ragas_dataset(llm_eval_dataset)
    print(f"✓ Prepared {len(evaluation_items)} items for evaluation")

    if len(evaluation_items) == 0:
        print("✗ No valid data found. Exiting.")
        return

    # Evaluate
    print("\n[3/3] Running evaluation...")
    evaluator = RAGEvaluator()
    results = await evaluator.evaluate_batch(
        evaluation_items=evaluation_items, push_to_langfuse=True
    )

    # Display results
    display_evaluation_results(results, evaluator)


async def evaluate_from_file(filepath: str, push_to_langfuse: bool = True) -> None:
    """
    Evaluate RAG pipeline using a JSON or CSV file.

    Args:
        filepath: Path to JSON or CSV file
        push_to_langfuse: Whether to push results to Langfuse
    """
    print("=" * 60)
    print("RAG EVALUATION FROM FILE")
    print("=" * 60)

    # Load dataset
    print(f"\n[1/3] Loading dataset from {filepath}...")
    try:
        if filepath.endswith(".json"):
            evaluation_items = load_test_dataset_from_json(filepath)
        elif filepath.endswith(".csv"):
            evaluation_items = load_test_dataset_from_csv(filepath)
        else:
            print("✗ Unsupported file format. Use .json or .csv")
            return

        print(f"✓ Loaded {len(evaluation_items)} items")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Validate items
    print("\n[2/3] Validating dataset...")
    valid_items = []
    for item in evaluation_items:
        if all(key in item for key in ["question", "answer", "contexts"]):
            valid_items.append(item)
        else:
            print(f"⚠ Skipping invalid item: {item.get('question', 'Unknown')[:50]}")

    print(f"✓ {len(valid_items)} valid items")

    if len(valid_items) == 0:
        print("✗ No valid items found. Exiting.")
        return

    # Evaluate
    print("\n[3/3] Running evaluation...")
    evaluator = RAGEvaluator()
    results = await evaluator.evaluate_batch(
        evaluation_items=valid_items, push_to_langfuse=push_to_langfuse
    )

    # Display results
    display_evaluation_results(results, evaluator)


async def evaluate_live_queries(
    questions: List[str],
    collection_name: str,
    top_k: int = 4,
    push_to_langfuse: bool = True,
) -> None:
    """
    Evaluate RAG pipeline by running live queries against a collection.

    Args:
        questions: List of questions to ask
        collection_name: Name of the collection to query
        top_k: Number of documents to retrieve
        push_to_langfuse: Whether to push results to Langfuse
    """
    print("=" * 60)
    print("RAG EVALUATION WITH LIVE QUERIES")
    print("=" * 60)

    print(f"\n[1/3] Initializing query pipeline for collection '{collection_name}'...")
    try:
        pipeline = get_query_pipeline(collection_name)
        if not pipeline.vector_store:
            print(f"✗ Collection '{collection_name}' not found or cannot be loaded")
            return
        print(f"✓ Query pipeline ready")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return

    # Execute queries
    print(f"\n[2/3] Executing {len(questions)} queries...")
    evaluation_items = []

    for idx, question in enumerate(questions, 1):
        print(f"  [{idx}/{len(questions)}] Querying: {question[:60]}...")
        try:
            result = pipeline.query(question, collection_name, top_k)

            if result.get("error"):
                print(f"    ✗ Error: {result['error']}")
                continue

            # Extract contexts from sources
            contexts = [source["content"] for source in result.get("sources", [])]

            evaluation_items.append(
                {
                    "question": question,
                    "answer": result["answer"],
                    "contexts": contexts,
                }
            )

            print(f"    ✓ Retrieved {len(contexts)} contexts")

        except Exception as e:
            print(f"    ✗ Query failed: {e}")
            continue

    if len(evaluation_items) == 0:
        print("✗ No successful queries. Exiting.")
        return

    print(f"✓ {len(evaluation_items)} successful queries")

    # Evaluate
    print("\n[3/3] Running evaluation...")
    evaluator = RAGEvaluator()
    results = await evaluator.evaluate_batch(
        evaluation_items=evaluation_items, push_to_langfuse=push_to_langfuse
    )

    # Display results
    display_evaluation_results(results, evaluator)


def display_evaluation_results(results: List[Dict], evaluator: RAGEvaluator) -> None:
    """Display evaluation results and save to CSV"""

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"✓ Evaluated {len(results)} items")

    # Calculate and display average scores
    print("\n" + "=" * 60)
    print("AVERAGE SCORES")
    print("=" * 60)

    average_scores = evaluator.calculate_average_scores(results)

    for metric_name, avg_score in average_scores.items():
        if avg_score is not None:
            print(f"{metric_name:40s}: {avg_score:.4f}")
        else:
            print(f"{metric_name:40s}: N/A")

    # Save results to CSV
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Flatten results for CSV
    flattened_results = []
    for result in results:
        flat_result = {
            "question": result["question"],
            "answer": result["answer"],
            "timestamp": result.get("timestamp", ""),
            "trace_id": result.get("trace_id", ""),
        }

        # Add scores
        for score_name, score_value in result["scores"].items():
            flat_result[f"score_{score_name}"] = score_value

        # Add number of contexts
        flat_result["num_contexts"] = len(result.get("contexts", []))

        flattened_results.append(flat_result)

    df = pd.DataFrame(flattened_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Results saved to '{output_file}'")

    print("\n" + "=" * 60)


async def interactive_evaluation() -> None:
    """Interactive CLI for running evaluations"""
    print("\n" + "=" * 60)
    print("RAG EVALUATION TOOL - INTERACTIVE MODE")
    print("=" * 60)
    print("\nChoose an evaluation method:")
    print("1. Evaluate from Langfuse dataset")
    print("2. Evaluate from JSON/CSV file")
    print("3. Evaluate live queries on a collection")
    print("4. Exit")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        dataset_name = input(
            "Enter Langfuse dataset name [llm_evaluation_dataset]: "
        ).strip()
        if not dataset_name:
            dataset_name = "llm_evaluation_dataset"
        await evaluate_from_langfuse_dataset(dataset_name)

    elif choice == "2":
        filepath = input("Enter path to JSON/CSV file: ").strip()
        if not filepath:
            print("✗ No file path provided")
            return

        push_to_langfuse = (
            input("Push results to Langfuse? (y/n) [y]: ").strip().lower() != "n"
        )
        await evaluate_from_file(filepath, push_to_langfuse)

    elif choice == "3":
        collection_name = input("Enter collection name: ").strip()
        if not collection_name:
            print("✗ No collection name provided")
            return

        print("\nEnter questions (one per line, empty line to finish):")
        questions = []
        while True:
            question = input("> ").strip()
            if not question:
                break
            questions.append(question)

        if not questions:
            print("✗ No questions provided")
            return

        top_k = input("Enter top_k value [4]: ").strip()
        top_k = int(top_k) if top_k else 4

        push_to_langfuse = (
            input("Push results to Langfuse? (y/n) [y]: ").strip().lower() != "n"
        )

        await evaluate_live_queries(questions, collection_name, top_k, push_to_langfuse)

    elif choice == "4":
        print("Goodbye!")
        return

    else:
        print("✗ Invalid choice")


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline with various data sources"
    )
    parser.add_argument(
        "--mode",
        choices=["langfuse", "file", "live", "interactive"],
        default="interactive",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--dataset", help="Langfuse dataset name (for langfuse mode)", default=None
    )
    parser.add_argument("--file", help="Path to JSON/CSV file (for file mode)")
    parser.add_argument(
        "--collection", help="Collection name (for live mode)", default=None
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        help="Questions to evaluate (for live mode)",
        default=None,
    )
    parser.add_argument(
        "--top-k", type=int, default=4, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Don't push results to Langfuse",
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        await interactive_evaluation()

    elif args.mode == "langfuse":
        dataset_name = args.dataset or "llm_evaluation_dataset"
        await evaluate_from_langfuse_dataset(dataset_name)

    elif args.mode == "file":
        if not args.file:
            print("✗ --file argument required for file mode")
            return
        await evaluate_from_file(args.file, not args.no_langfuse)

    elif args.mode == "live":
        if not args.collection:
            print("✗ --collection argument required for live mode")
            return
        if not args.questions:
            print("✗ --questions argument required for live mode")
            return

        await evaluate_live_queries(
            args.questions, args.collection, args.top_k, not args.no_langfuse
        )


if __name__ == "__main__":
    asyncio.run(main())
