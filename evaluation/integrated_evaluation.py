"""
Integrated Evaluation Module for RAG Pipeline

This module provides real-time and batch evaluation capabilities
that can be integrated into the query pipeline and API endpoints.
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_ollama import OllamaEmbeddings
from langfuse import Langfuse, get_client
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
)
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM
from ragas.run_config import RunConfig

load_dotenv()

# Initialize Langfuse
Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

langfuse = get_client()


class RAGEvaluator:
    """
    Handles evaluation of RAG queries using RAGAS metrics.
    Can be used for real-time evaluation or batch processing.
    """

    def __init__(
        self,
        metrics: Optional[List] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize the RAG evaluator.

        Args:
            metrics: List of RAGAS metrics to use. Defaults to standard metrics.
            embedding_model: Embedding model name. Defaults to env variable.
            llm_model: LLM model name for evaluation. Defaults to env variable.
        """
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL")
        self.llm_model = llm_model or os.getenv("CEREBRAS_LLM_MODEL")

        # Default metrics if none provided
        if metrics is None:
            self.metrics = [
                Faithfulness(),
                AnswerRelevancy(strictness=1),
                LLMContextPrecisionWithoutReference(),
            ]
        else:
            self.metrics = metrics

        # Initialize models
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.llm = ChatCerebras(model=self.llm_model, temperature=0)

        # Initialize metrics with models
        self._init_metrics()

    def _init_metrics(self):
        """Initialize RAGAS metrics with LLM and embeddings"""
        for metric in self.metrics:
            if isinstance(metric, MetricWithLLM):
                metric.llm = LangchainLLMWrapper(self.llm)
            if isinstance(metric, MetricWithEmbeddings):
                metric.embeddings = LangchainEmbeddingsWrapper(self.embeddings)
            run_config = RunConfig()
            metric.init(run_config)

    async def evaluate_single_query(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference_answer: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG query-answer pair.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings
            reference_answer: Optional ground truth answer
            trace_id: Optional Langfuse trace ID to attach scores to

        Returns:
            Dictionary containing evaluation scores
        """
        scores = {}

        # Create RAGAS sample
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=answer,
            reference=reference_answer,
        )

        # Calculate each metric
        for metric in self.metrics:
            try:
                score = await metric.single_turn_ascore(sample)
                scores[metric.name] = float(score) if score is not None else None
            except Exception as e:
                print(f"Error calculating {metric.name}: {e}")
                scores[metric.name] = None

        # Push all scores to Langfuse if trace_id provided
        if trace_id:
            # Small delay to ensure trace is fully written
            await asyncio.sleep(0.5)

            print(
                f"📊 Attaching {len([s for s in scores.values() if s is not None])} scores to trace {trace_id}"
            )

            for metric_name, score in scores.items():
                if score is not None:
                    try:
                        langfuse.create_score(
                            name=metric_name,
                            value=float(score),
                            trace_id=trace_id,
                        )
                        print(f"  ✓ {metric_name}: {float(score):.3f}")
                    except Exception as e:
                        print(f"  ✗ Failed to attach {metric_name}: {e}")

            # Flush scores to Langfuse
            langfuse.flush()
            print(f"✅ Evaluation scores flushed to Langfuse for trace {trace_id}")

        return scores

    def evaluate_single_query_sync(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference_answer: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for evaluate_single_query.
        Useful for integration with sync code.
        """
        return asyncio.run(
            self.evaluate_single_query(
                question, answer, contexts, reference_answer, trace_id
            )
        )

    async def evaluate_batch(
        self,
        evaluation_items: List[Dict[str, Any]],
        push_to_langfuse: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of queries.

        Args:
            evaluation_items: List of dicts with 'question', 'answer', 'contexts',
                            and optionally 'reference_answer'
            push_to_langfuse: Whether to create Langfuse traces

        Returns:
            List of evaluation results with scores
        """
        results = []

        print(f"\n{'=' * 60}")
        print(f"Evaluating {len(evaluation_items)} items")
        print(f"{'=' * 60}\n")

        for idx, item in enumerate(evaluation_items, 1):
            question = item["question"]
            answer = item["answer"]
            contexts = item["contexts"]
            reference_answer = item.get("reference_answer")

            print(f"[{idx}/{len(evaluation_items)}] Evaluating: {question[:60]}...")

            trace_id = None

            # Create Langfuse trace if requested
            if push_to_langfuse:
                with langfuse.start_as_current_observation(
                    as_type="span", name="rag_evaluation"
                ) as trace:
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

                    # Evaluate
                    scores = await self.evaluate_single_query(
                        question, answer, contexts, reference_answer, trace_id
                    )

                    result = {
                        "question": question,
                        "answer": answer,
                        "contexts": contexts,
                        "trace_id": trace_id,
                        "scores": scores,
                        "timestamp": datetime.now().isoformat(),
                    }

                    if reference_answer:
                        result["reference_answer"] = reference_answer

                    results.append(result)
                    print(f"  ✓ Scores: {scores}")
            else:
                # Evaluate without Langfuse trace
                scores = await self.evaluate_single_query(
                    question, answer, contexts, reference_answer
                )

                result = {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "scores": scores,
                    "timestamp": datetime.now().isoformat(),
                }

                if reference_answer:
                    result["reference_answer"] = reference_answer

                results.append(result)
                print(f"  ✓ Scores: {scores}")

        # Flush Langfuse data
        if push_to_langfuse:
            langfuse.flush()

        return results

    def calculate_average_scores(
        self, evaluation_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate average scores across multiple evaluations.

        Args:
            evaluation_results: List of evaluation result dicts

        Returns:
            Dictionary with average scores for each metric
        """
        averages = {}

        for metric in self.metrics:
            metric_name = metric.name
            scores = [
                r["scores"][metric_name]
                for r in evaluation_results
                if r["scores"].get(metric_name) is not None
            ]

            if scores:
                averages[metric_name] = sum(scores) / len(scores)
            else:
                averages[metric_name] = None

        return averages


class QueryPipelineEvaluator:
    """
    Wrapper to evaluate QueryPipeline results directly.
    Integrates seamlessly with the existing query pipeline.
    """

    def __init__(self, evaluator: Optional[RAGEvaluator] = None):
        """
        Initialize with an optional RAGEvaluator instance.

        Args:
            evaluator: RAGEvaluator instance. Creates default if None.
        """
        self.evaluator = evaluator or RAGEvaluator()

    async def evaluate_query_result(
        self,
        query_result: Dict[str, Any],
        reference_answer: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a query result from the QueryPipeline.

        Args:
            query_result: Result dict from QueryPipeline.query()
            reference_answer: Optional ground truth answer
            trace_id: Optional Langfuse trace ID

        Returns:
            Query result enhanced with evaluation scores
        """
        # Extract question, answer, and contexts from query result
        question = query_result.get("question", "")
        answer = query_result.get("answer", "")

        # Extract contexts from sources
        sources = query_result.get("sources", [])
        contexts = [source.get("content", "") for source in sources]

        # Evaluate
        scores = await self.evaluator.evaluate_single_query(
            question=question,
            answer=answer,
            contexts=contexts,
            reference_answer=reference_answer,
            trace_id=trace_id,
        )

        # Add scores to result
        enhanced_result = query_result.copy()
        enhanced_result["evaluation_scores"] = scores
        enhanced_result["evaluated_at"] = datetime.now().isoformat()

        return enhanced_result

    def evaluate_query_result_sync(
        self,
        query_result: Dict[str, Any],
        reference_answer: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for evaluate_query_result"""
        return asyncio.run(
            self.evaluate_query_result(query_result, reference_answer, trace_id)
        )


# Singleton instance
_evaluator_instance = None


def get_evaluator(
    metrics: Optional[List] = None,
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> RAGEvaluator:
    """
    Get or create a singleton RAGEvaluator instance.

    Args:
        metrics: List of RAGAS metrics
        embedding_model: Embedding model name
        llm_model: LLM model name

    Returns:
        RAGEvaluator instance
    """
    global _evaluator_instance

    if _evaluator_instance is None:
        _evaluator_instance = RAGEvaluator(
            metrics=metrics,
            embedding_model=embedding_model,
            llm_model=llm_model,
        )

    return _evaluator_instance


# Convenience function for quick evaluation
async def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quick evaluation function for a RAG response.

    Args:
        question: User question
        answer: Generated answer
        contexts: Retrieved contexts
        reference_answer: Optional ground truth
        trace_id: Optional Langfuse trace ID

    Returns:
        Dictionary with evaluation scores
    """
    evaluator = get_evaluator()
    return await evaluator.evaluate_single_query(
        question, answer, contexts, reference_answer, trace_id
    )
