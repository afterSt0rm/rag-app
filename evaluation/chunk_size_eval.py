import json
import os
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langfuse import Evaluation, Langfuse, get_client
from langfuse.experiment import ExperimentItem
from query.config import QueryConfig

# Load environment variables
load_dotenv()

# Initialize Langfuse client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

# Get the configured client instance
langfuse = get_client()


# Define the evaluation schema
class RetrieverRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        List[int],
        ...,
        "Please rate the relevance of each chunk to the question from 1 to 10",
    ]


def create_retriever_task(
    chunk_size: int, chunk_overlap: int, top_k: int, collection_name: str
):
    """Factory function to create a retriever task with specific chunk settings."""

    def retriever_task(*, item: ExperimentItem, **kwargs) -> Dict[str, Any]:
        try:
            question = item.input["question"]

            # Initialize embeddings
            embeddings = OllamaEmbeddings(model=QueryConfig.embedding_model)

            # Construct collection path
            collection_path = Path(QueryConfig.base_vector_store_path) / collection_name

            if not collection_path.exists():
                raise FileNotFoundError(
                    f"Collection path does not exist: {collection_path}"
                )

            # Initialize vector store
            vector_store = Chroma(
                persist_directory=str(collection_path),
                embedding_function=embeddings,
                collection_name=collection_name,
            )

            # Create retriever
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )

            # Retrieve documents
            docs = retriever.invoke(question)

            return {
                "documents": docs,
                "metadata": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "top_k": top_k,
                    "collection_name": collection_name,
                    "num_documents_retrieved": len(docs),
                },
            }

        except Exception as e:
            print(f"Error in retriever task: {str(e)}")
            return {
                "documents": [],
                "error": str(e),
                "metadata": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "top_k": top_k,
                    "collection_name": collection_name,
                },
            }

    return retriever_task


# Define retrieval relevance instructions with structured output format
retrieval_relevance_instructions = """You are evaluating the relevance of a set of documents to a question.
You will be given a QUESTION, an EXPECTED OUTPUT (ground truth answer), and a set of DOCUMENTS retrieved from the retriever.

Grade criteria:
1. Your goal is to identify DOCUMENTS that are completely unrelated to the QUESTION
2. It is OK if the facts have SOME information that is unrelated as long as it is relevant to answering the question
3. Rate each document on a scale of 1-10, where:
   - 1-3: Completely irrelevant to the question
   - 4-6: Somewhat relevant but contains mostly irrelevant information
   - 7-8: Mostly relevant with some irrelevant parts
   - 9-10: Highly relevant and directly useful for answering the question

Return your evaluation in JSON format with exactly these keys:
- "explanation": A brief explanation of your scoring reasoning
- "relevant": A list of integers (1-10) representing the relevance score for each document in the same order as provided

IMPORTANT: Return ONLY valid JSON with no additional text or formatting.
"""

# Initialize the LLM with structured output parsing
json_parser = JsonOutputParser(pydantic_object=RetrieverRelevanceGrade)

# Create prompt template for structured output
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", retrieval_relevance_instructions),
        (
            "human",
            "QUESTION: {question}\n\nEXPECTED OUTPUT: {expected_output}\n\nDOCUMENTS:\n{documents}",
        ),
    ]
)

# Initialize the LLM
retrieval_relevance_llm = ChatCerebras(
    model=QueryConfig.llm_model, temperature=0, max_tokens=1000
)

# retrieval_relevance_llm = ChatOllama(model=os.getenv("OLLAMA_LLM_MODEL"), temperature=0)

# Chain for structured output
retrieval_relevance_chain = prompt_template | retrieval_relevance_llm | json_parser


def relevant_chunks_evaluator(
    *, input, output, expected_output, metadata, **kwargs
) -> Evaluation:
    """Evaluator function to assess the relevance of retrieved chunks."""
    try:
        # Check if there was an error in retrieval
        if "error" in output:
            return Evaluation(
                name="retrieval_relevance",
                value=0.0,
                comment=f"Retrieval failed: {output['error']}",
            )

        documents = output.get("documents", [])
        if not documents:
            return Evaluation(
                name="retrieval_relevance", value=0.0, comment="No documents retrieved"
            )

        # Format documents for evaluation
        formatted_docs = []
        for i, doc in enumerate(documents):
            formatted_docs.append(
                f"DOCUMENT {i + 1}:\n{doc.page_content}"
            )  # Limit content length

        documents_text = "\n\n".join(formatted_docs)

        # Get LLM evaluation with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = retrieval_relevance_chain.invoke(
                    {
                        "question": input["question"],
                        "expected_output": expected_output["answer"],
                        "documents": documents_text,
                    }
                )

                # Validate the result structure
                if (
                    isinstance(result, dict)
                    and "relevant" in result
                    and "explanation" in result
                ):
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # Wait before retrying

        # Calculate average relevance score
        relevance_scores = result["relevant"]

        # Handle case where we have fewer scores than documents
        if len(relevance_scores) < len(documents):
            # Pad with minimum scores for missing documents
            relevance_scores.extend([1] * (len(documents) - len(relevance_scores)))
        elif len(relevance_scores) > len(documents):
            # Truncate if we have more scores than documents
            relevance_scores = relevance_scores[: len(documents)]

        avg_score = (
            sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        )

        # Create detailed comment with individual scores
        score_details = "\n".join(
            [
                f"Document {i + 1}: {score}/10"
                for i, score in enumerate(relevance_scores)
            ]
        )
        full_comment = f"{result['explanation']}\n\nIndividual scores:\n{score_details}"

        return Evaluation(
            name="retrieval_relevance",
            value=round(avg_score, 2),
            comment=full_comment,
            metadata={
                "individual_scores": relevance_scores,
                "num_documents": len(documents),
                "chunk_settings": metadata,
            },
        )

    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return Evaluation(
            name="retrieval_relevance",
            value=0.0,
            comment=f"Evaluation failed: {str(e)}",
            metadata={"error": str(e)},
        )


def run_benchmark_experiments():
    """Main function to run benchmark experiments."""
    try:
        # Get dataset
        dataset_name = input(
            "Enter the dataset you would like to evaluate on (default: LLM_Dataset): "
        ).strip()
        dataset_name = dataset_name or "LLM_Dataset"

        try:
            dataset = langfuse.get_dataset(name=dataset_name)
            print(f"Loaded dataset: {dataset_name} with {len(dataset.items)} items")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {str(e)}")
            print("Falling back to default dataset 'LLM_Dataset'")
            dataset = langfuse.get_dataset(name="LLM_Dataset")

        # Get collection name
        collection = input(
            "Enter the collection name you would like to evaluate (default: LLM): "
        ).strip()
        collection = collection or "LLM"

        # Get top_k value
        top_k_input = input(
            "Enter the top_k value for retrieval (default: 4): "
        ).strip()
        try:
            top_k = int(top_k_input) if top_k_input else 4
            if top_k <= 0:
                print("Invalid top_k value. Using default: 4")
                top_k = 4
        except ValueError:
            print("Invalid input. Using default top_k: 4")
            top_k = 4

        # Configuration parameters
        chunk_sizes = [128, 256, 512, 1024]
        overlap_percentages = [0, 20, 30]

        print("\nStarting benchmark experiments...")
        print("=" * 60)
        print(f"Configuration: top_k = {top_k}")
        print(f"Chunk sizes: {chunk_sizes}")
        print(f"Overlap percentages: {overlap_percentages}%")
        print("=" * 60)

        for chunk_size in chunk_sizes:
            print(f"\nRunning experiments for chunk_size: {chunk_size}")
            print("-" * 40)

            for overlap_pct in overlap_percentages:
                chunk_overlap = int(chunk_size * overlap_pct / 100)

                experiment_name = f"Chunk precision: chunk_size {chunk_size} and chunk_overlap {chunk_overlap} ({overlap_pct}%)"
                print(f"Running: {experiment_name}")

                try:
                    dataset.run_experiment(
                        name=experiment_name,
                        task=create_retriever_task(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            top_k=top_k,
                            collection_name=collection,
                        ),
                        evaluators=[relevant_chunks_evaluator],
                        metadata={
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "overlap_percentage": overlap_pct,
                            "top_k": top_k,
                        },
                    )
                    print("✓ Completed successfully")
                except Exception as e:
                    print(f"✗ Failed: {str(e)}")

        print("\n" + "=" * 60)
        print("All experiments completed successfully!")
        print("Flushing Langfuse client...")

        # Flush to ensure all data is sent
        langfuse.flush()
        print("✓ Langfuse client flushed successfully")

    except Exception as e:
        print(f"\nCritical error during benchmark execution: {str(e)}")
        print("Attempting to flush Langfuse client anyway...")
        try:
            langfuse.flush()
        except:
            pass
        raise


if __name__ == "__main__":
    print("Retriever Benchmarking Script")
    print("=============================")
    print(f"Using LLM Model: {QueryConfig.llm_model}")
    print(f"Using Embedding Model: {QueryConfig.embedding_model}")
    print(f"Base Vector Store Path: {QueryConfig.base_vector_store_path}")
    print()

    run_benchmark_experiments()
