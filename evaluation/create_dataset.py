import os

from dotenv import load_dotenv
from langfuse import Langfuse, get_client

load_dotenv()

Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

# Get the configured client instance
langfuse = get_client()

if __name__ == "__main__":
    # Create dataset
    langfuse.create_dataset(
        name="llm_dataset",
        # optional description
        description="Evaluation dataset for LLM collection.",
        # optional metadata
        metadata={"author": "Aashish", "date": "2025-12-10", "type": "benchmark"},
    )

    # Create dataset
    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is zero-shot prompting or general prompting?"},
        # any python object or value, optional
        expected_output={
            "answer": "A zero-shot prompt is the simplest type of prompt. It only provides a description of a task and some text for the LLM to get started with. This input could be anything: a question, a start of a story, or instructions. The name zero-shot stands for ’no examples’."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is one-shot prompting?"},
        # any python object or value, optional
        expected_output={
            "answer": "A one-shot prompt, provides a single example, hence the name one-shot. The idea is the model has an example it can imitate to best complete the task."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is few-shot prompting?"},
        # any python object or value, optional
        expected_output={
            "answer": "A few-shot prompt provides multiple examples to the model. This approach shows the model a pattern that it needs to follow. The idea is similar to one-shot, but multiple examples of the desired pattern increases the chance the model follows the pattern."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is system prompting?"},
        # any python object or value, optional
        expected_output={
            "answer": "System prompting sets the overall context and purpose for the language model. It defines the ‘big picture’ of what the model should be doing, like translating a language, classifying a review etc."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is contextual prompting?"},
        # any python object or value, optional
        expected_output={
            "answer": "Contextual prompting provides specific details or background information relevant to the current conversation or task. It helps the model to understand the nuances of what’s being asked and tailor the response accordingly."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is role prompting?"},
        # any python object or value, optional
        expected_output={
            "answer": "Role prompting assigns a specific character or identity for the language model to adopt. This helps the model generate responses that are consistent with the assigned role and its associated knowledge and behavior."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is step-back prompting?"},
        # any python object or value, optional
        expected_output={
            "answer": "Step-back prompting is a technique for improving the performance by prompting the LLM to first consider a general question related to the specific task at hand, and then feeding the answer to that general question into a subsequent prompt for the specific task. This ‘step back’ allows the LLM to activate relevant background knowledge and reasoning processes before attempting to solve the specific problem."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is chain of thought prompting?"},
        # any python object or value, optional
        expected_output={
            "answer": "Chain of Thought (CoT) prompting is a technique for improving the reasoning capabilities of LLMs by generating intermediate reasoning steps. This helps the LLM generate more accurate answers."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is react prompting?"},
        # any python object or value, optional
        expected_output={
            "answer": "Reason and act (ReAct) prompting is a paradigm for enabling LLMs to solve complex tasks using natural language reasoning combined with external tools (search, code interpreter etc.) allowing the LLM to perform certain actions, such as interacting with external APIs to retrieve information which is a first step towards agent modeling."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is self-consistency prompting?"},
        # any python object or value, optional
        expected_output={
            "answer": "Self-consistency combines sampling and majority voting to generate diverse reasoning paths and select the most consistent answer. It improves the accuracy and coherence of responses generated by LLMs. Self-consistency gives a pseudo-probability likelihood of an answer being correct, but obviously has high costs."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is tree of thoughts?"},
        # any python object or value, optional
        expected_output={
            "answer": "Now that we are familiar with chain of thought and self-consistency prompting, let’s review Tree of Thoughts (ToT). It generalizes the concept of CoT prompting because it allows LLMs to explore multiple different reasoning paths simultaneously, rather than just following a single linear chain of thought. \nThis approach makes ToT particularly well-suited for complex tasks that require exploration. It works by maintaining a tree of thoughts, where each thought represents a coherent language sequence that serves as an intermediate step toward solving a problem. The model can then explore different reasoning paths by branching out from different nodes in the tree."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "How do I deploy the model as onnx?"},
        # any python object or value, optional
        expected_output={
            "answer": "In order to export a model to ONNX, we need to run a model with a dummy input: the values of the input tensors don’t really matter; what matters is that they are the correct shape and type. By invoking the torch.onnx.export function, PyTorch will trace the computations performed by the model and serialize them into an ONNX file with the provided name: torch.onnx.export(seg_model, dummy_input, 'seg_model.onnx')"
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "Why should I use PyTorch?"},
        # any python object or value, optional
        expected_output={
            "answer": "PyTorch is easy to recommend because of its simplicity. Many researchers and practitioners find it easy to learn, use, extend, and debug. It’s Pythonic, and while like any complicated domain it has caveats and best practices, using the library generally feels familiar to developers who have used Python previously."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "Can you tell me about the AlexNet"},
        # any python object or value, optional
        expected_output={
            "answer": "The AlexNet architecture won the 2012 ILSVRC by a large margin, with a top-5 test error rate (that is, the correct label must be in the top 5 predictions) of 15.4%. By comparison, the second-best submission, which wasn’t based on a deep network, trailed at 26.2%. This was a defining moment in the history of computer vision: the moment when the community started to realize the potential of deep learning for vision tasks. That leap was followed by constant improvement, with more modern architectures and training methods getting top-5 error rates as low as 3%. A pretrained network that recognizes the subject of an image"
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is GAN game?"},
        # any python object or value, optional
        expected_output={
            "answer": "In the context of deep learning, what we’ve just described is known as the GAN game, where two networks, one acting as the painter and the other as the art historian, com- pete to outsmart each other at creating and detecting forgeries. GAN stands for generative adversarial network, where generative means something is being created (in this case, fake masterpieces), adversarial means the two networks are competing to outsmart the other, and well, network is pretty obvious. These networks are one of the most original outcomes of recent deep learning research."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is CycleGAN?"},
        # any python object or value, optional
        expected_output={
            "answer": "An interesting evolution of this concept is the CycleGAN. A CycleGAN can turn images of one domain into images of another domain (and back), without the need for us to explicitly provide matching pairs in the training set."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is Linear Algebra?"},
        # any python object or value, optional
        expected_output={
            "answer": "Linear algebra is a branch of mathematics that is widely used throughout science and engineering. However, because linear algebra is a form of continuous rather than discrete mathematics, many computer scientists have little experience with it."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What are scalars?"},
        # any python object or value, optional
        expected_output={
            "answer": "A scalar is just a single number, in contrast to most of the other objects studied in linear algebra, which are usually arrays of multiple numbers."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What is a vector?"},
        # any python object or value, optional
        expected_output={
            "answer": "A vector is an array of numbers. The numbers are arranged inorder. We can identify each individual number by its index in that ordering."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What are matrices?"},
        # any python object or value, optional
        expected_output={
            "answer": "A matrix is a 2-D array of numbers, so each element is identiﬁed by two indices instead of just one."
        },
    )

    langfuse.create_dataset_item(
        dataset_name="llm_dataset",
        # any python object or value, optional
        input={"question": "What are tensors?"},
        # any python object or value, optional
        expected_output={
            "answer": "In some cases we will need an array with more than two axes. In the general case, an array of numbers arranged on a regular grid with a variable number of axes is known as a tensor. "
        },
    )
