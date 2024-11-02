# Finance-Pulse using llm
Minor Project
Part 1: Fine-Tuning LLaMa 3 with Unsloth

1.Installation:

Installs Unsloth (optimization library for faster fine-tuning), Transformers, Xformers, and other required libraries.
Sets up API keys for accessing gated models (like LLaMa) and SEC filings data.

2.Model Initialization:

Loads the pre-trained LLaMa model and tokenizer from the Hugging Face model hub.
Applies Unsloth patching to potentially speed up training.
Sets various options for model loading like maximum sequence length, data type, and enabling 4-bit quantization.
Fine-Tuning Dataset Preparation:

Defines a prompt format for training data which includes user question, context, and expected response.
Loads the financial question answering dataset ("virattt/llama-3-8b-financialQA") and formats it according to the prompt.

3.Trainer Arguments:

Defines training arguments like batch size, gradient accumulation steps, learning rate, optimizer, etc. for the SFTTrainer from the TRL library.

4.Training:

Initializes the SFTTrainer with the model, tokenizer, dataset, and training arguments.
Trains the model on the financial question answering dataset using Unsloth.
Unsloth detects an outdated version of Transformers library and recommends updating for a gradient accumulation bug fix.
Saving the Fine-Tuned Model:

Saves the fine-tuned model and tokenizer locally to Google Drive.
Loading the Fine-Tuned Model (Optional):

Provides a function to load the fine-tuned model and tokenizer back from saved location.

5.Inference Functions:

Defines functions for generating responses to user questions given context:
inference: Takes question and context as input, generates tokens with the model, and decodes them into text.
extract_response: Extracts the generated response from the full output.

Part 2: Setting Up SEC 10-K Data Pipeline & Retrieval Functionality

1.10-K Retrieval Function:

Defines get_filings function that takes a ticker symbol as input.
Uses the SEC API key to retrieve the most recent 10-K filing for the ticker from the SEC Edgar database.
Extracts text from sections 1A (Risk Factors) and 7 (Management's Discussion and Analysis) of the 10-K filing.
2.Embeddings Setup:

Defines the path to a pre-trained sentence encoding model ("BAAI/bge-large-en-v1.5").
Initializes a Hugging Face embedding object for the chosen model with CUDA acceleration. Note: This uses the deprecated HuggingFaceEmbeddings class. Consider updating to the recommended langchain-huggingface package.
Overall, this code demonstrates how to fine-tune a large language model for financial question answering and integrate it with an API for retrieving relevant financial data.

Understanding the Final Part: Generating Responses

In the final part of the project, the system aims to provide informative and accurate answers to user queries related to financial information. This is achieved through a combination of techniques:

1.Query Understanding:

The system first analyzes the user's query to identify the key information being sought.
This involves natural language processing techniques to extract keywords and intent.

2.Context Retrieval:

The system accesses the vector database to retrieve relevant passages from the 10-K filings.
This is done by calculating the similarity between the query embedding and the embeddings of the stored documents.

3.Response Generation:

The fine-tuned LLM takes the query and retrieved context as input.
It generates a response by leveraging its knowledge and understanding of the financial domain.
The LLM can generate text, code, or other creative content.

4.Response Post-Processing:

The generated response may require further processing, such as:
Fact-checking: Ensuring the accuracy of the information provided.
Summarization: Condensing the response into a concise and informative summary.
Formatting: Structuring the response in a clear and readable format.

Example:

User Query: "What were the primary factors affecting Apple's revenue growth in 2023?"

System Response:

Apple's revenue growth in 2023 was primarily driven by strong sales of the iPhone 14 series, increased demand for MacBooks and iPads, and growth in the Services segment. However, foreign exchange rate fluctuations and supply chain constraints had a negative impact on revenue.
