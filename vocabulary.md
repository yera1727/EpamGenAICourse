# GenAI Vocabulary

Terms an SME in generative AI application development and usage should know. Explanations are in my own words; terms are taken from `research_report.md` or closely related to it.

| Term | Explanation |
|---|---|
| **LLM (Large Language Model)** | A neural network trained on huge amounts of text that predicts the next token, which lets it answer questions, write text and code, and follow instructions. |
| **Prompt** | The input text (instructions, context, examples) given to a model; the quality of the prompt strongly affects the quality of the answer. |
| **Token** | A small piece of text (a word or part of a word) that a model reads and generates; cost, speed and limits are usually measured in tokens. |
| **Context window** | The maximum amount of text (in tokens) a model can consider at once, including the prompt, documents and its own answer. |
| **Hallucination** | A confident but false or invented output, such as a fake citation, policy or code API; the main reliability risk of generative models. |
| **Grounding** | Forcing the model to base its answer on specific trusted sources (documents, databases) instead of its memory, which reduces hallucinations. |
| **RAG (Retrieval-Augmented Generation)** | An approach where relevant documents are first retrieved from a store and then passed to the model along with the question, so it answers from fresh, company-specific data. |
| **Embedding** | A numeric vector representing the meaning of a text (or image), so that similar content ends up close together and can be found by similarity search. |
| **Vector database** | A storage system optimized for saving embeddings and quickly finding the nearest ones to a query. |
| **Chunking** | Splitting long documents into smaller pieces before indexing them; chunk size and boundaries directly affect retrieval quality. |
| **Permission-aware retrieval** | Retrieval that returns only the documents the current user is allowed to see, preventing leaks of restricted information through the AI. |
| **Fine-tuning** | Additional training of an existing model on a narrow dataset to adapt its style, format or domain knowledge. |
| **AI agent** | A model-driven system that plans steps, calls tools (search, APIs, code execution) and acts toward a goal with limited human supervision. |
| **Agentic workflow** | A multi-step process where an agent decides what to do next, uses tools and checks results, instead of producing a single reply. |
| **Escalation (to a human)** | Handing a conversation over from the AI to a human operator when the bot cannot or should not handle the case. |
| **Guardrails** | Rules and checks around a model (input and output filters, policies, validation) that keep its behavior safe and within scope. |
| **Semantic model** | A description of business entities, metrics and relationships (for example, what "active customer" means) that helps AI generate correct queries on company data. |
| **Text-to-SQL** | Generating a database query from a natural language question. |
| **Ambient AI** | AI that passively listens to a real-world interaction (such as a doctor visit) and produces structured output like notes without manual input. |
| **Voice cloning** | Synthesizing speech that imitates a particular person's voice from a short sample; useful for dubbing but open to abuse. |
| **Deepfake** | Realistic fake audio, image or video of a real person produced by generative models, often used for misinformation or fraud. |
| **Prosody** | The rhythm, stress and intonation of speech; natural prosody is what makes synthetic voices sound human. |
| **Lip-sync** | Matching a speaker's mouth movements in video to the audio, essential for avatars and translated videos. |
| **In silico** | Done by computer simulation rather than in a physical lab, as in in-silico drug candidate screening. |
| **Interpretability** | How well humans can understand why a model produced a given output; low interpretability complicates trust and regulation. |
