# ChatBot-llm
AI ChatBot is a hands-on project aimed at learning how to build a multimodal chat application. This project is all about integrating different AI models to handle audio, images, and PDFs in a single chat interface. It's a great way for anyone interested in AI and software development to get practical experience with these technologies.

The main purpose here is to learn by doing. You'll see how different pieces like Whisper AI for audio, LLaVA for image processing, and Chroma DB for PDFs come together in a chat application.

## Features

### Quantized Model Integration
This app uses what are called "quantized models." These are special because they are designed to work well on regular consumer hardware, like the kind most of us have at home or in our offices. Normally, the original versions of these models are really big and need more powerful computers to run them. But quantized models are optimized to be smaller and more efficient, without losing much performance. This means you can use this app and its features without needing a super powerful computer. Quantized Models from TheBloke.

### Audio Chatting with Whisper AI
Leveraging Whisper AI's robust transcription capabilities, this app offers a sophisticated audio messaging experience. The integration of Whisper AI allows for accurate interpretation and response to voice inputs, enhancing the natural flow of conversations. Whisper models.

### Image Chatting with LLaVA
The app integrates LLaVA for image processing, which is essentially a fine-tuned LLaMA model equipped to understand image embeddings. These embeddings are generated using a CLIP model, making LLaVA function like a pipeline that brings together advanced text and image understanding. With LLaVA, the chat experience becomes more interactive and engaging, especially when it comes to handling and conversing about visual content. llama-cpp-python repo for Llava loading.

### PDF Chatting with Chroma DB
The app is tailored for both professional and academic uses, integrating Chroma DB as a vector database for efficient PDF interactions. This feature allows users to engage with their own PDF files locally on their device. Whether it's for reviewing business reports, academic papers, or any other PDF document, the app offers a seamless experience. It provides an effective way for users to interact with their PDFs, leveraging the power of AI to understand and respond to content within these documents. This makes it a valuable tool for personal use, where one can extract insights, summaries, and engage in a unique form of dialogue with the text in their PDF files. Chroma website.

## Getting Started

To get started with Local Multimodal AI Chat, clone the repository and follow these simple steps:

1. **Create a Virtual Environment**: I am using Python 3.10.12 currently.

2. **Upgrade pip**: 
   ```sh
   pip install --upgrade pip
   ```
3. **Install Requirements**:
   ```sh
   pip install -r requirements.txt
   ```
4.  **Setting Up Local Models**:Download the models you want to implement. Here is the llava model I used for image chat (ggml-model-q5_k.gguf and mmproj-model-f16.gguf). And the quantized mistral model form TheBloke (mistral-7b-instruct-v0.1.Q5_K_M.gguf).
5.  **Customize config file**: Check the config file and change accordingly to the models you downloaded.Optional - Change Profile Pictures: Place your user_image.pnd and/or bot_image.png inside the chat_icons folder.
6.  **Enter commands in terminal**:
```sh
i.python3 database_operations.py 
ii.streamlit run app.py
```
