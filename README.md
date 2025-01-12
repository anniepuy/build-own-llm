# Project Overview

## Purpose

The project is based off of Sebastian Raschka's "Build a Large Language Model", Manning Publishing. Code is modified for personal style of syntaxing and outputs. The learning objective is to understand how to modify and build a LLM from scratch so that I can apply my knowledge to modify a baseline model to supported advanced RAG techniques without reliance on current frameworks.

## Required Packages

PyTorch
Pyton 3.11

## Chapter Summaries

Chapter 2 moves reader through splitting text, tokenizing text, and then transforming the token IDs to tensors to support the final task of creating embeddings.

Key scripts and their overviews:
tokenizing_text: simple tokenization that does not account for unknown words
unknown_words_tokenizer: builds on the tokenizing text to account for unknown words through filling them in as UNK
bpe_sliding_window: Uses the byte pair encodings that account for unknown words through breaking down the words into characters or known subwords and applying token IDs afterwards. The sliding window approach allows for the tokenization to slide over the text to help predict the next word in the text sequence.
tensor_tokens.py: Takes the token IDs and transforms them into multi-dimensional tensors as preparation for creating embeddings.
creating_embeddings_vectors: Creates the embedding vector weight matrix through using PyTorch's tensors.
posiitonal \_embedding_vectors; Builds on the embedding vector class to account for the position of words or IDs in relation to others.
