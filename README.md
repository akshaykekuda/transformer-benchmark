# transformer-benchmark

## Introduction
Benchmarks for evaluating the training/inference time of Transformer models have not yet been fully developed. The goal of the project is to design a benchmark suite to evaluate the training/inference time of various task specific Transformer models. 

Tasks Supported: Sequence Classification, Token Classification (NER), Translation, Summarization
Models: All variants of bert-like models, XLNet, T5, Build your own transformer
Distributed Training Framework: Pytorch Horovod

## Features
1.	Customization of Model parameters for pretrained models (Bert, Albert, Roberta, Distilbert, XLnet, T5 and many more)

2.	Customizable Synthetic Dataset for each task.

3.	Build your Transformer from scratch

## Models and Tasks

|Model|Sentence Classification|NER|MCQ|Summarization|Translation|
|---|---|---|---|---|---|
|bert-base-uncased|&check;|&check;|&check;|&check;|&check;|
|albert-base-v2|&check;|&check;|&check;|&cross;|&cross;|
|roberta-base|&check;|&check;|&check;|&check;|&check;|
|xlnet-base-cased|&check;|&check;|&check;|&cross;|&cross;|
|t5-small|&check;|&check;|&cross;|&check;|&check;|
|Custom|&check;|&check;|&cross;|&check;|&check;|

The benchmark can also be extended to evaluate the following additional models:

**Sentence Classification**: BART, Camembert, deberta, Flaubert, GPT, Longformer, XLM, Megatron-Bert, Rembert, Funnel Transformer, squeezebert, Convbert, mobile-bert

**Named Entity Recognition**: camembert, convbert, Flaubert, distilbert, funnel transformer, gpt2, longformer, megatron-bert, mobilebert, Rembert, squeezebert, xlm

**Multiple Choice QA**: camembert, convbert, Flaubert, distilbert, funnel transformer, longformer, megatron-bert, mobilebert, Rembert, squeezebert, xlm

**Summarization & Translation**: bart, camembert, megatron-bert, Rembert, xlm

## Setup

The generic models are based on huggingface’s pre-trained models(https://huggingface.co/docs/transformers/model_summary). 

The following arguments can be used to set up your model for benchmarking:

### General Arguments
--batch_size : Batch Size to use during training. Can vary this according to what fits the best in your GPU.

--model : Model to use for a task. If using pretrained models use the names as such from huggingface. Tested Choices: [bert-base-uncased, albert-base-v2, roberta-base, xlnet-base-cased, t5-small, custom]. You could also use variations of these models like bert-base-cased, t5-large, and even new models like distilbert-base-uncased.

--task : Tasks supported. Choices: [classification, ner, mcq, translation, summarization]

--epochs: Number of epochs to train

--use_custom_config: Use this argument if you would like to customize the pretrained models. Do not use this argument if you would like to use the default configurations. 

### Dataset Specifc Arguments

Available arguments: --num_labels, --vocab_size, --max_seq_len, --num_examples, --dec_max_seq_len

These arguments help you setup the dataset for the supported tasks:

vocab_size – set this to the input vocab size of your dataset (default=30522)

max_seq_len- set this to generate the sequences with max_seq_len. This is also the max_seq_len for the transformer models (default=512)

num_examples - set this to the number of samples you want in your dataset

Below are some task specific arguments:

**classification**:

num_labels – set this to the number of classes for classification (default=2)

dec_max_seq_len – unused

**ner**:

num_labels – set this to the number of tags you want in your dataset (default=2)

dec_max_seq_len – unused

**mcq**:

num_labels – set this to the number of options for a question you want in your dataset (default=2)

dec_max_seq_len – unused

**translation**:

num_labels – set this to the output/decoder vocab size you want in your dataset (default=2)

dec_max_seq_len – set this to generate the output sequences with dec_max_seq_len. This is also the dec_max_seq_len for the transformer models (default=512)

**summarization**:

num_labels – unused.

vocab_size – set this to the input vocab size of your dataset (default=30522). For summarization task this is also the vocab size for your decoder.

dec_max_seq_len – set this to generate the output sequences with dec_max_seq_len. This is also the dec_max_seq_len for the decoder of transformer models (default=512)

### Model Specific Arguments

Use these arguments if you want to change the configuration of the pretrained models or to build your custom transformer:

--hidden_size – set this to the dimension of the model you want

--num_hidden_layers – set this to modify the number of layers in the model.

--num_attention_heads- set this to the number of attention heads you want in each layer. Make sure the model dimension is divisible by the number of attention heads.

--ffn_dim- set this to change the dimension of the feedforward layer after the self-attention layers.

These configurations apply to encoder only, decoder only and encoder of the encoder-decoder models. For the custom transformer model, these configurations setup the encoder of the transformer.

For configuring the decoder in the encoder-decoder models (custom transformer, bert-bert) for seq2seq tasks use the following arguments:

--dec_num_hidden_layers – set this to modify the number of layers in the decoder.

--dec_num_attention_heads- set this to the number of attention heads you want in each layer of decoder. Make sure the model dimension is divisible by the number of attention heads.

--dec_ffn_dim- set this to change the dimension of the feedforward layer after the self-attention layers in the decoder.

