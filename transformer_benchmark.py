from transformers import BertForMaskedLM, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import horovod.torch as hvd
import time 
import argparse
from transformers import AutoConfig, AutoModelForSequenceClassification, \
AutoModelForTokenClassification, T5ForConditionalGeneration, AutoModelForSeq2SeqLM,\
EncoderDecoderConfig, EncoderDecoderModel, AutoModelForMultipleChoice
from SyntheticDatasets import *
from CustomModels import *
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--task", default='classification', type=str)
parser.add_argument("--model", default='bert-base-uncased',type=str)
parser.add_argument("--use_default", default=False, action="store_true")
parser.add_argument("--epochs", default=3, type=int)

parser.add_argument("--hidden_size", default=512, type=int, required=False)
parser.add_argument("--num_hidden_layers", default=6, type=int, required=False)
parser.add_argument("--num_attention_heads", default=6, type=int, required=False)

parser.add_argument("--vocab_size", default=30522, type=int)
parser.add_argument("--max_seq_len", default=512, type=int)
parser.add_argument("--ffn_dim", default=2048, type=int)

parser.add_argument("--dec_num_attention_heads", default=8, type=int)
parser.add_argument("--dec_num_hidden_layers", default=6, type=int)
parser.add_argument("--dec_ffn_dim", default=2048, type=int)
parser.add_argument("--dec_max_seq_len", default=512, type=int)

parser.add_argument("--num_labels", default=2, type=int, required=True)
parser.add_argument("--num_examples", default=1000, type=int)

args = parser.parse_args()

print(args)

if args.model!='custom':
  if args.use_default:
    print("Using default configuration for {}".format(args.model))
    config = AutoConfig.from_pretrained(args.model, vocab_size=args.vocab_size)
    if args.task in ['translation', 'summarization']:
      config_encoder = config
      config_decoder = AutoConfig.from_pretrained(args.model, vocab_size=args.num_labels, is_decoder=True, add_cross_attention=True)
  else:
    print("Using custom configuration for {}".format(args.model))
    config = AutoConfig.from_pretrained(args.model, vocab_size=args.vocab_size, hidden_size=args.hidden_size,
    num_hidden_layers = args.num_hidden_layers, num_attention_heads=args.num_attention_heads, intermediate_size=args.ffn_dim)
    if args.task in ['translation', 'summarization']:
      config_encoder = config
      config_decoder = AutoConfig.from_pretrained(args.model, vocab_size=args.num_labels, hidden_size=args.dec_hidden_size,
    num_hidden_layers = args.dec_num_hidden_layers, num_attention_heads=args.dec_num_attention_heads, intermediate_size=args.dec_ffn_dim, \
    is_decoder=True, add_cross_attention=True)
  if 'bert' in args.model:
    config.max_position_embeddings = args.max_seq_len
    if args.task in ['translation', 'summarization']:
      config_encoder.max_position_embeddings = args.max_seq_len
      config_decoder.max_position_embeddings = args.dec_max_seq_len
    if 'roberta' in args.model:
      config.max_position_embeddings +=2
      if args.task in ['translation', 'summarization']:
        config_encoder.max_position_embeddings += 2
        config_decoder.max_position_embeddings += 2
  config.num_labels =args.num_labels
    
if args.task == "classification":
  if 't5' in args.model:
    model = T5ForConditionalGeneration(config)
  elif args.model =='custom':
    model = EncoderSequenceClassification(args.vocab_size, args.hidden_size, args.num_attention_heads, args.num_hidden_layers, args.num_labels)
  else:
    try:
      model = AutoModelForSequenceClassification.from_config(config)
    except:
      raise ValueError("Cannot initialize a {} model for {} task".format(args.model, args.task))
  train_dataset = SyntheticDataset(args.vocab_size, args.num_examples, args.num_labels, args.max_seq_len, 1)

elif args.task == "ner":
  if 't5' in args.model:
    model = T5ForConditionalGeneration(config)
  elif args.model =='custom':
    model = Encoder(args.vocab_size, args.hidden_size, args.num_attention_heads, args.num_hidden_layers, args.num_labels)
  else:
    try:
      model = AutoModelForTokenClassification.from_config(config)
    except:
      raise ValueError("Cannot initialize a {} model for {} task".format(args.model, args.task))
  train_dataset = SyntheticDataset(args.vocab_size, args.num_examples, args.num_labels, args.max_seq_len, args.max_seq_len)

elif args.task == "mcq":
  try:
    model = AutoModelForMultipleChoice.from_config(config)
  except:
    raise ValueError("Cannot initialize a {} model for {} task".format(args.model, args.task))
  train_dataset = SyntheticMCQDataset(args.vocab_size, args.num_examples, args.num_labels, args.max_seq_len)

elif args.task in ['translation', 'summarization']:
  print("Doing Transalation")
  if 't5' in args.model:
    model = AutoModelForSeq2SeqLM.from_config(config)
    config.vocab_size = max(args.vocab_size, args.num_labels)
  elif args.model == 'custom':
    print("setting up custom model")
    model = EncoderDecoder(args.vocab_size, args.hidden_size, args.num_attention_heads, args.num_hidden_layers, args.dec_num_attention_heads, args.dec_num_hidden_layers, args.ffn_dim, args.num_labels)
  else:
    try:
      config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
      model = EncoderDecoderModel(config=config)
    except:
       raise ValueError("Cannot initialize a {} model for {} task".format(args.model, args.task))
  if args.task == "translation":
    train_dataset = SyntheticDataset(args.vocab_size, args.num_examples, args.num_labels, args.max_seq_len, args.dec_max_seq_len)
  elif args.task == 'summarization':
    train_dataset = SyntheticDataset(args.vocab_size, args.num_examples, args.vocab_size, args.max_seq_len, args.dec_max_seq_len)

else:
  raise ValueError("Task {} not implemented".format(args.task))

if args.model!='custom':
  print("Configuration of the model", model.config)
  print("Model", model)
else:
  print("Transformer model:", model)

optimizer = torch.optim.Adam(lr=1e-5, params = model.parameters())
if torch.cuda.is_available():
  hvd.init()
  torch.cuda.set_device(hvd.local_rank())
  device = 'cuda'
  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
  optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
else:
  device='cpu'
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

model = model.to(device)
model.train()
print("Start of Training")
if args.task in ['translation', 'summarization']:
  for i in range(args.epochs):
      start = time.time()
      for batch in tqdm(train_loader):
          optimizer.zero_grad()
          i_ids = batch['input'].to(device)
          l = batch['labels'].to(device)
          outputs = model(input_ids=i_ids, labels=l, decoder_input_ids =l)
          loss = outputs.loss
          loss.backward()
          optimizer.step()
      end = time.time()
      print("Time for epoch {} = {}".format(i, end-start))  
else:
  for i in range(args.epochs):
      start = time.time()
      for batch in tqdm(train_loader):
          optimizer.zero_grad()
          i_ids = batch['input'].to(device)
          l = batch['labels'].to(device)
          outputs = model(input_ids=i_ids, labels=l)
          loss = outputs.loss
          loss.backward()
          optimizer.step()
      end = time.time()
      print("Time for epoch {} = {}".format(i, end-start))  
print("End of Training")
