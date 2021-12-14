from transformers.file_utils import ModelOutput
from torch import nn
from torch import optim
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, Transformer 

class EncoderDecoder(nn.Module):
    def __init__(self, en_vocab_size, d_model, nhead, num_layers, dec_nhead, dec_num_layers,dim_feedforward, num_labels):
        super(EncoderDecoder, self).__init__()
        self.en_emd = nn.Embedding(en_vocab_size, d_model)
        self.de_emd = nn.Embedding(num_labels, d_model)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=dec_nhead)
        transformer_decoder = TransformerDecoder(decoder_layer, num_layers=dec_num_layers)
        self.model = Transformer(custom_encoder=transformer_encoder, custom_decoder=transformer_decoder, d_model = d_model, dim_feedforward=dim_feedforward)
        self.fcn = nn.Linear(d_model, num_labels)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, labels, decoder_input_ids):
        src = input_ids.permute((1,0))
        tgt = decoder_input_ids.permute((1,0))
        en_embed_out = self.en_emd(src)
        de_embed_out = self.de_emd(tgt)
        out = self.model(en_embed_out, de_embed_out)
        out = self.fcn(out)
        loss = self.loss(out.view(-1, out.size()[-1]), labels.view(-1))
        ModelOutput.loss = loss
        ModelOutput.out = out.permute(1,0,2)
        return ModelOutput

class Encoder(nn.Module):
    def __init__(self, en_vocab_size, d_model, nhead, num_layers, num_labels):
        super(Encoder, self).__init__()
        self.en_emd = nn.Embedding(en_vocab_size, d_model)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.model = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fcn = nn.Linear(d_model, num_labels)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, labels):
        src = input_ids.permute((1,0))
        en_embed_out = self.en_emd(src)
        out = self.model(en_embed_out)
        out = self.fcn(out)
        loss = self.loss(out.view(-1, out.size()[-1]), labels.view(-1))
        ModelOutput.loss = loss
        ModelOutput.out = out.permute(1,0,2)
        return ModelOutput

class EncoderSequenceClassification(Encoder):
    def forward(self, input_ids, labels):
        src = input_ids.permute((1,0))
        en_embed_out = self.en_emd(src)
        out = self.model(en_embed_out)
        out = self.fcn(out)
        out = out.sum(dim=0)
        print(out.shape)
        print(labels.shape)
        loss = self.loss(out.view(-1, out.size()[-1]), labels.view(-1))
        ModelOutput.loss = loss
        ModelOutput.out = out
        return ModelOutput    

