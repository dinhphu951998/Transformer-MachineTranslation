from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_target, src_lang, target_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')])
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id('[EOS]')])
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id('[PAD]')])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]

        # split words from sentence and get ids for each word
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids 
        dec_input_tokens = self.tokenizer_target.encode(target_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # 1 for sos, 1 for eos => -2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # 1 for sos only

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # add sos and eos to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens),
                self.eos_token,
                torch.tensor([self.pad_token]* enc_num_padding_tokens)
            ]
        )

        # add sos to decoder input
        decoder_input = torch.cat(
                    [
                        self.sos_token,
                        torch.tensor(dec_input_tokens),
                        torch.tensor([self.pad_token]* dec_num_padding_tokens)
                    ]
                )

        # Add eos to the label - what we expect as output from decoder
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens),
                self.eos_token,
                torch.tensor([self.pad_token]* dec_num_padding_tokens)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len). Tại sao unsqueeze 2 lần?
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len) ,
            "label": label,
            "src_text": src_text,
            "target_text": target_text
        }

def causal_mask(size):
    # Everything above the diagonal will be returned
    # Everything above the diagonal have value of 1
    # so everything above the diagonal will return 1
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)

    # Everything above diagonal != 0, so it will return 0, which is the expectation
    return mask == 0


