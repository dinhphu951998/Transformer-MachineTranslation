from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset, tqdm
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_config, get_weights_file_path
from dataset import BilingualDataset
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter
import warnings

from dataset import causal_mask

def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_target, max_len, device):
    sos_idx = tokenizer_target.token_to_id("[SOS]")
    eos_idx = tokenizer_target.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(src, src_mask)

    # initialize the decoder intput with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(src).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mas for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:-1]) # ???

        # select token with max prob
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(src).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_target, max_len, device, print_msg, global_state, writer, num_examples = 2):
    model.eval()
    count = 0

    source_text = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad(): # disable gradient calculation
        for batch in validation_ds:
            count += 1
            encoding_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoding_input.size(0) == 1, "batch size must be 1 for validation"

            model_out = greedy_decode(model, encoding_input, encoder_mask, tokenizer_src, tokenizer_target, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = tokenizer_target.decode(model_out.detach().cpu().numpy())

            source_text.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # print message to the console and not intefer the progress bar
            print_msg('-'*console_width)
            print_msg(f'source: {source_text}')
            print_msg(f'target: {target_text}')
            print_msg(f'predicted: {model_out_text}')

            if count == num_examples:
                break

    if writer:
        # torch metrics char error rate, BLEU, word error rate
        pass


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):

    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # why min_freq = 2?
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('Helsinki-NLP/opus_books', f'{config['lang_src']}-{config['lang_tgt']}', split='train')

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_target, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_target, config['lang_src'], config['lang_tgt'], config['seq_len'])
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        target_ids = tokenizer_target.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(target_ids))

    print("Max length of source sentence", max_len_src)
    print("Max length of target sentence", max_len_tgt)

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_target


def get_model(config, vocab_src_len, vocab_target_len):
    model = build_transformer(vocab_src_len, vocab_target_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_target = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    # tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1) # 0.1% score of the highest prob and give it to other

    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            model.train()

            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len) hide only padding tokens
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len). hide subsequential tokens

            # run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # b, seq_len, d_model
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # b, seq_len, d_model
            projection_output = model.project(decoder_output) # b, seq_eng, target_vocab_size

            label = batch['label'].to(device) # b, seq_len

            # b, seq_len, target_vocab_size -> b, seq_len, target_vocab_size
            loss = loss_fn(projection_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1)) # b, seq-len, target_vocab_size
            batch_iterator.set_postfix({
                "loss": f"{loss.item():6.3f}"
            })

            # log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            run_validation(model, val_dataloader, tokenizer_src, tokenizer_target, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            global_step+=1

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)


if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

