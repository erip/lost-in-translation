#!/usr/bin/env python3


from pathlib import Path
from argparse import ArgumentParser

from transformers import pipeline, MBart50TokenizerFast, MBartForConditionalGeneration


def setup_argparse():
    parser = ArgumentParser()
    parser.add_argument("--in-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("-m", "--model", default="facebook/mbart-large-50-many-to-one-mmt")
    parser.add_argument("-s", "--src", default="es_XX", help="The mBART language code of the source language.")
    parser.add_argument("-t", "--tgt", default="en_XX", help="The mBART language of the target language")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="The batch size for inference")
    return parser

def batch(l, batch_size=16):
    for i in range(0, len(l), batch_size):
        yield l[i:i+batch_size]

def translate(model, tokenizer, src, tgt_code, batch_size):
    translations = []
    for b in batch(src, batch_size):
        model_inputs = tokenizer(b, padding=True, return_tensors="pt", max_length=512, truncation=True).to("cuda:0")

        generated_tokens = model.generate(
            **model_inputs,
           forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
           num_beams=5
        )
        translations.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    return translations

if __name__ == "__main__":
    args = setup_argparse().parse_args()

    model = MBartForConditionalGeneration.from_pretrained(args.model)
    tokenizer = MBart50TokenizerFast.from_pretrained(args.model)
    tokenizer.src_lang = args.src
    tokenizer.tgt_lang = args.tgt

    text = [line.strip() for line in open(args.in_file, encoding="utf-8")]
    translations = translate(model, tokenizer, text, args.tgt, args.batch_size)
    # For whatever reason the target token isn't getting stripped...
    with open(args.out_file, "w", encoding="utf-8") as f:
        for t in translations:
            print(t, file=f)
