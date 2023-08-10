#!/usr/bin/env python
# coding: utf-8
""" LLM Evaluator on MMLU Benchmark

This script loads a pre-trained Hugging Face Language model from its inputted path as well as datasets from each subset of MMLU.
The model is evaluated on the MMLU benchmark, which has multiple choice questions from 57 different subjects/topicsranging 
from high school mathematics to college genetics to professional law. For each question in the test set, a prompt string is generated 
containing 5 example questions and answers from the given subject's dev split along with the question for which the model is to determine 
the answer. The script will evaluate the accuracy of the HF model by comparing the generated answers to the actual answers from MMLU.
The generated answers, correct answers, and accuracy metrics by subset (and overall) will be saved to the inputted json file path.

Usage: 
    python Model_Eval.py --hf_model_path path/to/HF_model --output_path path/to/output_file

Parameters:
    --hf_model_path (str): The path to the directory containing the pre-trained Hugging Face model.
    --output_path (str): The path to the output JSON file where the results will be saved.

Author: 
    Jake Schorr
"""


import transformers, datasets
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import torch
import re
import json
import argparse
from utils import subsets, map_to_ABCD, get_dev_and_test_data, gen_examples, gen_prompt, clean_pred, evaluate, write_results


def load_model_and_tokenizer_and_pipe(model_path, device_idx):
    """
    Loads the model and tokenizer from Hugging Face corresponding to parameter 'model_path'
    Initializes pipeline using this model and tokenizer

    Parameters
    ----------
    model_path : str
        Hugging Face model path for desired model
    device_idx
        ID of desired device for pipeline 
    
    Returns
    ----------
    model
        loaded model from 'model_path'
    tokenizer
        loaded tokenizer from 'model_path'
    pipe
        initialized text-generation pipeline using 'model' and 'tokenizer'
    """

    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path)
    model = transformers.LlamaForCausalLM.from_pretrained(model_path)
    pipe = transformers.pipeline("text-generation", model = model, tokenizer = tokenizer, device=device_idx) 
    return model, tokenizer, pipe


def make_predictions(model_path, data_path = 'cais/mmlu', max_new_tokens = 1, device_idx=0):
    """
    This function performs the inference on the selected model and evaluation dataset. It calls the function 
    to load in the model and tokenizer, and loops through the subsets of the data set prompting the model to answer
    each question for each subset. The selected answer are recorded

    Parameters
    ----------
    model_path : str
        Hugging Face model path for desired model
    data_path : str
        Hugging Face path for desired evaluation dataset. Defaults to 'cais/mmlu'
    max_new_tokens
        number of new tokens to generate after prompting
    device_idx
        ID of desired device for pipeline 

    Returns
    ----------
    preds_dict
        dictionary containing answers provided by the model
    gold_dict
        dictionary containing the correct answers from Hugging Face
    """
    model, tokenizer, pipe = load_model_and_tokenizer_and_pipe(model_path, device_idx)
    preds_dict = {}
    gold_dict = {}
    for subset in subsets:
        print(f"Beginning subset: {subset} ----------------\n\n")
        test_data, dev_data = get_dev_and_test_data(data_path, subset)
        examples = gen_examples(dev_data)
        preds = []
        for i in range(len(test_data)): #will be len(data) instead of '5'
            prompt = ''
            prompt = gen_prompt(test_data, examples, i)
            pred = pipe(prompt, max_new_tokens = max_new_tokens, return_full_text=False) 
            pred_s = pred[0]['generated_text']
            pred_s = clean_pred(pred_s)
            preds.append(pred_s)
        preds_dict[subset] = preds
        gold_dict[subset] = test_data['answer']
    return preds_dict, gold_dict
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_path', type=str, help = "path to HF model", required=True)
    parser.add_argument('--output_path', type=str, help = "path to output file", required=True) 
    args = parser.parse_args()
    preds_dict, gold_dict = make_predictions(args.hf_model_path, 'cais/mmlu',1, 0) 
    write_results(args.output_path, preds_dict, gold_dict) 






