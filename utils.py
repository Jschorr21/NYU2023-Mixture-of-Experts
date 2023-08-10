#!/usr/bin/env python
# coding: utf-8
""" LLM Evaluator on MMLU Benchmark

This script contails utility functions used by model_eval.py and mixture_model.py


Author: 
    Jake Schorr
"""

import torch
import re
import json
import transformers
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
import argparse
from torch.nn.utils.rnn import pad_sequence



subsets = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']



def map_to_ABCD(data):
    """
    Maps the 'answer' column of the dataset from [1, 2, 3, 4] to [A, B, C, D] respectively

    Parameters
    ----------
    data : HF dataset
        Hugging Face dataset
    
    Returns
    ----------
    data
        data object with updated 'answer' column
    """
    choices = ['A', 'B', 'C', 'D']
    data = data.map(lambda example: {'answer_mapped': choices[example['answer']]})
    
    # remove the original 'answer' column and rename 'answer_mapped' to 'answer'
    data = data.remove_columns(['answer'])
    data = data.rename_column('answer_mapped', 'answer')
    return data




def get_dev_and_test_data(data_path, subset):
    """
    Loads in the subset of the desired dataset from Hugging Face

    Parameters
    ----------
    data_path : str
        path to Hugging Face dataset
    subset : str
        desired subset of the dataset
    
    Returns
    ----------
    test_data
        HF dataset object representing 'test' split
    dev_data
        HF dataset object representing 'dev' split

    """
    test_data = load_dataset(data_path, subset, split="test")
    dev_data = load_dataset(data_path, subset, split="dev")
    test_data = map_to_ABCD(test_data)
    dev_data = map_to_ABCD(dev_data)
    return test_data, dev_data


def gen_examples(dev_data):
    """
    Generates question-answer pair examples to be included in the prompt for the model

    Parameters
    ----------
    dev_data : HF Dataset
        'dev' split from HF Dataset
    
    Returns
    ----------
    prompt
        beginning of prompt string containing 5 question-answer pair examples
    """
    subset = dev_data.config_name.replace("_", " ")
    prompt = f"The following are multiple choice questions (with answers) about {subset}.\n\n"
    count = 0
    for row in dev_data:
        count +=1
        if subset == "high school european history" or subset == "high school us history":
            if count >= 4:
                break
        prompt += row['question']
        options = f" \nA. {row['choices'][0]}\nB. {row['choices'][1]}\nC. {row['choices'][2]}\nD. {row['choices'][3]}\n"
        prompt += options
        prompt += "Answer: "
        
        prompt += str(row['answer']) 
        prompt += "\n\n"
    return prompt
    

def gen_prompt(test_data, example_prompt, index):
    """
    Generates the prompt to feed into the model

    Parameters
    ----------
    test_data : HF Dataset
        'test' split from HF Dataset
    example_prompt : str
        5 examples generated from gen_examples()
    index : int
        'test' row number (Question #) 
    
    Returns
    ----------
    prompt
        prompt string containing 5 question-answer pair examples and the current question with no answer
    """
    prompt = example_prompt
    prompt +=test_data[index]['question']
    options = f" \nA. {test_data[index]['choices'][0]}\nB. {test_data[index]['choices'][1]}\nC. {test_data[index]['choices'][2]}\nD. {test_data[index]['choices'][3]}\n"
    prompt += options
    prompt += "Answer:"
    return prompt


def clean_pred(pred):
    """
    Cleans the prediction and removes any non-alphanumeric characters

    Parameters
    ----------
    pred : str
        unncleaned prediction
    
    Returns
    ----------
    pred
        cleaned prediction
    """
    pred = re.sub(r'[^a-zA-Z]', '', pred)
    return pred

def evaluate(preds_dict, gold_dict):
    """
    This function performs the evaluation on the results that the mixture model gave in
    mixture_model(). It computes the accuracy metric for each subset within the dataset,
    as well as an accuracy metric for the overall performence on all subsets

    Parameters
    ----------
    preds_dict
        dictionary containing answers provided by the mixture model
    gold_dict
        dictionary containing the correct answers from Hugging Face
    
    Returns
    ----------
    accuracy_by_subset
        dictionary containing the accuracy metric for each subset
    total_acc
        dictionary containing the accuracy metric for the models performence on all subsets
    """
    total_count = 0
    total_length = 0
    accuracy_by_subset = {}
    for subset in subsets:
        count = 0
        length = len(preds_dict[subset])
        total_length += length
        for i in range(length):
            if preds_dict[subset][i] == gold_dict[subset][i]:
                count +=1
        total_count+=count
        acc = round((count/len(gold_dict[subset]))*100, 2)
        accuracy_by_subset[subset] = f"{acc}%"
        print(f"Accuracy for {subset}: {acc}% ---- \n")
    total_acc = {}
    total_acc['Total Accuracy for Model'] = f"{round((total_count/total_length)*100, 2)}%"
    return accuracy_by_subset, total_acc

def write_results(results_path, preds_dict, gold_dict):
    """
    This function performs writes the results of the mixture model evaluation to an output
    file.

    Parameters
    ----------
    results_path
        path to results output file
    preds_dict
        dictionary containing answers provided by the mixture model
    gold_dict
        dictionary containing the correct answers from Hugging Face
    """
    accuracy_by_subset, total_acc = evaluate(preds_dict, gold_dict)
    with open(results_path, "w") as f:
        json.dump(preds_dict, f, indent = 2)
        json.dump(gold_dict, f, indent = 2)
        json.dump(accuracy_by_subset, f, indent = 2)
        json.dump(total_acc, f, indent = 2)


