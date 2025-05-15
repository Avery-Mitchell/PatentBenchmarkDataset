# Patent Misinformation Dataset and Model

This project is for CS 5001 - Natual Language Processing

### Contributors: Avery Mitchell

## Overview

This repository is a small dataset designed to fact check information about patents.

## Dataset Description

The dataset used in this project comes from articles taken from Snopes about patents. Each instance in the dataset consists of a claim about, or related to, a patent, Snopes's rating (True, False, Mixture, or Not Enough Information), metadata about each article, and the entirety of the Snopes article. 

## Dataset Usage

This dataset is useful for students, researchers, scientists, and parties interested in patents. To use the model to verify misinformation, you can provide the model with a claim about a patent and supporting evidence. The model will return a label: True, False, Mixture, and Not Enough Information.

Here's how you use it:

1. **Clone This Repository**: Clone this repository to your local machine or via the [Hugging Face Hub](https://huggingface.co/Avery90/Patent-Misinformation) and enter the directory

2. **Ensure All Dependencies Are Present**: Install all the dependencies in [requirements.txt](https://github.com/Avery-Mitchell/PatentBenchmarkDataset/blob/main/requirements.txt)

```sh
pip install -r requirements.txt
```

3. **API Key**: Set the HUGGINGFACE_API_KEY in [api_key.env](https://github.com/Avery-Mitchell/PatentBenchmarkDataset/blob/main/api_key.env) to your own personal API key. This is necessary if you are trying to use Hugging Face's Inference tools.

```sh
HUGGINGFACE_API_KEY = insert_your_key_here
```

4. **Try The Model**: Now you can experiment with the model

```sh
python .\fine_tuned.py --claim "insert claim here" --evidence "insert evidence for claim here"
```

This model works best when you provide the model with as much context (evidence) as you can. This fine-tuned model is built ontop of Mistral AI's Mistral-7B-Instruct-v0.2. Although this model does not necessarily produce the strongest results, it's much smaller in size (when compared to models like Llama and ChatGPT) and can be easily used locally (and in the cloud). It balances size with performance, relying on strong context for good results. 

## Discussion and Improvements

Although I am happy with this project as it currently stands, there is definitely room for improvement. The actual patent misinformation dataset is relatively sparse, in the future I would like to increase the number of instances in the dataset from other sources (not just Snopes). I also wish that I had the time and resources to try larger models (trying to fine-tune a model on a consumer-grade laptop is not the easiest). Given the time-constraints I had this semester, as well as the fact that my other group member dropped out and did not help much, I think the project turned out good. This was my first experience working with a language model like this.

## Acknowledgements 

Thank you Dr. Maity for your guidance, patience, and leniency with this project - I greatly appreciate it. 


