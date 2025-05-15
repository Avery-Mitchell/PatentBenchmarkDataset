# Patent Misinformation Dataset and Model

This project is for CS 5001 - Natual Language Processing

### Contributors: Avery Mitchell

## Overview

This project introduces a small, fine-tuned dataset and model aimed at fact-checking claims related to patents. It leverages real-world misinformation examples, focusing on patent-related claims, and provides a classification model that labels them as True, False, Mixture, or Not Enough Information.

## Dataset Description

The dataset used in this project comes from articles taken from Snopes about patents. Each instance in the dataset consists:
 - A claim about a patent
 - Snopes's truth rating (True, False, Mixture, Not Enough Information)
 - Relevant Metadata
 - The full article text

## Dataset Usage

This dataset is useful for students, researchers, scientists, and parties interested in patents. To use the model to verify misinformation, you can provide the model with a claim about a patent and supporting evidence. The model will return a label: True, False, Mixture, and Not Enough Information.

Here's how you use it:

1. **Clone This Repository**: 

```sh
git clone https://github.com/Avery-Mitchell/PatentBenchmarkDataset.git
cd PatentBenchmarkDataset
```

or access it directly via the [Hugging Face Hub](https://huggingface.co/Avery90/Patent-Misinformation) 

2. **Install Dependencies**: Make sure all required packages are installed

```sh
pip install -r requirements.txt
```

3. **Set Up API Key (Optional)**: If using Hugging Face's Inference API, insert your Hugging Face API key in [api_key.env](https://github.com/Avery-Mitchell/PatentBenchmarkDataset/blob/main/api_key.env)

```sh
HUGGINGFACE_API_KEY = insert_your_key_here
```

4. **Run The Model**: Run the model by providing a claim and supporting evidence

```sh
python .\fine_tuned.py --claim "insert claim here" --evidence "insert evidence for claim here"
```

This model works best when you provide the model with as much context (evidence) as you can. This fine-tuned model is built ontop of Mistral AI's Mistral-7B-Instruct-v0.2. Although this model does not necessarily produce the strongest results, it's much smaller in size (when compared to models like Llama and ChatGPT) and can be easily used locally (and in the cloud). It balances size with performance, relying on strong context for good results. 

## Model Details

 - **Base Model**: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
 - **Why Mistral?**
 It's a lightweight, effiecient model that can be run locally (or in the cloud). While it does not offer the same performance as larger models such as GPT and LlaMA, it has a good balance of performance and resource efficiency.

## Limitations and Future Work

While this project has a solid foundation, there is room for improvement
 - **Dataset Size**: The current dataset is small and only from Snopes. I want to add more instances from a broader range of sources.
 - **Model Performance**: More powerful models could produce better results, but limited resources (money and hardware) restricted experimentation.
 - **Collaboration Challenges**: This project started as a group effort. Unfortunately, my teammate dropped out midway, so all the work was completed solo.

Despite these issues, this project was an extremely valuable learning experience and an excellent introduction to building language models for real-world tasks. This was my first time completing a project like this and I learned a ton. 

## Acknowledgements 

Thank you Dr. Maity for your guidance, patience, and leniency with this project - I greatly appreciate it. 

## Contact

For questions about the project, feel free to reach out via my email or GitHub.


