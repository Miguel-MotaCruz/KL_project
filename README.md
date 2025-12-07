Sure! Here’s the full README.md formatted and ready to paste:

# Knowledge and Language Project 2025/2026

**Course:** Knowledge and Language, Masters in Informatics Engineering  
**University:** Department of Informatics Engineering - University of Coimbra   
**Professors:** Hugo Gonçalo Oliveira (hroliv@dei.uc.pt)  
**Assistant:** Bruno Ferreira (brunof@dei.uc.pt)  
**Authors:** Miguel Mota Cruz & José Branquinho Macedo

---

## Project Overview

This project studies **gender-occupation bias in Large Language Models (LLMs)** and investigates how injecting external knowledge from Wikidata affects fairness. We examine two knowledge-injection strategies:

- **Structured triples:** exposing the model to relational data directly.  
- **Natural-language statements:** converting triples into training sentences.

For each strategy, two datasets are created:

1. **Imbalanced:** reflecting real-world gender distributions across occupations.  
2. **Balanced:** equalizing male- and female-associated examples for each occupation.

Using **Flan-T5** as the base model, we fine-tune variants under each condition and evaluate fairness with **WinoBias** through pro- and anti-stereotypical F1 scores. The project also tracks overall performance, enabling analysis of trade-offs between fairness and predictive quality.

This repository contains:

- Data processing scripts and training pipelines  
- Template-based knowledge-injection utilities  
- Fine-tuning configurations and evaluation scripts  

> Large models and checkpoints are excluded from Git and managed separately.
