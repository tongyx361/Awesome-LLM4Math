# Awesome LLM4Math

> Curation of resources for LLM mathematical reasoning.

[![Awesome](https://awesome.re/badge.svg)](https://github.com/tongyx361/Awesome-LLM4Math)
[![License: Apache](https://img.shields.io/badge/License-Apache-green.svg)](https://opensource.org/licenses/Apache)

| [🐱 GitHub](https://github.com/tongyx361/Awesome-LLM4Math) | [🐦 Twitter](https://twitter.com/tongyx361/status/1780980512561185062) |

📢 If you have any suggestions, please don't hesitate to let us know. You can

- directly [E-mail *Yuxuan Tong*](tongyuxuan361@gmail.com),
- comment under the [*Twitter* thread](https://twitter.com/tongyx361/status/1780980512561185062),
- or post an issue in the [*GitHub* repository](https://github.com/tongyx361/Awesome-LLM4Math).

The following resources are listed in **(roughly) chronological order of publication**.

## Continual Pre-Training: Methods / Models / Corpora

- [*Llemma*](https://blog.eleuther.ai/llemma/) & [*Proof-Pile-2*](https://www.eleuther.ai/artifacts/proof-pile-2): **Open-sourced** re-implementation of ***Minerva***.
  - Open-sourced corpus *Proof-Pile-2* comprising **51.9B tokens** (by *DeepSeek* tokenizer).
  - Continually pre-trained based on ***CodeLLaMA*s**.
- [*OpenWebMath*](https://arxiv.org/abs/2310.06786):
  - **13.6B tokens** (by *DeepSeek* tokenizer).
  - Used by *Rho-1* to achieve performance comparable with *DeepSeekMath*.
- [*MathPile*](http://arxiv.org/abs/2312.17120):
  - **8.9B tokens** (by *DeepSeek* tokenizer).
  - Mainly comprising **arXiv papers**.
  - Shown **not effective (on 7B models)** by *DeepSeekMath*.
- [*DeepSeekMath*](http://arxiv.org/abs/2402.03300): Open-sourced **SotA** (as of 2024-04-18).
  - Continually pre-trained based on *DeepSeek-LLM*s and *DeepSeekCoder-7B*
- [*Rho-1*](https://arxiv.org/abs/2404.07965): Selecting tokens based on **loss/perplexity**, achieving performance **comparable with *DeepSeekMath*** but only based on 15B *OpenWebMath* corpus.

## SFT: Methods / Models / Datasets

### Natural language (only)

- [*RFT*](http://arxiv.org/abs/2308.01825): SFT on **rejection-sampled** model outputs is effective.
- [*MetaMath*](https://meta-math.github.io/): Constructing problems of **ground truth answer (but no necessarily feasible)** by **self-verification**.
  - Augmenting with **GPT-3.5-Turbo**.
- [*AugGSM8k*](http://arxiv.org/abs/2310.05506) : Common data augmentation on **GSM8k** **helps little** in generalization to **MATH**.
- [*MathScale*](http://arxiv.org/abs/2403.02884): **Scaling** synthetic data to **~2M samples** using **GPT-3.5-Turbo** with **knowledge graph**.
- [*KPMath*](https://arxiv.org/abs/2403.02333): **Scaling** synthetic data to **1.576M samples** using **GPT-4-Turbo** with **knowledge graph**.
- [*XWin-Math*](https://arxiv.org/abs/2403.04706): **Simple scaling** synthetic data to **480k MATH + 960k GSM8k samples** using **GPT-4-Turbo** with **knowledge graph**.

### Code integration

- [*MAmmoTH*](http://arxiv.org/abs/2309.05653): **SFT on CoT&PoT-mixing data** is effective.
- [*ToRA*](https://microsoft.github.io/ToRA/) & [*MARIO*](https://github.com/MARIO-Math-Reasoning/MARIO): The fisrt open-sourced model works to verify the effectiveness of **SFT for tool-integrated reasoning**.
- [*OpenMathInstruct-1*](http://arxiv.org/abs/2402.10176): **Scaling synthetic data to 1.8M** using **Mixtral-8x7B**

## RL: Methods / Models / Datasets

- [*Math-Shepherd*](http://arxiv.org/abs/2312.08935): Consturcting **step-correctness labels** based on an **MCTS**-like method.

## Evaluation: Benchmarks

Here we focus on **several the most important benchmarks**.

### [OpenAI `simple-evals` - Math](https://github.com/openai/simple-evals)

> - *MMLU(-Math)*: Measuring Massive Multitask Language Understanding, reference: https://arxiv.org/abs/2009.03300, https://github.com/hendrycks/test, MIT License
> - *MATH*: Measuring Mathematical Problem Solving With the MATH Dataset, reference: https://arxiv.org/abs/2103.03874, https://github.com/hendrycks/math, MIT License
> - *MGSM*: Multilingual Grade School Math Benchmark (MGSM), Language Models are Multilingual Chain-of-Thought Reasoners, reference: https://arxiv.org/abs/2210.03057, https://github.com/google-research/url-nlp, Creative Commons Attribution 4.0 International Public License (CC-BY)

### Other benchmarks

- [*miniF2F*](https://arxiv.org/abs/2109.00110): “a **formal** mathematics benchmark (translated across multiple formal systems) consisting of exercise statements **from olympiads (AMC, AIME, IMO) as well as high-school and undergraduate maths classes**”.
- [*OlympiadBench*](https://arxiv.org/abs/2402.14008): “an **Olympiad-level** bilingual **multimodal** scientific benchmark”.
  - **GPT-4V** attains an average score of **17.23% on OlympiadBench**, with a mere **11.28% in physics**.

## Curations, collections and surveys

- [GitHub - lupantech/dl4math: Resources of deep learning for mathematical reasoning (DL4MATH).](https://github.com/lupantech/dl4math)

## Events

- [*AIMO*](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize): “a new **$10mn** prize fund to spur the **open** development of AI models capable of performing as well as top human participants in the International Mathematical Olympiad (IMO)”.

