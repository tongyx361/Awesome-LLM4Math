# Awesome LLM4Math

> Curation of resources for **LLM mathematical reasoning**, most of which are **screened** by @tongyx361 to ensure **high quality** and accompanied with **elaborately-written concise descriptions** to help readers get the gist as quickly as possible.

[![Awesome](https://awesome.re/badge.svg)](https://github.com/tongyx361/Awesome-LLM4Math)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[🐱 GitHub](https://github.com/tongyx361/Awesome-LLM4Math) | [🐦 Twitter](https://twitter.com/tongyx361/status/1780980512561185062)

Featured by:

- Probably the **most comprehensive** list of **training prompt&answer datasets** for complex mathematical QA tasks across the web.
- Lists of the series of work implementing the current **open-source SotA** for mathematical problem-solving tasks.

You can **subscribe to our updates** in the following ways:

- **Follow** the [**X (Twitter) account** @tongyx361](https://x.com/tongyx361),
- **Follow** the [**Zhihu(知乎) account** @天欲雪](https://www.zhihu.com/people/bai-li-tian-he-84),
- **Watch releases in this GitHub repository**: upper right corner→Watch->Custom->Releases.

The following resources are listed in **(roughly) chronological order of publication**.

## (Training) Datasets

The following list mainly focuses (training) datasets for **complex mathematical QA**, leaving out most basic arithmetic datasets like AQuA-RAT, NumGLUE, MathQA, ASDiv, etc. For more other datasets, you can refer to the tables at the end of [A Survey of Deep Learning for Mathematical Reasoning](https://arxiv.org/abs/2212.10535).

- **[MATH](https://github.com/hendrycks/math)/Train**
  - **# of Queries**: 7,500
  - **Query Description**:
    - "The MATH dataset consists of problems from **mathematics competitions** including the **AMC 10, AMC 12, AIME**, and more."
    - "To provide a rough but informative comparison to human-level performance, we randomly sampled 20 problems from the MATH test set and gave them to humans. We artificially require that the participants have 1 hour to work on the problems and must perform calculations by hand. All participants are university students.
      - One participant who **does not like mathematics** got **8/20 = 40%** correct.
      - A participant **ambivalent toward mathematics** got **13/20**.
      - Two participants who **like mathematics** got **14/20 and 15/20**.
      - A participant who **got a perfect score on the AMC 10 exam and attended USAMO several times** got **18/20**.
      - A **three-time IMO gold medalist** got **18/20 = 90%**, though missed questions were exclusively due to small errors of arithmetic."
    - "Problems span various subjects and difficulties. The seven subjects are **Prealgebra, Algebra, Number Theory, Counting and Probability, Geometry, Intermediate Algebra, and Precalculus**."
    - "While **subjects like Prealgebra are generally easier than Precalculus**, within a subject problems can take on different difficulty levels. We encode a problem’s difficulty level from ‘1’ to ‘5,’ following AoPS. A subject’s easiest problems for humans are assigned a difficulty level of ‘1,’ and a subject’s hardest problems are assigned a difficulty level of ‘5.’ **Concretely, the first few problems of an AMC 8 exam are often level 1, while AIME problems are level 5.**"
  - **Answer Description**:
    - Solutions and final answers are by **experts from the AOPS community**.
    - "Problems and solutions are **consistently formatted** using **LaTeX** and the **Asymptote** vector graphics language."
- **[AMPS](https://github.com/hendrycks/math)/Khan-Academy**
  - **# of Queries**: 103,059
  - **Query Description**:
    - "The Khan Academy subset of AMPS has **693 exercise types** with over 100,000 problems and full solutions.
    - Problem types range **from elementary mathematics (e.g. addition) to multivariable calculus (e.g. Stokes’ theorem)**, and are used to teach actual **K-12** students.
    - The exercises can be regenerated using **code from github.com/Khan/khan-exercises/**."
  - **Answer Description**: Same as query.
- **[AMPS](https://github.com/hendrycks/math)/Mathematica**
  - **# of Queries**: 4,830,500
  - **Query Description**:
    - "With **Mathematica**, we designed **100 scripts** that test distinct mathematics concepts, 37 of which include full step-by-step LaTeX solutions in addition to final answers. We generated **around 50,000 exercises from each of our scripts**, or around 5 million problems in total."
    - "Problems include various aspects of **algebra, calculus, counting and statistics, geometry, linear algebra, and number theory**."
  - **Answer Description**: Same as query.
- **[GSM8K](https://github.com/openai/grade-school-math)/Train**
  - **# of Queries**: 7,473
  - **Query Description**:
    - "GSM8K consists of 8.5K high quality **grade school math problems** created by **human problem writers**."
    - These problems take **between 2 and 8 steps** to solve, and solutions primarily involve performing a sequence of **elementary calculations using basic arithmetic operations (+ − ×÷)** to reach the final answer. A bright **middle school** student should be able to solve every problem.
    - "We initially collected a starting set of a thousand problems and natural language solutions by hiring freelance contractors on **Upwork (upwork.com)**. We then worked with **Surge AI (surgehq.ai)**, an NLP data labeling platform, to scale up our data collection."
  - **Answer Description**:
    - Written by labelers from **Surge AI (main) and Upwork (auxilliary)**
    - "After collecting the full dataset, we asked workers to **re-solve all problems**, with no workers re-solving problems they originally wrote. We checked whether their final answers agreed with the original solutions, and any problems that produced disagreements were either repaired or discarded. We then performed **another round of agreement checks on a smaller subset of problems**, finding that 1.7% of problems still produce disagreements among contractors. We estimate this to be the fraction of problems that contain breaking errors or ambiguities. It is possible that a larger percentage of problems contain subtle errors."
- **[TAL-SCQ5K](https://huggingface.co/datasets/math-eval/TAL-SCQ5K)-EN/Train**
  - **# of Queries**: 3,000
  - **Query Description**:
    - "TAL-SCQ5K-EN/TAL-SCQ5K-CN are high quality **mathematical competition** datasets in English and Chinese language created by TAL Education Group, each consisting of 5K questions(3K training and 2K testing).
    - The questions are in the form of **multiple-choice**
    - and cover mathematical topics at the **primary,junior high and high school** levels."
  - **Answer Description**: Same as query.
- **[TAL-SCQ5K](https://huggingface.co/datasets/math-eval/TAL-SCQ5K)-CN/Train**
  - **# of Queries**: 3,000
  - **Query Description**: Similar to TAL-SCQ5K-EN/Train.
  - **Answer Description**: Same as query.
- **[CAMEL-Math](https://huggingface.co/datasets/camel-ai/math)**
  - **# of Queries**: 50,000
  - **Query Description**:
    - "Math dataset is composed of 50K problem-solution pairs obtained using **GPT-4**.
    - The dataset problem-solution pairs generating from **25 math topics, 25 subtopics for each topic and 80 problems for each "topic,subtopic" pairs.**"
  - **Answer Description**: Same as query.
- **[MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)-GSM8K/SV**
  - **# of Queries**: 29,283
  - **Query Description**:
    - "In **Self-Verification (SV)**, **the question with the answer is first rewritten into a declarative statement**, e.g., "How much did he pay?" (with the answer 110) is rewritten into "He paid $10". Then, a question for asking the value of $x$ is appended, e.g., "What is the value of unknown variable $x$?"."
  - **Answer Description**: Same as query.
- **[MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)-GSM8K/FOBAR**
  - **# of Queries**: 16,503
  - **Query Description**:
    - "**FOBAR** proposes to **directly append the answer to the question**, i.e., "If we know the answer to the above question is $a_{i}^*$ , what is the value of unknown variable $x$?""
  - **Answer Description**: Same as query.
- **[MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)-MATH/SV**
  - **# of Queries**: 4,596
  - **Query Description**: Similar to MetaMath-GSM8K/SV.
  - **Answer Description**: Same as query.
- **[MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)-MATH/FOBAR**
  - **# of Queries**: 4,911
  - **Query Description**: Similar to MetaMath-GSM8K/FOBAR.
  - **Answer Description**: Same as query.
- **[MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)/College-Math**
  - **# of Queries**: 1,840
  - **Query Description**:
    - "We use **GPT-4** to …
    - create question-CoT pairs through **Self-Instruct** (Wang et al., 2023h),
    - utilizing **a few seed exemplars found online**."
  - **Answer Description**: Same as query.
- **[MMIQC](https://huggingface.co/datasets/Vivacem/MMIQC)/AugSimilar**
  - **# of Queries**: (unspecified)
  - **Query Description**:
    - "In our practice, we find that GPT tends to generate several almost the same problems regardless of the given seed problem when asked to generate up to 10 new problems. Thus, we only **ask GPT to generate 3 problems (with a solution, for rejection sampling)** each time".
    - Considering rejection sampling needs the answer to the problem better to be correct, we use the stronger **GPT-4(-1106)** instead of GPT-3.5.
    - To control the cost, **our prompt emphasizes that the solution should be as brief as possible**.
    - "We use a **temperature $T = 1.0$** for both procedures."
  - **Answer Description**: Same as query.
- **[MMIQC](https://huggingface.co/datasets/Vivacem/MMIQC)/IQC**
  - **# of Queries**: (unspecified)
  - **Query Description**:
    - "Our approach, termed **IQC (Iterative Question Composing)**, deviates from this by iteratively constructing more complex problems. It **augments the initial problems, adding additional reasoning steps without altering their intrinsic logical structure.**"
    - "We perform Iterative Question Composing for **4 iterations**".
    - "We note that although **some of the questions are not rigorously a sub-problem or sub-step of the corresponding problem in the previous iteration as required in our prompt**, they are still valid questions that can increase the diversity of the dataset."
    - "Specifically, we use **GPT-4(-1106)** for question composing model $\pi_{q}$ with a **$T = 0.7$ temperature**."
  - **Answer Description**: Same as query.
- **[MMIQC](https://huggingface.co/datasets/Vivacem/MMIQC)/MathStackExchange**
  - **# of Queries**: 1,203,620
  - **Query Description**:
    - "we extract the data collected from **Mathematics Stack Exchange in RedPajama (Computer, 2023)** and pre-process it into question-response pairs.
    - For each Mathematics Stack Exchange page, we only retain the **answer ranked first by RedPajama**.
    - Then we **filter out the answer that does not contain a formula environment symbol ‘$’**."
  - **Answer Description**: **No** short final answers.
- **[MWPBench](https://github.com/microsoft/unilm/tree/master/mathscale/MWPBench)/CollegeMath/Train**
  - **# of Queries**: 1,281
  - **Query Description**:
    - "We curated a collection of **nine college mathematics textbooks**, each addressing a distinct topic.
    - These textbooks encompass **seven** critical mathematical disciplines: **algebra, pre-calculus, calculus, vector calculus, probability, linear algebra, and differential equations**.
    - These textbooks are originally in PDF format and we convert them to text format using the **MathPix** API, where equations are transformed to LaTeX format. Once converted a textbook to text format, we are ready to extract exercises and their solutions. For each book, we **first manually segment the book into chapter and identify pages with exercises and their solutions**. Then we extract questions in exercises and their associated short answers."
  - **Answer Description**: Same as query.
- **[AOPS](https://github.com/hkust-nlp/dart-math/blob/main/data/dsets/aops.jsonl)**
  - **# of Queries**: 3,886
  - **Query Description**:
    - Problems from mathematical competitions of **AIME+AMC≤2023**, crawled from [AOPS](https://artofproblemsolving.com/wiki/index.php) by @yulonghui .
  - **Answer Description**: Same as query.
- **[WebInstruct(Sub)](https://tiger-ai-lab.github.io/MAmmoTH2/)**
  - **# of Queries**: ~10M(2,335,220)
  - **Query Description**:
    - "In this paper, we aim to **mine these instruction-response pairs from the web** using a three-step pipeline.
      1) **Recall** step: We create a diverse seed dataset by crawling several quiz websites. We use this seed data to train a **fastText** model (Joulin et al., 2016) and employ it to recall documents from **Common Crawl** (Computer, 2023). GPT-4 is used to trim down the recalled documents by their root URL. We obtain 18M documents through this step.
      2) **Extract** step: We utilize open-source LLMs like **Qwen-72B** (Bai et al., 2023) to extract Q-A pairs from these documents, producing roughly 5M candidate Q-A pairs.
      3) **Refine** step: After extraction, we further employ **Mixtral-8×22B** (Jiang et al., 2024) and **Qwen-72B** (Bai et al., 2023) to refine (Zheng et al., 2024b) these candidate Q-A pairs. This refinement operation aims to **remove unrelated content, fix formality, and add missing explanations to the candidate Q-A pairs**. Eventually, we harvest a total of 10M instruction-response pairs through these steps."
    - "The pie chart reveals that WebInstruct is predominantly composed of science-related subjects, with **81.69%** of the data falling under the broad **"Science"** category.
      - Within this category, **Mathematics** takes up **the largest share at 68.36%**,
      - followed by **Physics, Chemistry, and Biology**.
    - The remaining **non-science categories, such as Business, Art & Design, and Health & Medicine**, contribute to the diversity of the dataset."
    - "In terms of **data sources**, the **vast majority (86.73%)** of the instruction-response pairs come from **exam-style questions**, while **forum discussions** make up the remaining **13.27%**."
    - "To quantify the error percentages, we randomly sample 50 refined QA examples and ask the human annotators to compare whether the refined examples are correct and significantly better than the extracted ones in terms of format and intermediate solutions.
      - As we can see from Figure 6, **78%** examples have been **improved** after refinement
      - and **only 10%** examples introduce **hallucinations** after refinement."
  - **Answer Description**: Same as query.

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
- [*AlphaMath*](https://arxiv.org/abs/2405.03553): Use **MCTS** to synthesize tool-integrated reasoning paths and **step-level reward labels**, then train the model with **a multi-task language model and reward model loss** to get a policy-and-value model.
  - Compared with *DeepSeekMath-7B-**RL*** **(58.8% pass@1) on *MATH***, *AlphaMath* catches up by merely SFT *DeepSeekMath-7B* with *MARIO* and *AlphaMath* and further improves to **68.6%** with ***Step-level Beam Search (SBS)*** decoding. (Table 4)

## RL: Methods / Models / Datasets

- [*Math-Shepherd*](http://arxiv.org/abs/2312.08935): Consturcting **step-correctness labels** based on an **MCTS**-like method.

## Prompting & Decoding: Methods

- [*DUP*](https://arxiv.org/abs/2404.14963): Prompting the model with three-stage *Deeply Understand the Problem* prompts, which comprises **1) core question extraction 2) problem-solving information extraction and 3) *CoT* reasoning**, improving more than *Plan-and-Solve* and *Least-to-Most* prompting on simple arithmetic, commonsense and symbolic reasoning tasks.

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

## Contribution Guidelines

If you have any suggestions, please don't hesitate to let us know. You can

- directly [E-mail *Yuxuan Tong*](tongyuxuan361@gmail.com),
- reply to the [X(Twitter) thread](https://twitter.com/tongyx361/status/1780980512561185062),
- or post an issue in the [GitHub repository](https://github.com/tongyx361/Awesome-LLM4Math).
