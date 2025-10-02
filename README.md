# Link Prediction for Job Recommendations

Jupyter notebook which explores link prediction on a heterogeneous graph of candidates, jobs, and skills. The model is trained to predict candidate–job matches using embeddings and evaluated with metrics such as ROC-AUC and Average Precision.


- [Installation](#installation)
- [Dataset](#dataset)
  - [Candidates](#candidates)
  - [Jobs](#jobs)
  - [Skills](#skills)
- [Modeling Approach](#modeling-approach)
  - [Node Features](#node-features)
  - [Graph Construction](#graph-construction)
  - [Learning Objective](#learning-objective)
  - [Training Strategy](#training-strategy)
- [Evaluation](#evaluation)
- [Results](#results)
  - [Test Performance](#test-performance)
  - [Inference & Recommendation](#inference--recommendation)
  - [Example Output](#example-output)
- [Future Work](#future-work)



# Instalation
```bash
git clone https://github.com/filipgjorgiev/job-recommendation.git
cd job-recommendation
pip install -r requirements.txt

# (Optional) create and activate a virtual environment
python -m venv .venv
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

The dataset used in this project represents a job recommendation scenario built around three main entities:

- **Candidates** – Candidate data was taken from a Kaggle dataset containing full resumes. For each candidate, I extracted structured information such as technical skills, soft skills, field of study, degree, and work experience using OpenAI API requests. After preprocessing, each candidate profile contains the following attributes:
  - Technical skills
  - Soft skills
  - Education and degree
  - Field of study
  - Total experience in months
  - Experience level

- **Jobs** – Job data was fetched from the public Remotive API. Each job includes fields such as
  `title`, `company_name`, `company_logo`, `category`, `tags`, `salary`, and `description`.
  We derive skill signals from two sources:

  1) **Tags-based skills** — Remotive’s `tags` are treated as lightweight skill hints.
  2) **Description-based skills** — We run a simple keyword-matching pass over `description`
     using a curated skill list to extract occurrences.

  After processing, the dataset includes column named extracted_skills which is a list of skills matched from the job description.

- **Skills** – Normalized skill terms extracted from both candidates and job descriptions.


## Modeling Approach

The problem is framed as a **link prediction task** on a heterogeneous graph consisting of three node types:  

- **Candidates**  
  Each candidate node is represented with multiple features that capture different aspects of their profile:
  - `cv_embedding` – semantic embedding of the candidate’s resume text, generated using [SentenceTransformers](https://www.sbert.net/)  
  - `cand_bow` – bag-of-words vector for extracted technical skills  
  - `cand_soft_bow` – bag-of-words vector for extracted soft skills  
  - `tenure` – total work experience expressed in months  

- **Jobs**  
  Each job posting is encoded with features derived from text and structured attributes:
  - `job_emb` – embedding of the job description text (also created with SentenceTransformers)  
  - `job_bow` – bag-of-words representation of the required skills  

- **Skills**  
  Each skill node is represented with:
  - `skill_embedding` – a dense embedding that captures semantic similarity between skills  

### Graph Construction (PyTorch Geometric)

We build a heterogeneous graph with three node types and four directed edge types using `torch_geometric.data.HeteroData`:

- **Nodes & features**
  - `candidate`: features come from a precomputed matrix `candidate_x` (concatenation of `cv_embedding`, `cand_bow`, `cand_soft_bow`, and `tenure`).
  - `job`: features come from `job_x` (concatenation of `job_emb` and `job_bow`).
  - `skill`: initialized with a 64-dim random vector per skill (placeholder embedding); these are **learned end-to-end** during training and can be replaced with text-derived embeddings later.

- **Edges**
  - `('candidate','has','skill')` and its reverse `('skill','had_by','candidate')`
  - `('job','requires','skill')` and its reverse `('skill','required_by','job')`

We add **reverse edges** explicitly to enable **bi-directional message passing** across relations. All node feature tensors are stored in `data[<node_type>].x`, and relation-specific edge indices are placed in `data[(src, rel, dst)].edge_index`. This setup allows a relation-aware GNN (HeteroConv + SAGEConv) to aggregate neighborhood information per edge type and learn task-relevant representations for **link prediction** between candidates and jobs.

### Learning Objective

The model learns embeddings for candidates, jobs, and skills such that:
- Candidate nodes are closer to jobs they are a good match for.  
- Skill nodes act as a bridge between candidates and jobs, improving representation quality.  

We train the model as a **binary link prediction task**, where candidate–job pairs are classified as either positive (match) or negative (no match). Positive pairs are derived from weak supervision (top-K similarity between CV and job description), while negative pairs are sampled randomly or selected as **hard negatives** (jobs that look similar but are not actual matches).  

---

### Training Strategy
- **Prior similarity signals**: Before GNN training, we compute prior similarity between candidates and jobs by combining:
  - Semantic similarity between `cv_embedding` and `job_emb` (cosine similarity).
  - Skill overlap between `cand_bow` and `job_bow`.  
  These priors are weighted (0.6 text + 0.4 skills) and used as additional bias when scoring links.

- **Loss function**: We use **Bayesian Personalized Ranking (BPR) loss**, which encourages the model to assign higher scores to positive pairs than to sampled negatives.

- **Negative sampling**:  
  - *Random negatives* are used initially to provide broad contrast.  
  - *Hard negatives* are later mined based on current model predictions — selecting job nodes that are highly ranked but not true matches — to make training more challenging and effective.

---

### Evaluation
Candidate–job link predictions are evaluated using:
- **ROC-AUC** – measures how well the model separates positive and negative pairs.  
- **Average Precision (AP)** – summarizes performance over the precision–recall curve.  

Training runs for 120 epochs, with metrics reported every 10 epochs.

## Results

The model was evaluated on held-out candidate–job pairs. Performance metrics:

- **ROC-AUC**: 0.880  
- **Average Precision (AP)**: 0.850  

These results indicate that the model is able to effectively distinguish true candidate–job links from negatives, and produces high-quality rankings in a job recommendation setting.

## Inference & Recommendation

After training, the model can be used to recommend jobs for **new candidates**. The workflow is:

1. **Extract candidate profile**  
   - Parse CV files (`.pdf` or `.docx`) into structured data (technical skills, soft skills, education, tenure).  
   - Generate a semantic embedding of the resume text using `SentenceTransformers`.  
   - Convert skills into bag-of-words vectors and scale tenure information.  
   - Build a unified feature row aligned with the training feature space.

2. **Integrate candidate into the graph**  
   - Append the candidate node and connect it to known skills.  
   - Run the trained GNN to compute embeddings for the new candidate and all jobs.

3. **Score candidate–job pairs**  
   - Compute cosine similarity between candidate and job embeddings.  
   - Scale scores into logits and pass them through a sigmoid for probabilities.  
   - Apply a calibrated logistic regression layer to improve probability estimates.

4. **Rank and interpret results**  
   - Jobs are ranked by predicted match probability.  
   - Each recommendation is labeled as:
     - **Strong match** (≥ 75%)  
     - **Moderate match** (60–74%)  
     - **Weak match** (< 60%)  

This final step transforms the learned embeddings into **actionable job recommendations** that can be directly interpreted by recruiters or candidates.


