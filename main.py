import chromadb
import pandas as pd
from utils import run_experiment, load_config
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, PromptTemplate, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import LLMRerank
from tonic_validate import ValidateScorer, ValidateApi
from tonic_validate.metrics import RetrievalPrecisionMetric, AnswerSimilarityMetric
import os
### SETUP --------------------------------------------------------------------------------------------------------------
# Load the config file (.env vars)
load_config()

# tonic_validate_api_key = os.getenv("TONIC_VALIDATE_API_KEY")
# tonic_validate_project_key = os.getenv("TONIC_VALIDATE_PROJECT_KEY")
# tonic_validate_benchmark_key = os.getenv("TONIC_VALIDATE_BENCHMARK_KEY")
# validate_api = ValidateApi(tonic_validate_api_key)

# Service context
azure_api_key = os.getenv("AZURE_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_api_version = os.getenv("AZURE_API_VERSION")

llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name="gpt-4o",
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)


from llama_index.core import Settings
Settings.embed_model = embed_model

# VDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Traditional VDB
chroma_collection = chroma_client.get_collection("ai_arxiv_full")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# Sentence window VDB
chroma_collection_sentence_window = chroma_client.get_collection("ai_arxiv_sentence_window")
vector_store_sentence_window = ChromaVectorStore(chroma_collection=chroma_collection_sentence_window)
index_sentence_window = VectorStoreIndex.from_vector_store(vector_store=vector_store_sentence_window)

# Prompt template
with open("resources/text_qa_template.txt", 'r', encoding='utf-8') as file:
    text_qa_template_str = file.read()

text_qa_template = PromptTemplate(text_qa_template_str)

# Tonic Validate setup
benchmark = validate_api.get_benchmark(tonic_validate_benchmark_key)
scorer = ValidateScorer(metrics=[RetrievalPrecisionMetric(), AnswerSimilarityMetric()],
                        model_evaluator="gpt-3.5-turbo")

### Define query engines -------------------------------------------------------------------------------------------------
# Naive RAG
query_engine_naive = index.as_query_engine(llm = llm,
                                           text_qa_template=text_qa_template,
                                           similarity_top_k=3,
                                           embed_model=embed_model)

# HyDE
hyde = HyDEQueryTransform(include_original=True)
query_engine_hyde = TransformQueryEngine(query_engine_naive, hyde)

## LLM Rerank
llm_rerank = LLMRerank(choice_batch_size=10, top_n=3)
query_engine_llm_rerank = index.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[llm_rerank],
    embed_model=embed_model,
    llm=llm
)
# HyDE + LLM Rerank
query_engine_hyde_llm_rerank = TransformQueryEngine(query_engine_llm_rerank, hyde)

# Sentence window retrieval
query_engine_sentence_window = index_sentence_window.as_query_engine(text_qa_template=text_qa_template,
                                                                     similarity_top_k=3,
                                                                      embed_model=embed_model,
                                                                      llm=llm)

# Sentence window retrieval + LLM Rerank
query_engine_sentence_window_llm_rerank = index_sentence_window.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[llm_rerank],
    embed_model=embed_model,
    llm=llm)

## Run experiments -------------------------------------------------------------------------------------------------------
# Dictionary of experiments, now referencing the predefined query engine objects
experiments = {
    "Classic VDB + Naive RAG": query_engine_naive,
    "Classic VDB + LLM Rerank": query_engine_llm_rerank,
    "Classic VDB + HyDE": query_engine_hyde,
    "Classic VDB + HyDE + LLM Rerank": query_engine_hyde_llm_rerank,
    "Sentence window retrieval": query_engine_sentence_window,
    "Sentence window retrieval + LLM Rerank": query_engine_sentence_window_llm_rerank
}


# Initialize an empty DataFrame to collect results from all experiments
all_experiments_results_df = pd.DataFrame(columns=['Run', 'Experiment', 'OverallScores'])

# Loop through each experiment configuration, run it, and collect results
for experiment_name, query_engine in experiments.items():
    experiment_results_df = run_experiment(experiment_name,
                                            query_engine,
                                            scorer,
                                            benchmark,
                                            validate_api,
                                            config['tonic_validate_project_key'],
                                            upload_results=True,
                                            runs=10)  # Adjust the number of runs as needed

    # Append the results of this experiment to the master DataFrame
    all_experiments_results_df = pd.concat([all_experiments_results_df, experiment_results_df], ignore_index=True)

# Assuming all_experiments_results_df is your DataFrame
all_experiments_results_df['RetrievalPrecision'] = all_experiments_results_df['OverallScores'].apply(lambda x: x['retrieval_precision'])

# Check for normality and homogeneity of variances
from scipy.stats import shapiro
# Test for normality for each group
for exp in experiments.keys():
    stat, p = shapiro(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'])
    print(f"{exp} - Normality test: Statistics={stat}, p={p}")
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

from scipy.stats import levene

# Test for equal variances
stat, p = levene(*(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'] for exp in experiments.keys()))
print(f"Levene’s test: Statistics={stat}, p={p}")
if p > alpha:
    print('Equal variances across groups (fail to reject H0)')
else:
    print('Unequal variances across groups (reject H0)')

import scipy
# ANOVA
f_value, p_value = scipy.stats.f_oneway(*(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'] for exp in experiments.keys()))
print(f"ANOVA F-Value: {f_value}, P-Value: {p_value}")

# If ANOVA assumptions are not met, use Kruskal-Wallis
# h_stat, p_value_kw = stats.kruskal(*(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'] for exp in experiments.keys()))
# print(f"Kruskal-Wallis H-Stat: {h_stat}, P-Value: {p_value_kw}")

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Extract just the relevant columns for easier handling
data = all_experiments_results_df[['Experiment', 'RetrievalPrecision']]

# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(endog=data['RetrievalPrecision'], groups=data['Experiment'], alpha=0.05)
print(tukey_result)

# You can also plot the results to visually inspect the differences
import matplotlib.pyplot as plt
tukey_result.plot_simultaneous()
plt.show()
