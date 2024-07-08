# Report

## Introduction

## YouTube Scraping

### Architecture

- YouTube Data API
  - Channel Information
  - Videos of Channel
- YouTube Scraping
  - Verified Medical Channels
  - Video Transcriptions
  - Video Comments (due to API call limitations)

### Data Model

- Channels
- Videos
- Comments

### Scraping Process

- to cover variety of medical content => prepared list of search terms for every medical discipline and collected channel names of search results
- ..

#### Problems / Lessons Learned

- problems with scraping transcriptions of ads
- scraping on colab
- storage of data

### Data Statistics

- number of videos
- number of channels
- ...

### Data Evaluation

- question: which medical topics are present in the data?

  - semantic cluster visualization

    - clustering by sentences => worked but difficult to extract keywords fro mclusters
    - clustering by title => workd but without preprocessing, clustering is done by Channel Info which is part of title

  - keywords metrics?
    - named entity recognition
      - MESH => CSV (nomentclature)!!!!
      - SNOMED

- question: what kind of language is in the dataset (expert / layman)?

  - readability metrics: https://pypi.org/project/py-readability-metrics/

- question: how informative are the comments?

  - remove comments which provide no additional information

- question: how can the dataset (comments) be further preprocessed? What are interesting use cases for the data?

- ...

## RAG System

### Final Architecture

#### Embedding

- t-systems-roberta-en-de

#### Vector Store

- pinecone

#### Retriever

#### LLM

- claude 3.5 sonnet

#### Query Engine

### Evaluation

question: how well does the RAG system perform in general? Are there big differences between medical disciplines (available data)? How does a standard LLM compare?

- automatically generate questions for each medical field, query and evaluate following metrics:

  - answer_relevancy_metric
  - faithfulness_metric
  - contextual_relevancy_metric
  - hallucination_metric

question: what language (layman / expert) is used in the answers? Can this be improved by prompting? How does a standard LLM compare?

- readability metrics of answers

### Problems / Lessons Learned

- used german-only embedding before, did not work because of english text
- T-Systems-roberta-en-de
- cosine instead of euclidean distance
- when giving instructions besides question, completely wrong documents are retrieved
