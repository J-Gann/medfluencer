# YouTube Videos of Medfluencer Channels as Source of Medical Information

## From Advanced Webscraping to Downstream Application

This repository was created as part of a 'Data and Coputer Science' practical at Heidelberg University. The goal of the pracical was to evaluate how medical data retrieved from YouTube could be used for machine learning applications, especially for Retrieval Augmented Generation (RAG).

The advantages of RAG systems are that they can generate text based on a given user prompt and verified or trusted information retrieved from a database. This is especially useful for medical applications, as the generated answer is based on trusted information and the source of the information can be cited for the user to verify.

To evaluate wether YouTube could be used as a source of medical information for RAG systems, a dataset of 94.422 video descriptions, titles and transcriptions as well as 998.721 comments from 362 different medical influencers was created. The content of the dataset was evaluated by semantic clustering. A RAG system was implemented to generate answers based on the dataset and the results were evaluated using "DeepEval".

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation and Usage](#installation-and-usage)
3. [Data Retrieval](#data-retrieval)
4. [Data Analysis](#data-analysis)
5. [RAG System](#rag-system)
6. [RAG Evaluation](#rag-evaluation)
7. [Conclusion](#conclusion)

## Project Structure

```
.
├── 📁 embeddings/    # Contains all embeddings and relevant additional information
│   ├── 📊 comment_embeddings_part_[1-4].npy   # Comment embeddings split into 4 parts
│   ├── 📊 video_description_embeddings.npy    # Embeddings of video descriptions
│   ├── 📊 video_ids.npy                       # Array of (filtered) video IDs
│   ├── 📊 video_title_embeddings.npy          # Embeddings of video titles
│   ├── 📊 video_transcription_chunks_embeddings_part_[1-3].npy  # Chunked transcription embeddings
│   ├── 📄 video_transcription_chunks.json     # Chunked transcriptions
│   └── 📊 video_transcription_embeddings.npy  # Full video transcription embeddings
│
├── 📁 evaluation/    # Contains all evaluation data and metrics
│   ├── 📄 answers_*.json                      # Various RAG answer sets for different scenarios
│   ├── 📄 comment_clustering.json             # Results of comment clustering
│   ├── 📊 comment_embeddings_2d*.npy          # 2D projections of comment embeddings
│   ├── 📦 evaluation_metrics_rag_*.pkl        # Evaluation metrics for RAG system variants
│   ├── 📄 questions_*.json                    # Various question sets for evaluation
│   ├── 📄 video_clustering.json               # Results of video clustering
│   └── 📊 video_description_embeddings_2d.npy # 2D projections of video description embeddings
│
├── 📁 mesh/  # Data from the Medical Subject Headings (MeSH) database
│   ├── 📝 ET.TXT   # Entry terms for MeSH
│   └── 📝 MH.TXT   # Main headings for MeSH
│
├── 📁 scraping/  # Contains all scraped data
│   ├── 📄 channels_scraping.json              # Scraped channel information
│   ├── 📄 comments_scraping_extended.json     # Scraped comments with replies
│   ├── 📄 comments_scraping.json              # Scraped comments without replies
│   ├── 📄 medfluencer_channel_names.json      # List of medfluencer channel names
│   ├── 📄 playlists_scraping.json             # Scraped playlist information
│   └── 📄 videos_scraping.json                # Scraped video information
│
├── 📓 medfluencer_data_analysis.ipynb    # Jupyter notebook for data analysis and clustering
├── 📓 medfluencer_index.ipynb            # Jupyter notebook for embedding and indexing
├── 📓 medfluencer_rag.ipynb              # Jupyter notebook for RAG implementation and evaluation
├── 📓 medfluencer_scrape_channel_names.ipynb  # Jupyter notebook for scraping channel names
├── 📓 medfluencer_scraping.ipynb         # Jupyter notebook for scraping data from YouTube
├── 📄 medical_fields.json                # List and categorization of medical fields
└── 📝 README.md                          # Project documentation and overview
```

## Installation and Usage

This project mainly uses Google Colab for executing juptyer notebooks. Once a notebook is opened on Google Colab, the necessary libraries can be installed with the `%pip install` commands at the beginning of each notebook. It is also recommended to clone this GitHub repository in the Google Colab environment to access the data. The necessary command can be found in ech notebook. A GitHub access token is required.

The scraping of YouTube data was not performed on Google Colab, but on a local machine. The commands for installation of necessary libraries are included at the beginning of the notebook.

## Data Retrieval

This project used a combination of approaches to retrieve relevant medical data from YouTube. This are most importantly:

1. Data Scraping using Selenium
2. Accessing the YouTube Data API

In the end, a dataset of 94.422 video descriptions, titles and transcriptions as well as 998.721 comments from 362 different medical professionals was created.

The implementation can be found in the [scraping notebook](./medfluencer_scraping.ipynb)

### Data Scraping using Selenium

YouTube enables channels to be verified as medical professionals. This verification is indicated by a blue ribbon above the description of each video. However the verification status of a channel is not directly accessible via the YouTube Data API. Therefore this information has to be scraped by inspecting the HTML code of a YouTube video. This was done using the Selenium library in Python.

In order to cover a broad range of medical topics, channel names were scraped by searching for various [medical fields](./medical_fields.json) in the YouTube search bar. The search results were then filtered by channels and the channel names were stored in a [JSON file](./medfluencer_scrape_channel_names.ipynb).

With the channel names available, most data could be accessed using the YouTube Data API, however the transcripts of videos were not available. As this was crucial data for the RAG system it had to be scraped. I decided to implement my own scraping algorithm using Selenium. However, there also exist libraries like `youtube-transcript-api` that can be used.

#### Practical Remarks

- It is important to make sure, that no advertisements are being played during the scraping, as this can lead to transcripts of the ad being scraped instead of the video.
- The text data of the transcripts often contains no punctuation and contains errors in word recognition
- The implementation of the scraping algorithm does not use a database but stores all data in JSON files at [./scraping](./scraping/). This is not recommended for large datasets.

### Accessing the YouTube Data API

For all other relevant data, the YoutTube API can be used. The API provides access to a wide range of data, such as video descriptions, titles, comments, replies, likes, dislikes, etc. The API can be accessed using a Google Cloud Platform account. Request authentication is done using an API key.

#### Practical Remarks

- There is a limit of 10.000 requests per day for the YouTube Data API. This can be a limiting factor when scraping large amounts of data.
- It is advised to look at the cost of each request type, as some requests can be more expensive than others.

## Data Analysis

To analyze the content of the YouTube data, we will first of all take a look at an example of an individual video:

```json
{
  "channel_name": "@MayoClinic",
  "description": "Dr. Burchill dives deeper into the ways his integrated practice meets the needs of patients in a more complete way, from focusing on mental health to understanding cultural background.\n\nFor more information on Dr. Burchill and the Mayo Clinic team’s care for ACHD patients ...",
  "transcription": "If we are genuinely\ncommittedto responding to the\nneeds of our patients,we have to think broadlyabout what those\nneeds are.And I've been\nin this fieldlong enough to know\nthat our patients arenot coming to\nspeak to me justabout their palpitation\nor their chest pain.They're coming with\ncomplicated livesand needs that\nrelate to , yesphysical issues, butalso mental health needs.Also cultural and\nspiritual, and so on ...",
  "title": "Supporting the complete human needs of patients with ACHD, Dr. Luke Burchill, Mayo Clinic"
}
```

For readability reasons, only the first part of the description and transcription are shown. The content is as expected with the description and title giving an overviwe of the video and the transcription being a machine generated text of the spoken content of the video. It is important to note that the transcription is not perfect and contains errors. quite frequently.

Here is an example of a comment:

```json
{
  "text": "only few people can live with enthusiasm like that guy, regardless of their situation.",
  "authorDisplayName": "@mubarakgidadoumar3247",
  "video_id": "ZeV0fL8eKAQ",
  "replies": []
}
```

Again, nothing out of the ordinary. The comment contains the text, the author and the video ID.
It should be noted, that the large comments dataset [./scraping/comments_scraping.json](./scraping/comments_scraping.json) contains comments mostly without replies, while the smaller dataset [./scraping/comments_scraping_extended.json](./scraping/comments_scraping_extended.json) contains comments with replies (when available). The reason lies in the scraping algorithm, which was not able to scrape replies to comments in large quantities due to YouTube rate limits.

To further analyze the content of the data, semantic clustering was used. The goal was to group similar videos or comments together in order to find frequnt topics. The implementation can be found in the [data analysis notebook](./medfluencer_data_analysis.ipynb).

### Semantic Clustering Videos

The semantic clustering for the video dataset was performed as follows:

1. Embed Video Descriptions​
2. Reduce Embedding Dimension to 2 (**UMAP**)​
3. Cluster Points by cosine similarity (**DBSCAN**)​
4. Retrieve Transcriptions of Videos for each Cluster​
5. Remove all words not part of medical keyword dataset (**MESH**)​
6. Sort words by frequency​
7. Ask LLM to infer topic label from top 15 words for each cluster

Here is the result:

![Semantic Clustering of Videos Dataset](./evaluation/images/videos_clustering.png)

On first glance it is impressive, that the clustering resulted in such distinct clusters. However, DBSCAN creates a special class which contains all points that are not part of a cluster. This class is often the largest and contains all outliers. In this visualization, the outlier class is not shown, hence the clusters appear to be very distinct.

The plot shows, that there exists a broad range of medical topics in the dataset, the most frequent ones being:

1. Psychotherapy
2. Orthopaedics
3. Nutrition
4. Skin
5. Infectious Diseases

### Semantic Clustering Comments

The semantic clustering for the comments dataset was performed similar to the video dataset. The only differences are, that the comment text was embedded and used for keyword-extraction and that the keywords were not filtered using the MESH dataset

Here is the result:

![Semantic Clustering of Comments Dataset](./evaluation/images/comments_clustering.png)

Again, the plot shows a broad range of topics. The most frequent ones being:

1. Nutrition / Drug Use
2. Orthopedics
3. COVID
4. Clinic / Operation
5. Deperession / Fear / Neurological Disorders
6. Age / Dementia
7. Parenting

These topics are in one way or another also contained within the video dataset. The following topics are unique to the comments dataset:

1. Opinion
2. Conspiracy
3. Emotion / Spirituality
4. Approval
5. Rejection

### Examples

Now that we have a good overview of the content of the dataset, we can look at some examples of interesting clusters.

!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## RAG System

One basic requirement for a RAG system to operate is the embedding of the dataset. As the data contains german as well as english videos, the embedding was done using an english-german sentence transformer: [T-Systems-onsite/cross-en-de-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer) loaded from huggingface. The implementation can be found in the [index notebook](./medfluencer_index.ipynb) and the resulting embeddings can be found at [./embeddings](./embeddings/). To efficiently use the embeddings in a RAG system, they were also uploaded to a vector store called Pinecone.

To put embedded and indexed dataset to use, two RAG systems were implemented. One for the videos dataset and one for the comments dataset. The implementation can be found in the [RAG notebook](./medfluencer_rag.ipynb).

The RAG systems were implemented using llamaindex and consists of the following modules:

1. Retriever: A retriever module that retrieves the most relevant documents from the dataset based on the user query. The retriever is configured to use the Pinecone vector store and retrieve the top 20 most similar documents.
2. Reranking: An english-german cross encoder is used to further filter the top 5 most relevant documents.
3. Response Synthesizer: An LLM called Claude 3.5 Sonnet was used for the generation of the final response.

## RAG Evaluation

To evaluate the quality of the RAG system and the underlying dataset, an LLM was used to generate 5 medical questions for each of the [medical fields](./medical_fields.json) which would then be answered by the RAG system.

To evaluate the answers of the RAG system, the DeepEval framework was used. The framework provides a set of evaluation metrics to evaluate the quality of the generated answers. The evaluation metrics are:

1. Answer Relevancy: This metric measures the relevance of the answer to the question.
2. Faithfulness: This metric measures if there are any contradictions between the answer and the retrieved context.
3. Context Relevancy: This metric measures the relevance of the retrieved context to the question.
4. Hallucination: This metric measures if the answer contains any information that is not present in the retrieved context.

Here are the results of the evaluation for the videos dataset:

### Contact

Jonas Gann\
Student Data and Computer Science\
gann@stud.uni-heidelberg.de
