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

### Project Structure

# Medfluencer Project Structure

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

### Contact

Jonas Gann\
Student Data and Computer Science\
gann@stud.uni-heidelberg.de
