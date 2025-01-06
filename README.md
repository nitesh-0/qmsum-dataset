# QMSum

### Overview
This repository maintains dataset for NAACL 2021 paper: *[QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization](https://arxiv.org/abs/2104.05938)*.

**QMSum** is a new human-annotated benchmark for query-based multi-domain meeting summarization task, which consists of 1,808 query-summary pairs over 232 meetings in multiple domains.

### Dataset
You can access the train/valid/test set of QMSum through the ```data/ALL``` folder. In addition, QMSum is composed of three domains: ```data/Academic```, ```data/Product``` and ```data/Committee``` contain data in a single domain.

The desired format of dataset is available in directory: data/product/processed/

Note: The dataset is converted into desired format only for product meetings since our project mainly focuses business domain and we also have mentioned the dataset is taken from AMI Corpus in our project proposal. 