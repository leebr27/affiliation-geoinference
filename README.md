# affiliation-geoinference
Geoinference of author affiliations using NLP-based text classification

- Jupyter notebooks #1, #2, and #3 include the code for creating clean training/validation sets from the main MapAffil dataset 

- Jupyter notebook #4 includes the code for creating/saving NLP text-classifier as well as vectorizer

- Jupyter notebook #5 includes the code for testing the model against several different test and validation sets

- Jupyter notebook #6 includes the code for testing different classifiers as well as vectorizers

- The NLP text-vectorizer and text-classifiers are saved to the two files named “geoinference_vectorizer_1mil.joblib.lzma” and “geoinference_linearsvc_1mil.joblib.lzma”   

Note: To reproduce the findings reported in the manuscript, Jupypter notebooks #1/2/3 are unnecessary to run (all the final training/validation sets are saved to the "data" folder). Thus, the MapAffil dataset is NOT included in the “data’ folder within the repository; you must download mapaffil2018.tsv.gz from https://databank.illinois.edu/datasets/IDB-2556310 and save to "data" folder -> this step is only necessary if you desire to run Jupyter notebooks #1/2/3. Jupyter notebooks #4/5/6 can be run without downloading the MapAffil dataset.
