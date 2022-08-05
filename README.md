# Searching for patterns in a large collection of children drawings

Author: Ravinithesh Annapureddy

Supervisor: Professor Frédéric Kaplan, Dr. Julien Fageot

Digital Humanities Lab, Spring Semester 2022

<br/>

## Abstract

The success of large-scale digitization projects at museums, archives, and libraries is pushing other cultural institutions to embrace digitization to preserve their collections. By juxtaposing digital tools with digitized collections, it is now possible to study these cultural objects at a previously unknown scale. This thesis is the first attempt to explore a recently digitized children's drawings collection while developing a system to identify patterns in them linked with popular cultural objects. Artists, as young as three and as old as 25, created nearly 90,000 drawings in the span of three decades from most countries in the world. The preliminary examination unveils that these drawings mirror a solid cultural ethos by using specific iconographic subjects, objects, and colors, and the distinction between children of different parts of the globe is visible in their works. These factors not only make the dataset distinct from other sketch datasets but place it distantly from them in terms of size and multifariousness of creations and the creators. The essential and another dimension of the project is matching the drawings and the popular cultural objects they represent. A deep learning model that learns a metric to rank the visual similarity between the images is used to identify the drawing-artwork pairs. Though the networks developed for image classification perform inadequately for the matching task, networks used for pattern matching in paintings show good performance. Fine-tuning the models increases the performance drastically. The primary outcomes of this work are (1) systems trained with a few methodically chosen examples perform comparably to the systems trained on thousands of generic samples and (2) using drawings enriched by adding generic effects of watercolor, oil painting, pencil sketch, and texturizing mitigates the situation of network learning examples by heart.

<br/>

## Structure

- `notebooks/`: Contains the notebooks used in creation of metadata, plots and result analysis.
  - `EDA/`:
    - `descriptive_analysis.ipynb`: This notebook explores the metadata to plot the data about the artists age groups, countries and the years of the drawings submissions.
    - `sample_drawings_viz.ipynb`: This notebook visualizes random drawings in a grid format.
  - `IO/`:
    - `create_drawings_metadata.ipynb`: This notebook generates the metadata file for the drawings.
    - `create_famous-artworks_metadata.ipynb`: This notebook generates the metadata file for the famous artworks dataset obtained on Kaggle.
    - `uid2path_file_creation.ipynb`: This notebook generates a pickle file with mapping between the image id and its path.
    - `extract_subset_metadata.ipynb`: This notebook extracts the metadata of the subset of drawings.
    - `drawings_style_transfer.ipynb`: This notebook converts the drawings into various styles.
    - `manual_annotation_to_train_split.ipynb`: This notebook splits the drawing-artwork pairs into sets to use in training and evaluation of the CNN model.
  - `results_analysis/`: Contains the results analysis of each model variant experimented in this project. In addition there are 2 notebooks (validation_metrics_plots, test_metrics_plots) that generates the plots and tables used in the report.
- `src/`: Contains the python script used to train the model
  - `model/`: Contains the dataloader and script to execute training
    - `dataloader_insdraw.py`: Contains the script to load the triplets
    - `drawings_net.py`: Contains the model structure and function to generate a feature vector for a given image.
    - `train_model.py`: Contains the function to train the model and set the hyperparameters.
  - `utils/`: Contains the utility functions
    - `metrics.py`: Contains the function to calculate the precision and recall
    - `utils.py`: Contains the utility functions ranging from calculating and saving embeddings of a dataset to split the pairs to train and test to setup the data for the webtool and print the metrics.
- `web_annotations/`: Contains the Html, CSS and JavaScript files used to visualise the artworks similar to drawings.
  - `server.py`: Contains the main script to launch the web server. See Readme in the folder on how to use.
- `data/`: Contains the (some) data required to execute the scripts and notebooks. Large files and folders are stored separately.
  - `embeddings/`: Contains the pre-computed embeddings on the artworks and drawings using pre-trained weights, fine-tuned weights and weights of the network trained to identify patterns in artworks. They are stored in a npy file which stores an array with each sub element of the array containing the id and the embedding.
    - `drawings_14-25_resnext-101_avg_280_epoch_0_initial.npy`: Contains the embeddings of drawings of children aged 14-25 computed using pre-trained weights of ResNeXt-101 model trained on the ImageNet with average pooling and 280 image resolution.
    - `drawings_learning_resnet101_avg_280_epoch_0_initial.npy`: Contains the embeddings of subset drawings from `drawings_14-25_resnext-101_avg_280_epoch_0_initial.npy` that are used in the training of the model.
    - `drawings_14-25_resnext-101_avg_280_ft_aug.npy`: Contains the embeddings of drawings of children aged 14-25 computed using the fine tuned ResNeXt-101 model trained with style augmented examples.
    - `drawings_learning_resnext-101_avg_280_ft_aug.npy`: Contains the embeddings of subset drawings from `drawings_14-25_resnext-101_avg_280_ft_aug.npy` that are used in the training of the model.
    - `drawings_14-25_resnext-101_avg_280_mini_replica.npy`: Contains the embeddings of drawings of children aged 14-25 computed using the pre-trained weights of ResNeXt-101 model trained for patterns recognition in artworks.
    - `drawings_learning_resnext-101_avg_280_mini_replica.npy`: Contains the embeddings of subset drawings from `drawings_14-25_resnext-101_avg_280_mini_replica.npy` that are used in the training of the model.
    - `drawings_14-25_resnext-101_avg_280_clus_replica.npy`: Contains the embeddings of drawings of children aged 14-25 computed using the pre-trained weights of ResNeXt-101 model trained for patterns clustering in artworks.
    - `drawings_learning_resnext-101_avg_280_clus_replica.npy`: Contains the embeddings of subset drawings from `drawings_14-25_resnext-101_avg_280_clus_replica.npy` that are used in the training of the model.
    - `drawings_14-25_resnext-101_avg_280_mini_replica_ft_aug.npy`: Contains the embeddings of drawings of children aged 14-25 computed using the fine tuned weights of ResNeXt-101 model trained for patterns recognition in artworks using style augmented drawings.
    - `drawings_learning_resnext-101_avg_280_mini_replica_ft_aug.npy`: Contains the embeddings of subset drawings from `drawings_14-25_resnext-101_avg_280_mini_replica_ft_aug.npy` that are used in the training of the model.
    - `drawings_14-25_resnext-101_avg_280_clus_replica_ft_aug.npy`: Contains the embeddings of drawings of children aged 14-25 computed using the pre-trained weights of ResNeXt-101 model trained for patterns clustering in artworks using style augmented drawings.
    - `drawings_learning_resnext-101_avg_280_clus_replica_ft_aug.npy`: Contains the embeddings of subset drawings from `drawings_14-25_resnext-101_avg_280_clus_replica_ft_aug.npy` that are used in the training of the model.
    - `famous_artworks_resnext-101_avg_280_epoch_0_initial.npy`: Contains the embeddings of artworks (BAAT dataset) computed using pre-trained weights of ResNeXt-101 model trained on the ImageNet with average pooling and 280 image resolution.
    - `famous_artworks_resnext-101_avg_280_ft_aug.npy`: Contains the embeddings of famous artworks (BAAT dataset) computed using the fine tuned ResNeXt-101 model trained with style augmented examples.
    - `famous_artworks_resnext-101_avg_280_mini_replica.npy`: Contains the embeddings of famous artworks (BAAT dataset) computed using the pre-trained weights of ResNeXt-101 model trained for patterns recognition in artworks.
    - `famous_artworks_resnext-101_avg_280_clus_replica.npy`: Contains the embeddings of famous artworks (BAAT dataset) computed using the pre-trained weights of ResNeXt-101 model trained for patterns clustering in artworks.
    - `famous_artworks_resnext-101_avg_280_mini_replica_ft_aug.npy`: Contains the embeddings of famous artworks (BAAT dataset) computed using the fine tuned weights of ResNeXt-101 model trained for patterns recognition in artworks using style augmented drawings.
    - `famous_artworks_resnext-101_avg_280_clus_replica_ft_aug.npy`: Contains the embeddings of famous artworks (BAAT dataset) computed using the pre-trained weights of ResNeXt-101 model trained for patterns clustering in artworks using style augmented drawings.
  - `image_data/`: Contains the drawings and artworks
    - `drawings/`: Train_01 to Train_06 folders contain the scans of drawings as provided by IMAJ. Train_00 contains the style transferred drawings.
    -`famous_artworks/`: Contains the artworks of famous artists per folder, others folder contains images added additionally to BAAT dataset.
  - `manual_annotation/`:
    - `annotated_drawing_artwork_pairs_without_style_transfer.csv`: Contains file that holds the drawing id and artwork id in two columns for drawings that have a similarity with an artwork. The CSV contains three columns, namely, the id of the drawing, the id of the artwork and a boolean that indicates whether the drawing-artwork pair should be considered or not.
    - `annotated_drawing_artwork_pairs.csv`: Contains file that holds the drawing id and artwork id in two columns for drawings that have a similarity with an artwork. The CSV contains four columns, namely, the id of the drawing, the id of the artwork, boolean that indicates whether the drawing-artwork pair should be considered or not and the parent id of the drawing (parent drawing id is useful to identify the original drawings from a style augmented drawing).
  - `model_learning/`: Contains the files used in training the model
    - `learning_rates/`, `losses/`, `results/`, `scores/`, `triplets/`, `weights/`: The folders are used to saved the respective data during the training process.
    - `drawing_artwork_pair_train_test_split.csv` and 10 other versions of it contains the data split into train, validation and test sets. Each CSV contains the 5 columns namely the id for the drawing artwork pair, the id of the drawing, id of the artwork, the set of the pair (train, test or validation) and the parent drawing id.
  - `text_data/`: Contains the metadata files and id to path mapping.
- `report/`: Contains the tex and files used in generating the thesis document.

<br/>

## Installation and Usage

### Installation

```bash
pip install -r requirements.txt
```

### Sequence of execution

Below are the steps to be followed to (re-)generate the files required to run the notebooks and training step. It is assumed that the repository is cloned and the minimum data (list images of drawings and artwork, annoated pairs CSV file, id to path mapping files, initial embeddings) is present.

1. `EDA/descriptive_analysis` can be used to obtain the descriptive statistics on the drawings dataset and `EDA/sample_drawings_viz` can be used to visualise random drawings.
2. `annotated_drawing_artwork_pairs_without_style_transfer.csv` contains the a list of drawing artwork pairs and `uid2path.pkl` contains the id to image path mapping. Using these two files, `IO/drawings_style_transfer` notebook can be used to create style augmented drawings.
3. The (updated) metadata files can be created using the `IO/create_drawings_metadata` and `IO/create_famous-artworks_metadata` notebooks.
4. The (updated) id to path mapping file can be created using the `uid2path_file_creation` notebook.
5. `IO/extract_subset_metadata` notebook is executed to obtain the subsIO/et metadata.
6. `IO/drawing_painting_pairs_to_pdf` notebook can be used to visualise the drawing-artwork pairs that were manually annotated.
7. `IO/manual_annotation_to_train_split` notebook can be used to split the drawing-artwork pairs into train, validation and test sets.

### Executing

#### Web Annotation Flask Application

- A small-scale browser tool to help in annotating/visualizing drawing-artwork pairs that can be used in (re-)training of the Deep Neural Network models we have created.
- See `./web_annotation/server.py` for the arguments

```bash
python web_annotation/server.py
```

### Model Training

- To train the Deep Neural Network CNN model run the script below.
- See `./src/model/train_model.py` for the arguments

```bash
python src/model/train_model.py
```

<br/>

## License

Searching for patterns in a large collection of children drawings - Ravinithesh Annapureddy

Copyright (c) 2022 EPFL

This program is licensed under the terms of the GNU GPLv3.
