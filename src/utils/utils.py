import os
import pickle
import sys
from datetime import datetime
from typing import List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import umap
from PIL import Image
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import BallTree
from tqdm import tqdm

from metrics import mean_average_precision, recall_at_k

sys.path.insert(0, "./src/model/")


def winapi_path(file_path: str, windows: bool = True) -> str:
    """Returns the windows path of a file. Helpful for the long path names.

    Parameters
    ----------
    file_path : str
        The original path of the file
    windows : bool
        Set to False if the OS is not windows

    Returns
    -------
    str
        The updated path of the file
    """
    if windows:
        path = os.path.abspath(file_path)
        return "\\\\?\\" + path
    else:
        return file_path


def read_and_preprocess_image(
    image_path: str,
    pytorch_transformations: List,
) -> torch.Tensor:
    """Return the image tensor after following the transformations: Resize to the resolution,
       ColorJitter, RandomhorizontalFlip and Normalize.

    Parameters
    ----------
    image_path : str
        The path to the image
    pytorch_transformations : List
        The pytorch image transformations to be applied to the image

    Returns
    -------
    torch.Tensor
        The image tensor after transformations
    """

    img = Image.open(image_path)
    img = img.convert("RGB")

    return pytorch_transformations(img).unsqueeze(0)


def get_embedding(image_sample: torch.Tensor, model, device: str) -> torch.Tensor:
    """Returns the embedding of a single image using the model.

    Parameters
    ----------
    image_sample : torch.Tensor
        The image to get the embedding
    model :
        The model to use to get the embedding
    device : str
        GPU or CPU device to use


    Returns
    -------
    image_sample_embedding : torch.Tensor
        The feature vector of the image normalized to 1.

    """
    image_sample_embedding = (
        model.predict(image_sample.squeeze(1).to(device))[0].cpu().detach().numpy()
    )
    # image_sample_embedding = (
    #     model.predict(image_sample.to(device)).cpu().detach().numpy()
    # )
    return image_sample_embedding


def get_lower_dimension_embeddings(
    current_embeddings: np.ndarray, reduced_dimension=512, method: str = "umap"
) -> np.ndarray:
    """Returns the embeddings reduced to the lower dimension.

    Parameters
    ----------
    current_embeddings : np.ndarray
        The tensor of tensors containing the uids and embeddings.
    reduced_dimension : int, optional
        The excepted lower dimension, by default 512
    method : str, optional
        The method to use to reduce the dimension, by default "umap"

    Returns
    -------
    np.ndarray
        The tensor of tensors containing the lower dimension embeddings.
    """

    if method == "umap":
        reduced_embeddings = umap.UMAP(
            n_components=reduced_dimension, metric="cosine"
        ).fit(np.vstack(current_embeddings[:, 1]))
        reduced_embeddings = reduced_embeddings.embedding_
    elif method == "pca":
        reduced_embeddings = PCA(n_components=reduced_dimension).fit_transform(
            np.vstack(current_embeddings[:, 1])
        )
    elif method == "svd":
        reduced_embeddings = TruncatedSVD(n_components=reduced_dimension).fit_transform(
            np.vstack(current_embeddings[:, 1])
        )
    elif method == "tsne":
        reduced_embeddings = TSNE(n_components=reduced_dimension).fit_transform(
            np.vstack(current_embeddings[:, 1])
        )
    else:
        print("unknow method to reduce the dimensions of embeddings")
        reduced_embeddings = current_embeddings

    return reduced_embeddings


def make_tree(artworks_embeddings: np.ndarray, reverse_map: bool = False):
    """Returns the BallTree build based on the artworks that are used to compare the drawings.

    Parameters
    ----------
    artworks_embeddings : ndarray
        The array of embeddings of the artworks that are used to compare the drawings
    reverse_map : bool, optional
        If True, the reverse map is returned, by default False.
    Returns
    -------
    _type_
        _description_
    """
    if reverse_map:
        kdt = BallTree(np.vstack(artworks_embeddings[:, 1]), metric="euclidean")
        reverse_map = {
            k: artworks_embeddings[k, 0] for k in range(len(artworks_embeddings))
        }
        return kdt, reverse_map
    else:
        kdt = BallTree(np.vstack(artworks_embeddings[:, 1]), metric="euclidean")
        return kdt


def find_most_similar_artwork(
    uid: str,
    tree,
    drawings_embeddings: np.ndarray,
    uids: List[str],
    man_anno_arts: List[str] = [],
    n=401,
) -> List[str]:
    """Returns the ids of the most similar artworks.

    Parameters
    ----------
    uid : str
        The ids of the children drawing for which we want to find the most similar artworks.
    tree : _type_
        The BallTree build based on the artworks that are used to compare the drawings
    drawings_embeddings : np.ndarray
        The array of array of id and embeddings for the children drawings
    uids : List[str]
        The list of ids of the artworks
    man_anno_arts : List[str]
        The list of ids of the artworks that are manually annotated similar to the drawing.
    n : int, optional
        The number of artworks to check the similarity, by default 401. This number purne the tree and increases the speed.

    Returns
    -------
    List[str]
        The list of ids of the artworks that are most similar to the children drawing.
    """
    img = np.vstack(
        drawings_embeddings[drawings_embeddings[:, 0] == uid][:, 1]
    ).reshape(1, -1)
    dists, ids = tree.query(img, k=n)[0][0], tree.query(img, k=n)[1][0]
    # return [uids[c] for c in cv if uids[c] != uid]
    if man_anno_arts:
        return [
            (uids[c], round(dist, 3))
            for dist, c in zip(dists, ids)
            if uids[c] not in man_anno_arts
        ]
    return [(uids[c], round(dist, 3)) for dist, c in zip(dists, ids)]


def annotation_setup(
    drawings_metadata_path: str,
    artworks_metadata_path: str,
    drawings_embeddings_path: str,
    artworks_embeddings_path: str,
    size: int = 1000,
):
    """_summary_

    Parameters
    ----------
    drawings_metadata_path : str
        _description_
    artworks_metadata_path : str
        _description_
    drawings_embeddings_path : str
        _description_
    artworks_embeddings_path : str
        _description_
    size : int, optional
        _description_, by default 1000

    Returns
    -------
    _type_
        _description_
    """

    drawings_df = pd.read_csv(drawings_metadata_path)
    values_to_remove = ["_gray", "_oil", "_pencil_gray", "_texture", "_water"]
    pattern = "|".join(values_to_remove)
    drawings_df = drawings_df.loc[
        ~(drawings_df["uid"].str.contains(pattern, case=False))
    ]
    artworks_df = pd.read_csv(artworks_metadata_path)
    drawings_embeddings = np.load(drawings_embeddings_path, allow_pickle=True)
    artworks_embeddings = np.load(artworks_embeddings_path, allow_pickle=True)
    artworks_names = artworks_embeddings[:, 0].tolist()

    artworks_tree = make_tree(artworks_embeddings)

    return (
        drawings_embeddings,
        artworks_tree,
        drawings_df,
        artworks_df,
        artworks_names,
    )


def get_links(
    drawings_embeddings: np.ndarray,
    drawings_df: pd.DataFrame,
    artworks_df: pd.DataFrame,
    artworks_tree,
    artworks_names: List[str],
    drawing_uid=False,
    number_of_similar_artworks: int = 15,
):
    """_summary_

    Parameters
    ----------
    drawings_embeddings : np.ndarray
        _description_
    drawings_df : pd.DataFrame
        _description_
    artworks_df : pd.DataFrame
        _description_
    artworks_tree : _type_
        _description_
    artworks_names : List[str]
        _description_
    drawing_uid : bool, optional
        _description_, by default False
    number_of_similar_artworks : int, optional
        _description_, by default 15

    Returns
    -------
    _type_
        _description_
    """

    if drawing_uid:
        drawing_row = drawings_df[drawings_df["uid"] == drawing_uid]
    else:
        drawing_row = drawings_df.sample()
        drawing_uid = drawing_row["uid"].values[0]

    drawing_path = drawing_row["path"].values[0]

    similar_artwork_uids_dists = find_most_similar_artwork(
        drawing_uid,
        artworks_tree,
        drawings_embeddings,
        artworks_names,
        n=number_of_similar_artworks,
    )

    similar_artworks = []

    for similar_artwork_uid, similar_artwork_dist in similar_artwork_uids_dists:
        similar_artworks.append(
            (
                similar_artwork_uid,
                artworks_df[artworks_df["uid"] == similar_artwork_uid]["path"].values[
                    0
                ],
                similar_artwork_dist,
            ),
        )

    return (drawing_uid, drawing_path), similar_artworks


def store_edges(
    drawing_name: str,
    artwork_names: List[str],
    edges_file_path: str,
    drawing_to_artworks: bool,
):

    current_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    if drawing_to_artworks:
        links = [
            [
                drawing_name + "_" + artwork_name,
                drawing_name,
                artwork_name,
                "POSITIVE",
                current_time,
            ]
            for artwork_name in artwork_names
        ]
    else:
        links = [
            [
                artwork_name + "_" + drawing_name,
                artwork_name,
                drawing_name,
                "POSITIVE",
                current_time,
            ]
            for artwork_name in artwork_names
        ]

    new_edges = pd.DataFrame(
        links,
        columns=[
            "connection_uid",
            "drawings",
            "artwork",
            "connection_type",
            "annotated_at",
        ],
    )

    if os.path.exists(edges_file_path):
        with open(edges_file_path, "rb") as f:
            current_edges = pickle.load(f)
    else:
        current_edges = pd.DataFrame([])

    updated_edges = pd.concat([current_edges, new_edges], axis=0)
    updated_edges = updated_edges.drop_duplicates(subset="connection_uid", keep="first")
    print("New pairs: ", updated_edges.tail())
    print(
        "Number of edges before {}, Number of new edges {} and Number of edges after update {}".format(
            current_edges.shape, new_edges.shape, updated_edges.shape
        )
    )
    with open(edges_file_path, "wb") as f:
        pickle.dump(updated_edges, f)


def show_images(imgages_paths: List):
    f, axarr = plt.subplots(1, 3, figsize=(20, 10))
    axarr = axarr.flatten()
    for i, name in enumerate(imgages_paths):
        img = Image.open(name)
        axarr[i].imshow(img)

    plt.show()


def find_pos_matches(uids_sim, uids_match, how="all", n=400):
    matched = list(filter(lambda i: uids_sim[i] in uids_match, range(len(uids_sim))))
    while len(matched) < len(uids_match):
        matched.append(n)
    if how == "all":
        if len(matched) > 0:
            return matched
        else:
            return [n]
    elif how == "first":
        if len(matched) > 0:
            return matched[0]
        else:
            return n
    elif how == "median":
        if len(matched) > 0:
            return np.median(np.array(matched))
        else:
            return n


def make_rank(uids_sim, uids_match):
    return [1 if uid in uids_match else 0 for uid in uids_sim]


def get_scores(
    drawings_embeddings,
    artworks_embeddings,
    draw_art_pairs,
    set_type: List[str],
    number_of_similar_artworks: int = 15,
    list_downloaded=False,
):

    artworks_tree = make_tree(artworks_embeddings)
    artworks_names = artworks_embeddings[:, 0].tolist()

    nonmatching_inputs = []
    matching_inputs = []
    pos = []
    ranks = []

    if not list_downloaded:
        list_downloaded = list(draw_art_pairs["drawing_id"]) + list(
            draw_art_pairs["artwork_id"]
        )

    for i in tqdm(range(draw_art_pairs.shape[0])):
        if (
            (draw_art_pairs["drawing_id"][i] in list_downloaded)
            and (draw_art_pairs["artwork_id"][i] in list_downloaded)
            and (draw_art_pairs["set"][i] in set_type)
        ):
            list_theo = (
                list(
                    draw_art_pairs[
                        draw_art_pairs["drawing_id"] == draw_art_pairs["drawing_id"][i]
                    ]["artwork_id"]
                )
                + list(
                    draw_art_pairs[
                        draw_art_pairs["artwork_id"] == draw_art_pairs["drawing_id"][i]
                    ]["drawing_id"]
                )
                + [draw_art_pairs["drawing_id"][i]]
            )
            matching_inputs.append(list_theo)

            similar_artwork_uids_dists = find_most_similar_artwork(
                draw_art_pairs["drawing_id"][i],
                artworks_tree,
                drawings_embeddings,
                artworks_names,
                [],
                n=number_of_similar_artworks,
            )
            # n=min(draw_art_pairs.shape[0], 4000)

            list_sim = [
                similar_artwork_uid
                for similar_artwork_uid, _ in similar_artwork_uids_dists
            ]

            # nonmatching_inputs.append(list_sim[:400])
            # matches = find_pos_matches(list_sim[:400], list_theo, how="all")
            nonmatching_inputs.append(list_sim)
            matches = find_pos_matches(list_sim, list_theo, how="all")
            pos.append(matches)
            rank = make_rank(list_sim, list_theo)
            ranks.append(rank)

    posses = [po for p in pos for po in p]
    posses_min = [p[0] for p in pos]
    posses_med = [np.median(np.array(p)) for p in pos]

    mean_position = np.mean(np.array(posses))
    mean_min_position = np.mean(np.array(posses_min))
    mean_median_position = np.mean(np.array(posses_med))

    print("Mean position of artwork: ", mean_position)
    print("Mean of Minimum position of artwork: ", mean_min_position)
    print("Mean of median position of artwork: ", mean_median_position)

    map = mean_average_precision(ranks)
    print("Mean Average Precision: ", map)

    recall_400 = np.mean([recall_at_k(ranks[i], 400) for i in range(len(ranks))])
    recall_200 = np.mean([recall_at_k(ranks[i], 200) for i in range(len(ranks))])
    recall_100 = np.mean([recall_at_k(ranks[i], 100) for i in range(len(ranks))])
    recall_50 = np.mean([recall_at_k(ranks[i], 50) for i in range(len(ranks))])
    recall_20 = np.mean([recall_at_k(ranks[i], 20) for i in range(len(ranks))])
    print("recall @ 400", recall_400)
    print("recall @ 200", recall_200)
    print("recall @ 100", recall_100)
    print("recall @ 50", recall_50)
    print("recall @ 20", recall_20)

    return (
        mean_position,
        mean_min_position,
        mean_median_position,
        map,
        recall_400,
        recall_200,
        recall_100,
        recall_50,
        recall_20,
    )


def catch(x, uid2path):
    try:
        return uid2path.loc[uid2path["uid"] == x, "path"].iloc[0]
    except:
        return np.nan


def make_training_set_orig(
    drawings_embeddings,
    artworks_embeddings,
    train_test_data,
    data_dir,
    uid2path,
    epoch,
    n=10,
):

    artworks_tree = make_tree(artworks_embeddings)
    artworks_names = artworks_embeddings[:, 0].tolist()

    nonmatching_inputs = []
    for i in tqdm(range(train_test_data.shape[0])):
        list_theo = (
            list(
                train_test_data[
                    train_test_data["drawing_id"] == train_test_data["drawing_id"][i]
                ]["artwork_id"]
            )
            + list(
                train_test_data[
                    train_test_data["artwork_id"] == train_test_data["drawing_id"][i]
                ]["drawing_id"]
            )
            + [train_test_data["drawing_id"][i]]
        )

        similar_artwork__wo_theor_uids_dists = find_most_similar_artwork(
            train_test_data["drawing_id"][i],
            artworks_tree,
            drawings_embeddings,
            artworks_names,
            list_theo,
            n=n + 1,
        )

        list_sim = [
            similar_artwork_uid
            for similar_artwork_uid, _ in similar_artwork__wo_theor_uids_dists
        ]

        nonmatching_inputs.append(list_sim)

    train_test_data["C"] = nonmatching_inputs

    triplets = train_test_data[["drawing_id", "artwork_id", "C", "set"]].explode("C")
    triplets.columns = ["A", "B", "C", "set"]
    triplets["A_path"] = triplets["A"].apply(lambda x: catch(x, uid2path))
    triplets["B_path"] = triplets["B"].apply(lambda x: catch(x, uid2path))
    triplets["C_path"] = triplets["C"].apply(lambda x: catch(x, uid2path))

    triplets = triplets[
        triplets["C_path"].notnull()
        & triplets["A_path"].notnull()
        & triplets["B_path"].notnull()
    ]
    print("Model learning pairs size: ", triplets.shape)

    print(
        "Saving train, test and validation triplets at {}model_learning/triplet_XXXX_{}.csv".format(
            data_dir, epoch
        )
    )

    triplets[triplets["set"] == "train"].reset_index().to_csv(
        data_dir + "model_learning/triplets/triplet_train_" + str(epoch) + ".csv",
        index=False,
    )

    triplets[triplets["set"] == "test"].reset_index().to_csv(
        data_dir + "model_learning/triplets/triplet_test_" + str(epoch) + ".csv",
        index=False,
    )

    triplets[triplets["set"] == "val"].reset_index().to_csv(
        data_dir + "model_learning/triplets/triplet_val_" + str(epoch) + ".csv",
        index=False,
    )

    return triplets


def make_training_set_classification(
    drawings_embeddings,
    artworks_embeddings,
    train_test_data,
    data_dir,
    uid2path,
    epoch,
    n=10,
):

    artworks_tree = make_tree(artworks_embeddings)
    artworks_names = artworks_embeddings[:, 0].tolist()

    nonmatching_inputs = []
    for i in tqdm(range(train_test_data.shape[0])):
        list_theo = list(
            train_test_data[
                train_test_data["drawing_id"] == train_test_data["drawing_id"][i]
            ]["artwork_id"]
        )

        similar_artwork__wo_theor_uids_dists = find_most_similar_artwork(
            train_test_data["drawing_id"][i],
            artworks_tree,
            drawings_embeddings,
            artworks_names,
            list_theo,
            n=n + 1,
        )

        list_sim = [
            similar_artwork_uid
            for similar_artwork_uid, _ in similar_artwork__wo_theor_uids_dists
        ]

        nonmatching_inputs.append(list_sim)

    similar_pairs = train_test_data[["drawing_id", "artwork_id", "set"]]
    similar_pairs["label"] = 1
    similar_pairs.columns = ["A", "B", "set", "label"]

    train_test_data["C"] = nonmatching_inputs

    dissimilar_pairs = train_test_data[["drawing_id", "C", "set"]].explode("C")
    dissimilar_pairs["label"] = -1
    dissimilar_pairs.columns = ["A", "B", "set", "label"]

    pairs = pd.concat([similar_pairs, dissimilar_pairs], ignore_index=True)

    pairs["A_path"] = pairs["A"].apply(lambda x: catch(x, uid2path))
    pairs["B_path"] = pairs["B"].apply(lambda x: catch(x, uid2path))

    pairs = pairs[pairs["A_path"].notnull() & pairs["B_path"].notnull()]
    print("Model learning pairs size: ", pairs.shape)

    print(
        "Saving train, test and validation triplets at {}model_learning/pairs/pair_XXXX_{}.csv".format(
            data_dir, epoch
        )
    )

    pairs[pairs["set"] == "train"].reset_index().to_csv(
        data_dir + "model_learning/pairs/pair_train_" + str(epoch) + ".csv",
        index=False,
    )

    pairs[pairs["set"] == "test"].reset_index().to_csv(
        data_dir + "model_learning/pairs/pair_test_" + str(epoch) + ".csv",
        index=False,
    )

    pairs[pairs["set"] == "val"].reset_index().to_csv(
        data_dir + "model_learning/pairs/pair_val_" + str(epoch) + ".csv",
        index=False,
    )

    return pairs


def get_embeddings_dataset(
    images_metadata_df,
    model,
    device,
    transformations,
    data_dir,
    images_names_list: Union[List[str], None] = None,
):
    """Returns the embeddings of all images in a dataset

    Parameters
    ----------
    images_metadata_df :pd.DataFrame
        Dataframe containing the metadata i.e. the uid(unique id of the image), path(relative to the data directory) of the images
    model : The CNN model to be used to compute the embeddings
        _description_
    device : str
       cpu or cuda
    transformations : _type_
        Transformations to be applied to the images
    images_names_list : Union[List[str], None], optional
        The uids of images for which embeddings are needed, by default None then embeddings for all images are returned

    Returns
    -------
    _type_
        _description_
    """

    reqd_images_metadata_df = images_metadata_df.copy()
    if images_names_list:
        reqd_images_metadata_df = images_metadata_df[
            images_metadata_df["uid"].isin(images_names_list)
        ]

    reqd_image_embeddings = [
        [
            uid,
            get_embedding(
                read_and_preprocess_image(
                    winapi_path(data_dir + path, False), transformations
                ),
                model,
                device,
            ),
        ]
        for uid, path in tqdm(reqd_images_metadata_df[["uid", "path"]].values.tolist())
    ]

    return np.array(reqd_image_embeddings, dtype=np.ndarray)


def get_train_test_split(pairs_df):
    """Take input of drawing, artwork pairs and splits them into train, test and val sets based on connected components.
    Columns of pairs_df: drawing_id, artwork_id, uid_connection, annotated

    Parameters
    ----------
    pairs_df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    morphograph_edges_df = pairs_df.copy()
    # creating connected components
    G = nx.from_pandas_edgelist(
        morphograph_edges_df,
        source="drawing_id",
        target="artwork_id",
        create_using=nx.DiGraph(),
        edge_key="uid_connection",
    )
    components = [x for x in nx.weakly_connected_components(G)]

    print(morphograph_edges_df.shape, len(components))
    # adding set specification to df
    mapper = {it: number for number, nodes in enumerate(components) for it in nodes}
    morphograph_edges_df["cluster"] = morphograph_edges_df["drawing_id"].apply(
        lambda x: mapper[x]
    )

    morphograph_edges_df["set"] = [
        "test" if cl % 4 == 0 else "train" for cl in morphograph_edges_df["cluster"]
    ]

    morphograph_edges_df["set"] = [
        "val" if cl % 6 == 0 else set_
        for cl, set_ in zip(
            morphograph_edges_df["cluster"], morphograph_edges_df["set"]
        )
    ]

    ################################################
    ## uncomment to make only train and val set ##
    ################################################

    # morphograph_edges_df["set"] = [
    #     "val" if cl % 4 == 0 else "train" for cl in morphograph_edges_df["cluster"]
    # ]
    morphograph_edges_df = morphograph_edges_df.reindex(
        columns=[
            "uid_connection",
            "drawing_id",
            "artwork_id",
            "annotated",
            "cluster",
            "set",
            "parent",
        ]
    )

    return morphograph_edges_df


def save_embeddings(
    data_dir: str,
    model_name: str,
    resolution: int,
    epoch: int,
    time: str,
    dataset_name: str,
    embeddings_array: np.ndarray,
):
    updated_embeddings_path = (
        data_dir
        + "embeddings/"
        + dataset_name
        + "_"
        + model_name
        + "_"
        + "avg_"
        + str(resolution)
        + "_epoch_"
        + str(epoch)
        + "_"
        + time
        + ".npy"
    )
    np.save(updated_embeddings_path, embeddings_array, allow_pickle=True)


def save_scores(file_path: str, scores_list: List):
    pd.DataFrame(
        scores_list,
        columns=[
            "mean_position",
            "mean_min_position",
            "mean_median_position",
            "map",
            "recall_400",
            "recall_200",
            "recall_100",
            "recall_50",
            "recall_20",
        ],
    ).to_csv(
        file_path,
        index=False,
    )


def save_losses(file_path: str, losses_list: List, column: str = "loss"):
    pd.DataFrame(losses_list, columns=[column],).to_csv(
        file_path,
        index=False,
    )
