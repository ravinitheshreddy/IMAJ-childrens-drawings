#!/usr/bin/python
import argparse
import copy
import os
import pickle
import sys
import time
from datetime import datetime
from tkinter import E

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader_insdraw import InsdrawDataset
from drawingsnet import DrawingsNet

sys.path.insert(0, "./src/utils/")

from utils import (
    get_embeddings_dataset,
    get_scores,
    make_training_set_orig,
    save_embeddings,
    save_losses,
    save_scores,
    winapi_path,
)

now = datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M-%S")


def main(
    data_dir,
    uid2path_file,
    draw_art_pairs_file,
    drawings_metadata_file,
    artworks_metadata_file,
    model_name,
    prev_model_weights_file=None,
    cuda=True,
    prev_drawing_embeddings_file=None,
    prev_artworks_embeddings_file=None,
    prev_training_epochs=0,
    resolution=480,
    num_epochs=20,
    num_c=10,
    batch_size=4,
    learning_rate=1e-4,
    margin=1.0,
    style_augmentation=True,
):
    """Main function to train a model starting from pre-trained weights.

    Parameters
    ----------
    data_dir : str
        The parent directory where data is stored
    uid2path_file : str
        File path for the drawings and artworks id to path mapping relative to the data directory
    draw_art_pairs_file : str
        Directory where drawing artwork paris are stored relative to the data directory
    drawings_metadata_file : str
        CSV file path for the drawings metadata relative to the data directory
    artworks_metadata_file : str
        CSV file path for the artworks metadata relative to the data directory
    model_name : str
        Name of pretrained model to use. Options available: resnet50, resnet100, resnet152, densenet161, resnext-101, regnet_y_32gf, vit_b_16, convnext_tiny, efficientnet0, efficientnet7
    prev_model_weights_file : str, optional
        File path for the previous model weights relative to the data directory, if None then pre-trained weights provided by PyTorch are used.
    cuda : bool, optional
        Boolean to indicate whether to use GPu or not, by default True
    prev_drawing_embeddings_file : str, optional
        File path for the embeddings for the drawings. If None, embeddings with the loaded model will be created using PyTorch pre-trained weights, by default None
    prev_artworks_embeddings_file : str, optional
        File path for the embeddings for the artworks. If None, embeddings with the loaded model will be created using PyTorch pre-trained weights, by default None
    prev_training_epochs : int, optional
        If continuing a training process, this value will be used to read the previous epoch's embeddings and triplets, by default 0 which indicates a fresh training.
    resolution : int, optional
        The size to scale the drawings and artworks, by default 480
    num_epochs : int, optional
        The number of epochs to train the model, by default 20
    num_c : int, optional
        The number of triplets per training pair, by default 10
    batch_size : int, optional
        The mini batch size to be used in each epoch of training, by default 4
    learning_rate : _type_, optional
        Learning rate for updating the weights, by default 1e-4
    margin : float, optional
        Margin for triplet loss, by default 1.0
    style_augmentation : bool, optional
        Boolean to indicate whether to use style augmented drawings in training or not. Default is True.

    """

    best_loss = 10000000
    best_epoch = -1
    train_losses = []
    validation_losses = []
    train_scores = []
    test_scores = []
    validation_scores = []
    test_val_scores = []
    learning_rates = []
    best_drawings_embeddings, best_artworks_embeddings = None, None

    # Boolean to check if the machine is linux or windows to account for long file names.
    is_windows = True
    if sys.platform == "linux" or sys.platform == "linux2":
        is_windows = False

    # The set of transformations to be performed on the training images
    train_transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resolution + 50, resolution + 50)),
            transforms.RandomResizedCrop(
                (resolution, resolution),
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(degrees=5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomGrayscale(p=0.4),
        ]
    )

    # The set of transformations to be performed on the images used in validation and test.
    test_transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resolution, resolution)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # setting the location of the drawings and artworks embeddings to a variable.
    drawings_embeddings_file = ""
    artworks_embeddings_file = ""

    if prev_drawing_embeddings_file is not None:
        drawings_embeddings_file = prev_drawing_embeddings_file
    else:
        drawings_embeddings_file = (
            "embeddings/drawings_learning_"
            + model_name
            + "_avg_"
            + str(resolution)
            + "_epoch_0_initial.npy"
        )

    if prev_artworks_embeddings_file is not None:
        artworks_embeddings_file = prev_artworks_embeddings_file
    else:
        artworks_embeddings_file = (
            "embeddings/famous_artworks_"
            + model_name
            + "_avg_"
            + str(resolution)
            + "_epoch_0_initial.npy"
        )

    device = torch.device("cuda" if cuda else "cpu")

    with open(data_dir + uid2path_file, "rb") as outfile:
        uid2path = pickle.load(outfile)

    model = DrawingsNet(model_name, device)

    if prev_model_weights_file is not None:
        prev_model_weights_file = winapi_path(
            data_dir + prev_model_weights_file, is_windows
        )
        if os.path.exists(prev_model_weights_file):
            print("Loading model weights from {}".format(prev_model_weights_file))
            model.load_state_dict(
                torch.load(prev_model_weights_file, map_location="cuda:0")
            )
        else:
            print(
                "The provided model weights file {} does not exist".format(
                    prev_model_weights_file
                )
            )

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    # setting up the triplet loss
    triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
        margin=margin,
        reduction="sum",
    )

    # setting up the Adam optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0.0001
    )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

    print("Setting up model...")
    print("\tModel name: {}".format(model_name))
    print("\tResolution: {}".format(resolution))
    print("\tBatch size: {}".format(batch_size))
    print("\tLearning rate: {}".format(learning_rate))
    print("\tNumber of epochs: {}".format(num_epochs))
    print("\tNumber of c's: {}".format(num_c))
    print("\tMargin for Triplet Loss: {}".format(margin))
    print("\n train_transformations:", train_transformations)

    print(
        "{}_decay0.0001, {}_10-0.1".format(
            optimizer.__class__.__name__,
            scheduler.__class__.__name__,
        )
    )

    # read the drawing and artworks pairs
    draw_art_pairs_df_full = pd.read_csv(data_dir + draw_art_pairs_file)

    if style_augmentation:
        # retain the augmented data only in training set and remove the augmented data in the test and validation set.
        draw_art_pairs_df = draw_art_pairs_df_full[
            (
                (draw_art_pairs_df_full["set"].isin(["train"]))
                | (
                    (draw_art_pairs_df_full["set"].isin(["test", "val"]))
                    & (
                        draw_art_pairs_df_full["drawing_id"]
                        == draw_art_pairs_df_full["parent"]
                    )
                )
            )
        ].reset_index(drop=True)
    else:
        # retain the non augmented data.
        draw_art_pairs_df = draw_art_pairs_df_full[
            (draw_art_pairs_df_full["drawing_id"] == draw_art_pairs_df_full["parent"])
        ].reset_index(drop=True)

    # read the metadata of the drawings and artworks
    drawings_metadata_df = pd.read_csv(data_dir + drawings_metadata_file)
    artworks_metadata_df = pd.read_csv(data_dir + artworks_metadata_file)

    drawings_used = draw_art_pairs_df["drawing_id"].unique().tolist()

    # Load the embedding of the drawings and artworks. If the embeddings are not available on disk, embeddings using PyTorch pre-trained weights are created and stored.
    drawings_embeddings_path = winapi_path(
        data_dir + drawings_embeddings_file, is_windows
    )
    artworks_embeddings_path = winapi_path(
        data_dir + artworks_embeddings_file, is_windows
    )

    if os.path.exists(drawings_embeddings_path):
        drawings_embeddings = np.load(drawings_embeddings_path, allow_pickle=True)
        print("Loaded drawings embeddings...")
    else:
        print(
            "Drawings embeddings not found. Computing embeddings for drawings used in learning..."
        )
        drawings_embeddings = get_embeddings_dataset(
            drawings_metadata_df,
            model,
            device,
            test_transformations,
            data_dir,
            images_names_list=drawings_used,
        )

        save_embeddings(
            data_dir,
            model_name,
            resolution,
            0,
            "initial",
            "drawings_learning",
            drawings_embeddings,
        )

    print("drawings embeddings shape: ", drawings_embeddings.shape)

    if os.path.exists(artworks_embeddings_path):
        artworks_embeddings = np.load(artworks_embeddings_path, allow_pickle=True)
        print("Loaded artworks embeddings...")
    else:
        print("Artworks embeddings not found. Computing embeddings for artworks...")
        artworks_embeddings = get_embeddings_dataset(
            artworks_metadata_df,
            model,
            device,
            test_transformations,
            data_dir,
            images_names_list=None,
        )

        save_embeddings(
            data_dir,
            model_name,
            resolution,
            0,
            "initial",
            "famous_artworks",
            artworks_embeddings,
        )

    print("artworks embeddings shape: ", artworks_embeddings.shape)

    number_of_artworks = len(artworks_embeddings)

    # The initial metrics (before training) are computed
    print("Computing scores for training pairs")
    train_scores_before_train = get_scores(
        drawings_embeddings,
        artworks_embeddings,
        draw_art_pairs_df,
        ["train"],
        number_of_artworks,
    )
    train_scores.append(train_scores_before_train)

    print("Computing scores for validation pairs")
    val_scores_before_train = get_scores(
        drawings_embeddings,
        artworks_embeddings,
        draw_art_pairs_df,
        ["val"],
        number_of_artworks,
    )
    validation_scores.append(val_scores_before_train)

    print("Computing scores for Test pairs")
    test_scores_before_train = get_scores(
        drawings_embeddings,
        artworks_embeddings,
        draw_art_pairs_df,
        ["test"],
        number_of_artworks,
    )

    test_scores.append(test_scores_before_train)

    print("Computing scores for Val & Test pairs")
    test_val_scores_before_train = get_scores(
        drawings_embeddings,
        artworks_embeddings,
        draw_art_pairs_df,
        ["val", "test"],
        number_of_artworks,
    )

    test_val_scores.append(test_val_scores_before_train)

    # Creation of Triplets
    print("Making the train, validation and test triplets")
    make_training_set_orig(
        drawings_embeddings,
        artworks_embeddings,
        draw_art_pairs_df,
        data_dir,
        uid2path,
        epoch=prev_training_epochs,
        n=num_c,
    )
    print(
        "Unique C's in training: ",
        pd.read_csv(
            data_dir
            + "model_learning/triplets/triplet_train_"
            + str(prev_training_epochs)
            + ".csv"
        )["C"].nunique(),
    )
    print("Reloading the data loaders")
    print("Creating dataloaders...")
    loaders = {
        x: InsdrawDataset(
            data_dir
            + "model_learning/triplets/triplet_"
            + x
            + "_"
            + str(prev_training_epochs)
            + ".csv",
            data_dir,
            train_transformations,
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(loaders[x]) for x in ["train", "val"]}

    train_dataloaders = {
        x: DataLoader(loaders[x], batch_size=batch_size, shuffle=True)
        for x in ["train", "val"]
    }

    for param in model.modules():
        if isinstance(param, nn.BatchNorm2d):
            param.requires_grad = False

    # Training/Optimization
    for epoch in range(num_epochs):

        epoch += 1
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)
        current_lr_for_epoch = scheduler.get_last_lr()
        learning_rates.append(current_lr_for_epoch)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:  # , 'val'

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for a, b, c in tqdm(train_dataloaders[phase]):
                a = a.squeeze(1).to(device)
                b = b.squeeze(1).to(device)
                c = c.squeeze(1).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    # Forward pass
                    A, B, C = model(a, b, c)
                    # Compute and print loss
                    loss = triplet_loss(A, B, C)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * a.size(0)

            print("Calculating Loss")
            epoch_loss = running_loss * batch_size / dataset_sizes[phase]
            if phase == "train":
                train_losses.append(epoch_loss)
                scheduler.step()
            elif phase == "val":
                validation_losses.append(epoch_loss)

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "val":

                print("In validation: Recomputing drawings embeddings")
                drawings_embeddings = get_embeddings_dataset(
                    drawings_metadata_df,
                    model,
                    device,
                    test_transformations,
                    data_dir,
                    images_names_list=drawings_used,
                )

                print(
                    "drawings embeddings shape after update: ",
                    drawings_embeddings.shape,
                )

                save_embeddings(
                    data_dir,
                    model_name,
                    resolution,
                    epoch,
                    now,
                    "drawings_learning",
                    drawings_embeddings,
                )

                print("In validation: Recomputing artworks embeddings")
                artworks_embeddings = get_embeddings_dataset(
                    artworks_metadata_df,
                    model,
                    device,
                    test_transformations,
                    data_dir,
                    images_names_list=None,
                )

                print(
                    "artworks embeddings shape after update: ",
                    artworks_embeddings.shape,
                )

                save_embeddings(
                    data_dir,
                    model_name,
                    resolution,
                    epoch,
                    now,
                    "famous_artworks",
                    artworks_embeddings,
                )

                prev_val_score = validation_scores[-1][-1]

                print("Computing scores for training pairs")
                train_scores.append(
                    get_scores(
                        drawings_embeddings,
                        artworks_embeddings,
                        draw_art_pairs_df,
                        ["train"],
                        number_of_artworks,
                    )
                )

                print("Computing scores for validation pairs")
                validation_scores.append(
                    get_scores(
                        drawings_embeddings,
                        artworks_embeddings,
                        draw_art_pairs_df,
                        ["val"],
                        number_of_artworks,
                    )
                )

                print("Computing scores for Test pairs")
                test_scores.append(
                    get_scores(
                        drawings_embeddings,
                        artworks_embeddings,
                        draw_art_pairs_df,
                        ["test"],
                        number_of_artworks,
                    )
                )

                print("Computing scores for Val & Test pairs")
                test_val_scores.append(
                    get_scores(
                        drawings_embeddings,
                        artworks_embeddings,
                        draw_art_pairs_df,
                        ["val", "test"],
                        number_of_artworks,
                    )
                )

                # scheduler.step(validation_scores[-1][-1])

            # deep copy the model
            if phase == "val" and (
                epoch_loss < best_loss or validation_scores[-1][-1] > prev_val_score
            ):  # needs to be changed to val

                print(
                    "Model updating! Best loss so far or better recall @ 20 validation score"
                )
                best_loss = epoch_loss
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

                print("Saving Model weights after {} epochs".format(epoch))
                updated_model_weights_file = (
                    data_dir
                    + "model_learning/weights/model_weights_"
                    + now
                    + "_"
                    + str(epoch)
                    + "_"
                    + model_name
                )
                torch.save(
                    model.state_dict(),
                    updated_model_weights_file,
                )

                best_drawings_embeddings = drawings_embeddings

                best_artworks_embeddings = artworks_embeddings

                print("Making the train, validation and test triplets")
                make_training_set_orig(
                    drawings_embeddings,
                    artworks_embeddings,
                    draw_art_pairs_df,
                    data_dir,
                    uid2path,
                    epoch=epoch,
                    n=num_c,
                )

                print("Reloading the data loaders")
                loaders["train"].__reload__(
                    data_dir
                    + "model_learning/triplets/triplet_train_"
                    + str(epoch)
                    + ".csv"
                )
                loaders["val"].__reload__(
                    data_dir
                    + "model_learning/triplets/triplet_val_"
                    + str(epoch)
                    + ".csv"
                )
                train_dataloaders = {
                    x: DataLoader(loaders[x], batch_size=batch_size, shuffle=True)
                    for x in ["train", "val"]
                }

        if len(train_losses) > 2 and len(set(train_losses[:3])) == 1:
            print("Last 3 train losses are the same. Stopping training")
            break
    time_elapsed = time.time() - since
    print("Trained for {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val loss: {:4f}".format(best_loss))

    print("Computing scores for non-train pairs with best model")
    _ = get_scores(
        best_drawings_embeddings,
        best_artworks_embeddings,
        draw_art_pairs_df,
        ["test", "val"],
        number_of_artworks,
    )

    print("Saving Train Scores")
    save_scores(
        data_dir + "model_learning/scores/train_scores_" + str(now) + ".csv",
        train_scores,
    )

    print("Saving Test Scores")
    save_scores(
        data_dir + "model_learning/scores/test_scores_" + str(now) + ".csv", test_scores
    )

    print("Saving Validation Scores")
    save_scores(
        data_dir + "model_learning/scores/validation_scores_" + str(now) + ".csv",
        validation_scores,
    )

    print("Saving Non Train Scores")
    save_scores(
        data_dir + "model_learning/scores/test-and-val_scores_" + str(now) + ".csv",
        test_val_scores,
    )

    print("Saving Train Losses")
    save_losses(
        data_dir + "model_learning/losses/train_loss_" + str(now) + ".csv",
        train_losses,
    )

    print("Saving Validation Losses")
    save_losses(
        data_dir + "model_learning/losses/validation_loss_" + str(now) + ".csv",
        validation_losses,
    )

    save_losses(
        data_dir + "model_learning/learning_rates/learning_rate" + str(now) + ".csv",
        learning_rates,
        "lr",
    )

    # load best model weights
    model.load_state_dict(best_model_wts)

    print("Saving Best Model weights at Last Epoch")
    updated_model_weights_file = (
        data_dir
        + "model_learning/weights/model_weights_"
        + now
        + "_best_epoch_"
        + str(best_epoch)
        + "_"
        + model_name
    )
    torch.save(
        model.state_dict(),
        updated_model_weights_file,
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model specifics")
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        type=str,
        help="Directory where data is stored",
        default="./data/",
    )
    parser.add_argument(
        "--draw_art_pairs_file",
        dest="draw_art_pairs_file",
        type=str,
        help="Directory where drawing artwork paris are stored relative to the data directory",
        default="model_learning/drawing_artwork_pair_train_test_split.csv",
    )
    parser.add_argument(
        "--drawings_metadata_file",
        dest="drawings_metadata_file",
        type=str,
        help="CSV file path for the drawings metadata relative to the data directory",
        default="text_data/metadata/drawings_learning_metadata.csv",
    )
    parser.add_argument(
        "--artworks_metadata_file",
        dest="artworks_metadata_file",
        type=str,
        help="CSV file path for the artworks metadata relative to the data directory",
        default="text_data/metadata/famous_artworks_metadata_complete.csv",
    )
    parser.add_argument(
        "--uid2path_file",
        dest="uid2path_file",
        type=str,
        help="File path for the drawings and artworks id to path mapping relative to the data directory",
        default="text_data/uid2path.pkl",
    )
    parser.add_argument(
        "--prev_model_weights_file",
        dest="prev_model_weights_file",
        type=str,
        help="File path for the previous model weights relative to the data directory",
        default=None,
    )

    parser.add_argument(
        "--prev_drawing_embeddings_file",
        dest="prev_drawing_embeddings_file",
        type=str,
        help="File path for the embeddings for the drawings. If None, embeddings with the loaded model will be created.",
        default=None,
    )

    parser.add_argument(
        "--prev_artworks_embeddings_file",
        dest="prev_artworks_embeddings_file",
        type=str,
        help="File path for the embeddings for the artworks. If None, embeddings with the loaded model will be created.",
        default=None,
    )

    parser.add_argument(
        "--prev_training_epochs",
        dest="prev_training_epochs",
        type=int,
        help="The epoch value to read the previously stored embeddings",
        default=0,
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, help="Batch size", default=4
    )
    parser.add_argument(
        "--num_epochs",
        dest="num_epochs",
        type=int,
        help="Number of epochs to train the model",
        default=10,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        help="Name of pretrained model to use",
        default="resnext-101",
    )
    parser.add_argument(
        "--cuda",
        dest="cuda",
        help="Set True to use GPU for training",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        help="The size to scale the drawings and artworks",
        default=280,
    )
    parser.add_argument(
        "--num_c",
        dest="num_c",
        type=int,
        help="Number of triplets per training pair",
        default=10,
    )

    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        help="Learning rate for the Optimizer",
        default=1e-6,
    )

    parser.add_argument(
        "--margin",
        dest="margin",
        type=float,
        help="Margin for triplet loss",
        default=0.7,
    )

    parser.add_argument(
        "--style_augmentation",
        dest="style_augmentation",
        help="Set True to use style augmented drawings for training",
        default=True,
        action="store_true",
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        uid2path_file=args.uid2path_file,
        draw_art_pairs_file=args.draw_art_pairs_file,
        drawings_metadata_file=args.drawings_metadata_file,
        artworks_metadata_file=args.artworks_metadata_file,
        model_name=args.model_name,
        prev_model_weights_file=args.prev_model_weights_file,
        cuda=args.cuda,
        prev_drawing_embeddings_file=args.prev_drawing_embeddings_file,
        prev_artworks_embeddings_file=args.prev_artworks_embeddings_file,
        prev_training_epochs=args.prev_training_epochs,
        resolution=args.resolution,
        num_epochs=args.num_epochs,
        num_c=args.num_c,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        margin=args.margin,
        style_augmentation=args.style_augmentation,
    )
