import os
import sys

sys.path.insert(0, "./src/utils")


import argparse

from flask import Flask, render_template, request, send_from_directory, url_for
from utils import annotation_setup, get_links, store_edges

parser = argparse.ArgumentParser(description="Model specifics")
parser.add_argument("--n_subset", dest="n_subset", type=int, help="", default=10000)
parser.add_argument("--n_show", dest="n_show", type=int, help="", default=99)

parser.add_argument(
    "--edges_file",
    dest="edges_file",
    type=str,
    help="",
    default="./data/generated/edges.pkl",
)

###############################################################
## use these arguments to get similar artworks for a drawing ##
###############################################################

parser.add_argument(
    "--drawings_metadata_path",
    dest="drawings_metadata_path",
    type=str,
    help="Path for the metadata on the drawings.",
    default="./data/text_data/metadata/drawings_14_25_metadata.csv",
)
parser.add_argument(
    "--drawings_embeddings_path",
    dest="drawings_embeddings_path",
    type=str,
    help="Path for the embeddings of the drawings. Current defaults are set to embeddings created with pre-trained weights on ImageNet.",
    default="./data/embeddings/drawings_14-25_resnext-101_avg_280_epoch_0_initial.npy",
)
parser.add_argument(
    "--artworks_metadata_path",
    dest="artworks_metadata_path",
    type=str,
    help="Path for the metadata on the artworks.",
    default="./data/text_data/metadata/famous_artworks_metadata_complete.csv",
)
parser.add_argument(
    "--artworks_embeddings_path",
    dest="artworks_embeddings_path",
    type=str,
    help="Path for the embeddings of the artworks. Current defaults are set to embeddings created with pre-trained weights on ImageNet.",
    default="./data/embeddings/famous_artworks_resnext-101_avg_280_epoch_0_initial.npy",
)

parser.add_argument(
    "--drawing_with_artworks",
    dest="drawing_with_artworks",
    help="Boolean to indicate whether to compare drawing with artworks or artwork with drawings. Set True to get similar artworks for a one drawing",
    default=True,
    action="store_true",
)


args = parser.parse_args()

if not args.drawing_with_artworks:
    (
        args.drawings_metadata_path,
        args.artworks_metadata_path,
        args.drawings_embeddings_path,
        args.artworks_embeddings_path,
    ) = (
        args.artworks_metadata_path,
        args.drawings_metadata_path,
        args.artworks_embeddings_path,
        args.drawings_embeddings_path,
    )

(
    drawings_embeddings,
    artworks_tree,
    drawings_df,
    artworks_df,
    artworks_names,
) = annotation_setup(
    args.drawings_metadata_path,
    args.artworks_metadata_path,
    args.drawings_embeddings_path,
    args.artworks_embeddings_path,
    size=args.n_subset,
)

DATA_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/drawings/<path:filename>")
def download_drawing_file(filename):
    try:
        return send_from_directory(DATA_FOLDER, filename, as_attachment=True)
    except:
        return send_from_directory(DATA_FOLDER, filename, as_attachment=True)


@app.route("/annotate", methods=["GET", "POST"])
def annotate_images():
    similar_artworks = []
    number_of_results = 0
    drawing_name = ""
    drawing_path = ""
    if request.method == "POST":
        if request.form["submit"] in ["text_search", "random_search"]:
            if request.form["submit"] == "text_search":
                drawing_name = request.form["item"]
            else:
                drawing_name = False

            compared_drawing, similar_artworks = get_links(
                drawings_embeddings,
                drawings_df,
                artworks_df,
                artworks_tree,
                artworks_names,
                drawing_uid=drawing_name,
                number_of_similar_artworks=args.n_show + 1,
            )

            drawing_name, drawing_path = compared_drawing
            drawing_path = url_for("download_drawing_file", filename=drawing_path)
            similar_artworks = [
                (
                    artwork_name,
                    url_for("download_drawing_file", filename=artwork_path),
                    artwork_distance,
                )
                for artwork_name, artwork_path, artwork_distance in similar_artworks
            ]

            number_of_results = len(similar_artworks)

        if request.form["submit"] == "similar_images":
            similar_artworks_names = []
            for form_key in request.form.keys():
                if "ckb" in form_key:
                    similar_artworks_names.append(request.form[form_key])
            store_edges(
                request.form["UID_A"],
                similar_artworks_names,
                args.edges_file,
                args.drawing_with_artworks,
            )

    return render_template(
        "annotate.html",
        results=similar_artworks,
        uploaded_image_url=drawing_path,
        number_of_results=number_of_results,
        item=drawing_name,
        cold_start=request.method == "GET",
    )


if __name__ == "__main__":

    app.run(port=7777)
