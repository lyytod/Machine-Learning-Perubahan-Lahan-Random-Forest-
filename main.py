# main.py
import yaml
from preprocessing import preprocess
from ndvi_to_class import classify_ndvi
from model import train_and_predict
from evaluate import evaluate_model
from analysis import analyze_statistics
from visualization import generate_statistical_graph
from generate_interactive_map import create_interactive_map
from utils import setup_logging
import logging


def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load configuration
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Step 1: Preprocessing
    preprocess(
        ndvi_from_path=config["paths"]["ndvi_from"],
        ndvi_to_path=config["paths"]["ndvi_to"],
        boundary_path=config["paths"]["boundary"],
        output_dir=config["outputs"]["preprocessed"]
    )

    # Step 2: NDVI to Class
    classify_ndvi(
        ndvi_path=config["outputs"]["preprocessed"] + "/ndvi_from_clipped.tif",
        output_path=config["outputs"]["classified"] + "/ndvi_class_from.tif"
    )
    classify_ndvi(
        ndvi_path=config["outputs"]["preprocessed"] + "/ndvi_to_clipped.tif",
        output_path=config["outputs"]["classified"] + "/ndvi_class_to.tif"
    )

    # Step 3: Change Detection
    from change_detection import detect_change
    detect_change(
        label_from_path=config["outputs"]["classified"] + "/ndvi_class_from.tif",
        label_to_path=config["outputs"]["classified"] + "/ndvi_class_to.tif",
        output_raster_path=config["paths"]["change_map"],
        stats_csv_path=config["outputs"]["analysis"] + "/luas_perubahan.csv"
    )

    # Step 4: Train and Predict
    train_and_predict(
        rgb_path=config["paths"]["rgb"],
        label_path=config["outputs"]["classified"] + "/ndvi_class_to.tif",
        boundary_path=config["paths"]["boundary"],
        model_output=config["outputs"]["model"],
        prediction_output=config["outputs"]["prediction"]
    )

    # Step 5: Evaluation
    evaluate_model(
        ground_truth_path=config["outputs"]["classified"] + "/ndvi_class_to.tif",
        prediction_path=config["outputs"]["prediction"],
        output_dir=config["outputs"]["evaluation"]
    )

    # Step 6: Analysis
    analyze_statistics(
        from_path=config["outputs"]["classified"] + "/ndvi_class_from.tif",
        to_path=config["outputs"]["classified"] + "/ndvi_class_to.tif",
        output_csv=config["outputs"]["analysis"] + "/luas_perubahan.csv"
    )

    # Step 7: Visualization
    generate_statistical_graph(
        csv_path=config["outputs"]["analysis"] + "/luas_perubahan.csv",
        output_path=config["outputs"]["visualization"] + "/grafik_perubahan.png"
    )

    # Step 8: Interactive Map
    create_interactive_map(
        tif_path=config["paths"]["change_map"],
        output_html=config["outputs"]["interactive_map"]
    )

    logger.info("Pipeline selesai dijalankan.")


if __name__ == "__main__":
    main()
