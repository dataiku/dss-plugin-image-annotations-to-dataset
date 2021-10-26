# -*- coding: utf-8 -*-

import logging

import pandas as pd

import dataiku

from dataiku.core import dkujson
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role


# from lib import format_annotations
def generate_path_list(folder: dataiku.Folder):
    """Generate a dataframe of file paths in a Dataiku Folder matching a list of extensions
    Args:
        folder: Dataiku managed folder where files are stored
            This folder can be partitioned or not, this function handles both
        file_extensions: list of file extensions to match, ex: ["JPG", "PNG"]
            Expected format is not case-sensitive but should not include leading "."
        path_column: Name of the column in the output dataframe
    Returns:
        DataFrame with one column named `path_column` with all the file paths matching the list of `file_extensions`
    Raises:
        RuntimeError: If there are not files matching the list of `file_extensions`
    """
    path_list = []
    if folder.read_partitions:
        for partition in folder.read_partitions:
            path_list += folder.list_paths_in_partition(partition)
    else:
        path_list = folder.list_paths_in_partition()
        logging.info(path_list)

    return [folder.get_path_details(path) for path in path_list]


input_folder = dataiku.Folder(get_input_names_for_role("input_folder")[0])
logging.info(generate_path_list(input_folder))

output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])
logging.info(output_dataset)

parameters = get_recipe_config()
data_format = parameters.get("data_format")

if data_format == "coco":
    logging.info("coco detected")
    image_folder = parameters.get("image_folder")
    logging.info(image_folder)

    annotations_file_path = parameters.get("annotations_file_path")
    logging.info(annotations_file_path)
    annotations = dkujson.load_from_filepath(annotations_file_path)
    logging.info(annotations)

output_df = pd.DataFrame([{"annotations": "toto", "path": "toto/tata/titi.jpeg"}])

output_dataset.write_with_schema(output_df)
