# -*- coding: utf-8 -*-

import logging

import pandas as pd

import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role

# from lib import format_annotations

input_folder = dataiku.Folder(get_input_names_for_role("input_folder")[0])
logging.info(input_folder)

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

output_df = pd.DataFrame(columns=["annotations", "path"],
                         data=["toto", "toto/tata/titi.jpeg"])

output_dataset.write_with_schema(output_df)
