# -*- coding: utf-8 -*-

import logging
import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role
from format_annotations_lib import create_dataset_df_from_coco_json_file, create_dataset_df_from_voc_files

input_folder = dataiku.Folder(get_input_names_for_role("input_folder")[0])
parameters = get_recipe_config()
input_data_format, images_folder_path = parameters["input_data_format"], parameters["images_folder_path"]

if input_data_format == "coco":
    logging.info("Annotations format: COCO format")

    coco_json_file = input_folder.read_json(parameters["coco_json_filepath"])
    output_df = create_dataset_df_from_coco_json_file(coco_json_file, images_folder_path)

elif input_data_format == "voc":
    logging.info("Annotations format: VOC pascal format")
    output_df = create_dataset_df_from_voc_files(input_folder, images_folder_path,
                                                 parameters["annotations_folder_path"])

else:
    raise Exception("Annotations format unknown: {}".format(input_data_format))

output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])
output_dataset.write_with_schema(output_df)
