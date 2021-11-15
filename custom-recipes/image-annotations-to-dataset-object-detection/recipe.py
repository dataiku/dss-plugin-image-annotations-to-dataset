# -*- coding: utf-8 -*-

import logging
import dataiku
from dataiku.base.utils import safe_exception
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role
from image_annotations_to_dataset.object_detection import create_dataframe_from_coco_json, create_dataframe_from_voc_files

logger = logging.getLogger(__name__)

input_folder = dataiku.Folder(get_input_names_for_role("input_folder")[0])
parameters = get_recipe_config()
input_data_format, images_folder_path = parameters["input_data_format"], parameters.get("images_folder_path", "")

if not input_folder.get_path_details(images_folder_path)["exists"]:
    raise safe_exception(Exception, "Images path '{}' not found in input folder".format(images_folder_path))

if input_data_format == "coco":
    logger.info("Annotations format: COCO format")

    if not input_folder.get_path_details(parameters["coco_json_filepath"])["exists"]:
        raise safe_exception(Exception,
                             "Annotation file '{}' not found in folder".format(parameters["coco_json_filepath"]))
    coco_json_file = input_folder.read_json(parameters["coco_json_filepath"])

    output_df = create_dataframe_from_coco_json(coco_json_file, images_folder_path)

elif input_data_format == "voc":
    logger.info("Annotations format: VOC pascal format")

    annotations_folder_details = input_folder.get_path_details(parameters["voc_annotations_folder_path"])
    if not annotations_folder_details['exists'] or not annotations_folder_details['directory']:
        raise safe_exception(Exception, "Annotation folder path '{}' not found in input folder or not a folder itself"
                                        .format(parameters["voc_annotations_folder_path"]))

    output_df = create_dataframe_from_voc_files(input_folder, images_folder_path, annotations_folder_details)

else:
    raise safe_exception(Exception, "Annotations format unknown: {}".format(input_data_format))

output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])
output_dataset.write_with_schema(output_df)
