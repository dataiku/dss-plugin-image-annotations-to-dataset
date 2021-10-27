# -*- coding: utf-8 -*-

import logging
from collections import defaultdict

import pandas as pd
import os
import dataiku
import xml.etree.ElementTree as ET

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
        path_list = folder.list_paths_in_partition()  # ['/coco val2017/coco val2017/000000182611.jpg',  ... '/coco val2017.zip'] liste TOUT en relatif depuis racine
        logging.info(path_list)

    return [folder.get_path_details(path) for path in path_list]

def read_voc_annotation_file(annotation_file_stream):
    image_annotations = []
    tree = ET.parse(annotation_file_stream)
    root = tree.getroot()

    for o in root.iter('object'):
        difficult = int(o.find('difficult').text == '1')
        category = o.find('name').text.lower().strip()

        bbox = o.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        height = ymax - ymin
        width = xmax - xmin

        image_annotations.append(
            {"bbox": [xmin, ymin, width, height],
             "area": height * width,
             "iscrowd": False,
             "category": category,
             "difficult": difficult}
        )
    return image_annotations


input_folder = dataiku.Folder(get_input_names_for_role("input_folder")[0])
output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])

parameters = get_recipe_config()
input_data_format = parameters.get("input_data_format")
image_folder_path = parameters.get("image_folder_path")

if input_data_format == "coco":
    logging.info("COCO format")

    # annotations = dkujson.load_from_filepath(annotations_file_path) # ne marche qu'en local avec path complet.
    # ou alors fich = get_download_stream(filepath) puis dkujson.load_from_filepath(fich)?
    annotations_file_content = input_folder.read_json(parameters.get("annotations_file_path"))

    # build intermediate dicts to facilitate formatting:
    images_id_to_path = {img.get("id"): os.path.join(image_folder_path, img.get("file_name"))
                         for img in annotations_file_content.get("images")}
    category_id_to_name = {cat.get("id"): cat.get("name") for cat in annotations_file_content.get("categories")}

    single_annotations = annotations_file_content.get("annotations")
    annotations_per_img = defaultdict(list)
    for single_annotation in single_annotations:
        # add category name to annotation dict
        single_annotation["category"] = category_id_to_name.get(single_annotation.get("category_id"))

        # a single image can have multiple annotations, add this one to the list (create a new list if needed)
        img_id = single_annotation.pop("image_id")
        annotations_per_img[img_id].append(single_annotation)

    output_df = pd.DataFrame([{"target": annotations_per_img[img_id],
                               "file_path": img_path}
                              for img_id, img_path in images_id_to_path.items()])

elif input_data_format == "voc":
    logging.info("VOC pascal format")
    folder_path_details = input_folder.get_path_details(path=parameters.get("annotations_folder_path"))

    output_list = []
    for image_annotations_file in folder_path_details.get("children"):
        with input_folder.get_download_stream(image_annotations_file.get("fullPath")) as annotations_file_stream:
            output_list.append({
                "target": read_voc_annotation_file(annotations_file_stream),
                "file_path": image_annotations_file.get("fullPath")
            })
            logging.info("Last row included: ")
            logging.info(output_list[-1])
    output_df = pd.DataFrame(output_list)

else:
    raise Exception("Input format unknown: {}".format(input_data_format))

output_dataset.write_with_schema(output_df)
