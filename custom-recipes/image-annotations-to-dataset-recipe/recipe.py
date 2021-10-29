# -*- coding: utf-8 -*-

import logging
from collections import defaultdict

import pandas as pd
import os
import dataiku
import xml.etree.ElementTree as ET

import json

from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role


def retrieve_annotations_from_xml_file(annotation_file_stream):
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
images_folder_path = parameters.get("images_folder_path")

if input_data_format == "coco":
    logging.info("COCO format")

    annotations_file_content = input_folder.read_json(parameters.get("annotations_file_path"))

    # build intermediate dicts to facilitate formatting:
    images_id_to_path = {img.get("id"): os.path.join(images_folder_path, img.get("file_name"))
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

    output_df = pd.DataFrame([{"images_annotations": json.dumps(annotations_per_img[img_id]),
                               "images_path": img_path}
                              for img_id, img_path in images_id_to_path.items()])

elif input_data_format == "voc":
    logging.info("VOC pascal format")

    images_basename_to_fullpath = {os.path.splitext(img.get("name"))[0]: img.get("fullPath")
                                   for img in input_folder.get_path_details(images_folder_path).get("children")}

    output_list = []
    # loop over annotations files from annotation folder:
    for image_annotations_file in input_folder.get_path_details(parameters.get("annotations_folder_path")).get("children"):
        xml_fullpath = image_annotations_file.get("fullPath")
        image_basename = os.path.splitext(image_annotations_file.get("name"))[0]
        if image_basename not in images_basename_to_fullpath:
            raise Exception("No image file corresponding to annotations file {} in folder {}".format(
                xml_fullpath, images_folder_path
            ))
        with input_folder.get_download_stream(xml_fullpath) as annotations_file_stream:
            output_list.append({
                "images_annotations": json.dumps(retrieve_annotations_from_xml_file(annotations_file_stream)),
                "images_path": images_basename_to_fullpath.get(image_basename)
            })
    output_df = pd.DataFrame(output_list)

else:
    raise Exception("Input format unknown: {}".format(input_data_format))

output_dataset.write_with_schema(output_df)
