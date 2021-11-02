# -*- coding: utf-8 -*-
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import logging

import numpy as np
import pandas as pd


def format_labelling_plugin_annotations(image_annotations):
    # images can be skipped in labelling plugin, in which case their annotations is nan
    image_annotations = json.loads(image_annotations) if isinstance(image_annotations, str) else []
    deephub_image_annotations = []
    for annotation in image_annotations:
        deephub_image_annotations.append(
            {"bbox": [annotation.get("left"), annotation.get("top"), annotation.get("width"), annotation.get("height")],
             "area": annotation.get("height") * annotation.get("width"),
             "iscrowd": False,
             "category": annotation.get("label")
             })
    image_annotations = json.dumps(deephub_image_annotations)
    return image_annotations


def create_dataset_df_from_coco_json_file(coco_json_file, images_folder_path):
    # build intermediate dicts to facilitate formatting:
    images_id_to_path = {img.get("id"): os.path.join(images_folder_path, img.get("file_name"))
                         for img in coco_json_file.get("images")}
    category_id_to_name = {cat.get("id"): cat.get("name") for cat in coco_json_file.get("categories")}

    annotations_per_img = defaultdict(list)
    for single_annotation in coco_json_file.get("annotations", []):
        # add category name to annotation dict
        single_annotation["category"] = category_id_to_name.get(single_annotation.get("category_id"))

        # a single image can have multiple annotations, add this one to the list (create a new list if needed)
        img_id = single_annotation.pop("image_id")
        annotations_per_img[img_id].append(single_annotation)
    return pd.DataFrame([{"images_annotations": json.dumps(annotations_per_img[img_id]), "images_path": img_path}
                         for img_id, img_path in images_id_to_path.items()])


def get_basename(path_details_dict):
    return os.path.splitext(path_details_dict.get("name"))[0]

def retrieve_annotations_from_voc_xml_file(annotation_file):
    image_annotations = []
    tree = ET.parse(annotation_file)
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

def create_dataset_df_from_voc_files(input_folder, images_folder_path, annotations_folder_path):
    # build intermediate dicts to facilitate formatting:
    images_basename_to_fullpath = {get_basename(img_path): img_path.get("fullPath")
                                   for img_path in input_folder.get_path_details(images_folder_path).get("children")}

    output_list = []
    # loop over annotations files from annotation folder:
    children_files = input_folder.get_path_details(annotations_folder_path).get("children", [])
    # retrieve only xml files (annotations & images files might be in the same folder)
    xml_annotations_files = [filedetails for filedetails in children_files
                             if filedetails.get("mimeType", "") == "application/xml"]
    for image_annotations_file in xml_annotations_files:
        xml_fullpath = image_annotations_file.get("fullPath")
        image_basename = get_basename(image_annotations_file)

        if image_basename not in images_basename_to_fullpath:
            raise Exception("No image file corresponding to annotations file {} in folder {}".format(
                xml_fullpath, images_folder_path
            ))
        with input_folder.get_download_stream(xml_fullpath) as annotations_file_stream:
            output_list.append({
                "images_annotations": json.dumps(retrieve_annotations_from_voc_xml_file(annotations_file_stream)),
                "images_path": images_basename_to_fullpath.get(image_basename)
            })
    return pd.DataFrame(output_list)
