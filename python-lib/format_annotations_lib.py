# -*- coding: utf-8 -*-
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
import logging


def create_dataset_df_from_coco_json_file(coco_json_file_content, images_folder_path):
    """
    :param coco_json_file_content: Dict from coco json file containing 3 sections:
                - "images" : mapping between images-ids and images filenames
                - "categories" : mapping between categories-ids and categories names
                - "annotations" : associates images-ids with categories-ids and gives the bounding boxes coordinates.
    :param images_folder_path: base path for the images (will be used as a prefix to create image full path)

    :return: Dataframe containing two columns : images_annotations and images_path (from input folder), 1 row per image.
             Format compatible with deephub object detection.
    """
    # build intermediate dicts to facilitate formatting:
    images_id_to_path = {img.get("id"): os.path.join(images_folder_path, img.get("file_name"))
                         for img in coco_json_file_content.get("images")}
    category_id_to_name = {cat.get("id"): cat.get("name") for cat in coco_json_file_content.get("categories")}

    annotations_per_img = defaultdict(list)
    for single_annotation in coco_json_file_content.get("annotations", []):
        # add category name to annotation dict
        single_annotation["category"] = category_id_to_name.get(single_annotation.get("category_id"))

        # a single image can have multiple annotations, add this one to the list (create a new list if needed)
        img_id = single_annotation.pop("image_id")
        annotations_per_img[img_id].append(single_annotation)
    return pd.DataFrame([{"images_annotations": json.dumps(annotations_per_img[img_id]), "images_path": img_path}
                         for img_id, img_path in images_id_to_path.items()])


def retrieve_annotations_from_voc_xml_file(annotation_file_content):
    """
    :param annotation_file_content: file-like object containing xml annotations for a single image
           It must contain: (see Pascal VOC format)
                - 'filename': name (including extension) of the image
                - list of 'object' with 'name' (category), 'difficult' and 'bndbox' (xmin, xmax, ymin, ymax)
    :return: tuple of image_annotations, image_filename where image_annotations is a list of dicts of the form:
            {"bbox": [xmin, ymin, width, height],
             "area": 14560,
             "iscrowd": False,
             "category": "jellyfish",
             "difficult": 0}
    """
    image_annotations = []
    tree = ET.parse(annotation_file_content)
    root = tree.getroot()
    image_filename = root.find('filename').text

    for o in root.iter('object'):

        bbox = o.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        height = ymax - ymin
        width = xmax - xmin

        image_annotations.append(
            {"bbox": [xmin, ymin, width, height],
             "area": height * width,  # to remove?
             "iscrowd": False,  # to remove?
             "category": o.find('name').text.lower().strip(),
             "difficult": int(o.find('difficult').text == '1')  # to remove?
             }
        )
    return image_annotations, image_filename


def create_dataset_df_from_voc_files(input_folder, images_folder_path, annotations_folder_path):
    """
        :param input_folder: DSS managed folder
        :param images_folder_path: base path for the images (will be used as a prefix to create image full path)
        :param annotations_folder_path: path of the folder containing all the xml annotations files

        :return: Dataframe containing two columns : images_annotations and images_path (from input folder), 1 row per image.
                 Format compatible with deephub object detection.
    """

    output_list = []
    # loop over annotations files from annotation folder & retrieve only xml files
    # (annotations & images files might be in the same folder)
    xml_annotations_files = [details
                             for details in input_folder.get_path_details(annotations_folder_path).get("children", [])
                             if details.get("mimeType", "") == "application/xml"]

    if len(xml_annotations_files) == 0:
        raise Exception("No annotation-xml file had been found in folder {}, stopping.".format(annotations_folder_path))

    for image_annotations_file in xml_annotations_files:
        with input_folder.get_download_stream(image_annotations_file.get("fullPath")) as annotations_file_stream:

            try:
                image_annotations, image_filename = retrieve_annotations_from_voc_xml_file(annotations_file_stream)
                output_list.append({
                    "images_annotations": json.dumps(image_annotations),
                    "images_path": os.path.join(images_folder_path, image_filename)
                })
            except ET.ParseError:
                logging.warning("XML file {} could not be parsed as an annotation file, skipping".format(
                    image_annotations_file.get("fullPath")
                ))

    if len(output_list) == 0:
        raise Exception("All the xml files found were badly formatted, stopping")
    return pd.DataFrame(output_list)
