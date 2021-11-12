# -*- coding: utf-8 -*-
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
import logging


def create_dataframe_from_coco_json(coco_json_file_content, images_folder_path):
    """
    :param coco_json_file_content: Dict from coco json file containing 3 sections:
                - "images" : mapping between images-ids and images filenames
                - "categories" : mapping between categories-ids and categories names
                - "annotations" : associates images-ids with categories-ids and gives the bounding boxes coordinates.
                see https://cocodataset.org/#format-data for more details on COCO format.
    :param images_folder_path: base path for the images (will be used as a prefix to create image full path)

    :return: Dataframe containing two columns : images_annotations and images_path (from input folder), 1 row per image.
             Format compatible with deephub object detection.
    """
    # build intermediate dicts to facilitate formatting:
    images_id_to_path = {img["id"]: images_folder_path + "/" + img["file_name"]
                         for img in coco_json_file_content["images"]}
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_json_file_content["categories"]}

    annotations_per_img = defaultdict(list)
    for single_annotation in coco_json_file_content["annotations"]:
        # a single image can have multiple annotations, add this one to the list (create a new list if needed)
        annotations_per_img[single_annotation.pop("image_id")].append({
            "category": category_id_to_name[single_annotation["category_id"]],
            "bbox": single_annotation["bbox"]
        })
    return pd.DataFrame([{"images_annotations": json.dumps(annotations_per_img[img_id]), "images_path": img_path}
                         for img_id, img_path in images_id_to_path.items()])


def retrieve_annotations_from_voc_xml_file(annotation_file_content):
    """
    :param annotation_file_content: file-like object containing xml annotations for a single image
           It must contain:
                - 'filename': name (including extension) of the image
                - list of 'object' with 'name' (category), and 'bndbox' (xmin, xmax, ymin, ymax)
           see Pascal VOC format for object detection:
           http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000
    :return: tuple of image_annotations, image_filename where image_annotations is a list of dicts of the form:
            {"bbox": [xmin, ymin, width, height]
             "category": "jellyfish" }
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

        image_annotations.append({
            "bbox": [xmin, ymin, width, height],
            "category": o.find('name').text.lower().strip()
        })
    return image_annotations, image_filename


def create_dataframe_from_voc_files(input_folder, images_folder_path, annotations_folder_details):
    """
        :param input_folder: DSS managed folder
        :param images_folder_path: base path for the images (will be used as a prefix to create image full path)
        :param annotations_folder_details: dict containing the path of annotations folder and a list of one dict per
        folder's child, containing details on file type and its fullPath: it will be used to iterate over annotation files.

        :return: Dataframe containing two columns : images_annotations and images_path (from input folder), 1 row per image.
                 Format compatible with deephub object detection.
    """

    output_list = []

    # loop over files from annotation folder & retrieve only XML files: annotations & images might be in the same folder
    xml_annotations_files = [details for details in annotations_folder_details["children"]
                             if details.get("mimeType", "") == "application/xml"]
    if len(xml_annotations_files) == 0:
        raise Exception("No annotation-XML file had been found in folder {}."
                        .format(annotations_folder_details.get("fullPath")))

    for annotations_file in xml_annotations_files:
        with input_folder.get_download_stream(annotations_file.get("fullPath")) as annotations_file_stream:
            try:
                image_annotations, image_filename = retrieve_annotations_from_voc_xml_file(annotations_file_stream)
                output_list.append({
                    "images_annotations": json.dumps(image_annotations),
                    "images_path": images_folder_path + "/" + image_filename
                })
            except (ET.ParseError, AttributeError) as e:
                logging.exception("XML file {} could not be parsed as an annotation file, skipping"
                                  .format(annotations_file.get("fullPath")))

    if len(output_list) == 0:
        raise Exception("All the XML files found were badly formatted.")
    return pd.DataFrame(output_list)
