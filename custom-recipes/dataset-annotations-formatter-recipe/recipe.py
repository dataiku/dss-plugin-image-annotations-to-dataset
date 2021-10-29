# -*- coding: utf-8 -*-

import logging
import dataiku
import json

from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role


def format_annotations(image_annotations):
    image_annotations = json.loads(image_annotations)
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


input_dataset = dataiku.Dataset(get_input_names_for_role("input_dataset")[0])
output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])
parameters = get_recipe_config()

# copy all the columns from input dataset (only target one will be changed)
output_df = input_dataset.get_dataframe()
target_column = parameters.get("target_column")
if target_column not in output_df.columns:
    logging.error("column {} not in input df columns {}".format(target_column, output_df.columns))
    raise Exception("column {} not in input df {}".format(target_column, output_df))

logging.info("ML-assisted-labelling-plugin")
logging.info(target_column)
logging.info(output_df[target_column].head())
output_df = output_df
output_df[target_column] = output_df[target_column].map(format_annotations)
logging.info("after change:")
logging.info(output_df[target_column].head())

output_dataset.write_with_schema(output_df)
