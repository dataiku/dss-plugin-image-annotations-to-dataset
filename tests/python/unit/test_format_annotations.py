# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information
import pandas as pd
import json
import io
from image_annotations_to_dataset.object_detection import create_dataframe_from_coco_json, retrieve_annotations_from_voc_xml


def test_create_dataframe_from_coco_json():
    images_folder_path = 'coco_aquarium/coco_aquarium/valid'
    coco_json_file_content = {
        "categories": [
            {"id": 3, "name": "fish"},
            {"id": 5, "name": "jellyfish"},
            {"id": 12, "name": "penguin"}
        ],
        "images": [
            {"id": 11, "file_name": "toto.jpg"},
            {"id": 12, "file_name": "tata.jpeg"}
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 11,
                "category_id": 5,
                "bbox": [
                    138,
                    316,
                    391,
                    364
                ],
                "area": 142324,
                "segmentation": [],
                "iscrowd": 0
            },
            {
                "id": 1,
                "image_id": 11,
                "category_id": 12,
                "bbox": [
                    126,
                    395,
                    94,
                    181
                ],
                "area": 17014,
                "segmentation": [],
                "iscrowd": 0
            },
            {
                "id": 2,
                "image_id": 12,
                "category_id": 5,
                "bbox": [
                    331,
                    761,
                    436,
                    163
                ],
                "area": 71068,
                "segmentation": [],
                "iscrowd": 0
            }]
    }

    expected_df = pd.DataFrame({
        "images_annotations": [
            json.dumps([
                {"category": "jellyfish", "bbox": [138, 316, 391, 364]},
                {"category": "penguin", "bbox": [126, 395, 94, 181]}
            ]),
            json.dumps([
                {"category": "jellyfish", "bbox": [331, 761, 436, 163]}
            ])
        ],
        "images_path": ['coco_aquarium/coco_aquarium/valid/toto.jpg', 'coco_aquarium/coco_aquarium/valid/tata.jpeg']
    })

    output_df = create_dataframe_from_coco_json(coco_json_file_content, images_folder_path)
    assert isinstance(output_df, pd.DataFrame)
    assert expected_df.equals(output_df)


def test_retrieve_annotations_from_voc_xml():

    with io.StringIO(
            """<annotation>
            <folder></folder>
            <filename>toto.jpg</filename>
            <object>
                <name>jellyfish</name>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>417</xmin>
                    <xmax>473</xmax>
                    <ymin>273</ymin>
                    <ymax>533</ymax>
                </bndbox>
            </object>
            <object>
                <name>fish</name>
                <difficult>1</difficult>
                <bndbox>
                    <xmin>1</xmin>
                    <xmax>294</xmax>
                    <ymin>18</ymin>
                    <ymax>193</ymax>
                </bndbox>
            </object>
            </annotation>"""
    ) as annotation_file_content:
        expected_image_annotations = [
            {"bbox": [416, 272, 56, 260], "category": "jellyfish"},
            {"bbox":  [0, 17, 293, 175], "category": "fish"}
        ]
        expected_image_filename = "toto.jpg"
        image_annotations, image_filename = retrieve_annotations_from_voc_xml(annotation_file_content)

        assert isinstance(image_annotations, list)
        assert expected_image_annotations == image_annotations
        assert expected_image_filename == image_filename
