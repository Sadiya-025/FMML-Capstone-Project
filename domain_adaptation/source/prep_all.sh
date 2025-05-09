#! /bin/bash

PYTHON=/c/Users/Lapto/Anaconda3/python

bdd_dd='./domain_adaptation/source/datasets/bdd100k/seg/'
mapillary_dd='./domain_adaptation/source/datasets/mapillary-vistas-dataset_public_v1.1/'
gta_dd='./domain_adaptation/source/datasets/gta/'
cityscapes_dd='./domain_adaptation/source/datasets/cityscapes/'

sd='./domain_adaptation/source/source_datasets_dir/'
mkdir -p sd

$PYTHON ./domain_adaptation/source/core/cityscapes.py ${cityscapes_dd} ${sd}
$PYTHON ./domain_adaptation/source/core/mapillary.py ${mapillary_dd} ${sd}
$PYTHON ./domain_adaptation/source/core/gta.py ${gta_dd} ${sd}
$PYTHON ./domain_adaptation/source/core/bdds.py ${bdd_dd} ${sd}