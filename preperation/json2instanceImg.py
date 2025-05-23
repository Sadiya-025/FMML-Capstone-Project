#!/usr/bin/python
#
# Reads labels as polygons in JSON format and converts them to instance images,
# where each pixel has an ID that represents the ground truth class and the
# individual instance of that class.
#
# The pixel values encode both, class and the individual instance.
# The integer part of a division by 1000 of each ID provides the class ID,
# as described in labels.py. The remainder is the instance ID. If a certain
# annotation describes multiple instances, then the pixels have the regular
# ID of that class.
#
# Example:
# Let's say your labels.py assigns the ID 26 to the class 'car'.
# Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .
# A group of cars, where our annotators could not identify the individual
# instances anymore, is assigned to the ID 26.
#
# Note that not all classes distinguish instances (see labels.py for a full list).
# The classes without instance annotations are always directly encoded with
# their regular ID, e.g. 11 for 'building'.
#
# Usage: json2instanceImg.py [OPTIONS] <input json> <output image>
# Options:
#   -h   print a little help text
#   -t   use train IDs
#
# Can also be used by including as a module.
#
# Uses the mapping defined in 'labels.py'.
#
# See also createTrainIdInstanceImgs.py to apply the mapping to all annotations in Cityscapes.
#
# python imports
import os
import sys

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'helpers')))

from anue_labels import labels, name2label
from annotation import Annotation
import getopt
from tqdm import tqdm

# Image processing
# Check if PIL is actually Pillow as expected
try:
    from PIL import Image
except ImportError:
    print("Please install the module 'Pillow' for image processing, e.g.")
    print("pip install pillow")
    sys.exit(-1)

try:
    import PIL.Image as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

# print(sys.path)

# Print the information

def printHelp():
    print('{} [OPTIONS] inputJson outputImg'.format(
        os.path.basename(sys.argv[0])))
    print('')
    print(' Reads labels as polygons in JSON format and converts them to instance images,')
    print(' where each pixel has an ID that represents the ground truth class and the')
    print(' individual instance of that class.')
    print('')
    print(' The pixel values encode both, class and the individual instance.')
    print(' The integer part of a division by 1000 of each ID provides the class ID,')
    print(' as described in labels.py. The remainder is the instance ID. If a certain')
    print(' annotation describes multiple instances, then the pixels have the regular')
    print(' ID of that class.')
    print('')
    print(' Example:')
    print(' Let\'s say your labels.py assigns the ID 26 to the class "car".')
    print(' Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .')
    print(' A group of cars, where our annotators could not identify the individual')
    print(' instances anymore, is assigned to the ID 26.')
    print('')
    print(' Note that not all classes distinguish instances (see labels.py for a full list).')
    print(' The classes without instance annotations are always directly encoded with')
    print(' their regular ID, e.g. 11 for "building".')
    print('')
    print('Options:')
    print(' -h                 Print this help')
    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')

# Print an error message and quit


def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    printHelp()
    sys.exit(-1)

# Convert the given annotation to a label image


def createInstanceImage(inJson,annotation, encoding):
    # the size of the image
    size = (annotation.imgWidth, annotation.imgHeight)

    # the background
    if encoding == "id":
        backgroundId = name2label['unlabeled'].id
    elif encoding == "csId":
        backgroundId = name2label['unlabeled'].csId
    elif encoding == "csTrainId":
        backgroundId = name2label['unlabeled'].csTrainId
    elif encoding == "level4Id":
        backgroundId = name2label['unlabeled'].level4Id
    elif encoding == "level3Id":
        backgroundId = name2label['unlabeled'].level3Id
    elif encoding == "level2Id":
        backgroundId = name2label['unlabeled'].level2Id
    elif encoding == "level1Id":
        backgroundId = name2label['unlabeled'].level1Id
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    instanceImg = Image.new("I", size, backgroundId)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw(instanceImg)

    # a dict where we keep track of the number of instances that
    # we already saw of each class
    nbInstances = {}
    for labelTuple in labels:
        if labelTuple.hasInstances:
            nbInstances[labelTuple.level3Id] = 0

    # loop over all objects
    for obj in annotation.objects:
        label = obj.label
        # if label == 'person':
        #     print "person"
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted or len(polygon) < 2:
            continue

        # if the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        # also we know that this polygon describes a group
        isGroup = False
        if (not label in name2label) and label.endswith('group'):
            label = label[:-len('group')]
            isGroup = True

        if not label in name2label:
            print("Label '{}' not known.".format(label))
            tqdm.write("Something wrong in: " + inJson)
            continue

        # the label tuple
        labelTuple = name2label[label]

        # get the class ID
        if encoding == "id":
            id = labelTuple.id
        elif encoding == "csId":
            id = labelTuple.csId
        elif encoding == "csTrainId":
            id = labelTuple.csTrainId
        elif encoding == "level4Id":
            id = labelTuple.level4Id
        elif encoding == "level3Id":
            id = labelTuple.level3Id
        elif encoding == "level2Id":
            id = labelTuple.level2Id
        elif encoding == "level1Id":
            id = labelTuple.level1Id

        # if this label distinguishs between invidudial instances,
        # make the id a instance ID
        if labelTuple.hasInstances and not isGroup:
            id = id * 1000 + nbInstances[labelTuple.level3Id]
            nbInstances[labelTuple.level3Id] += 1

        # If the ID is negative that polygon should not be drawn
        if id < 0:
            continue

        # print 'id is ', id

        try:
            # if id > 24000 and id < 25000:
                # print id
            drawer.polygon(polygon, fill=id)
        except:
            print("Failed to draw polygon with label {} and id {}: {}".format(
                label, id, polygon))
            raise

    return instanceImg

# A method that does all the work
# inJson is the filename of the json file
# outImg is the filename of the instance image that is generated
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs


def json2instanceImg(inJson, outImg, encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(inJson)
    instanceImg = createInstanceImage(inJson, annotation, encoding)
    instanceImg.save(outImg)

# The main method, if you execute this script directly
# Reads the command line arguments and calls the method 'json2instanceImg'


def main(argv):
    trainIds = False
    try:
        opts, args = getopt.getopt(argv, "ht")
    except getopt.GetoptError:
        printError('Invalid arguments')
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit(0)
        elif opt == '-t':
            trainIds = True
        else:
            printError("Handling of argument '{}' not implementend".format(opt))

    if len(args) == 0:
        printError("Missing input json file")
    elif len(args) == 1:
        printError("Missing output image filename")
    elif len(args) > 2:
        printError("Too many arguments")

    inJson = args[0]
    outImg = args[1]

    if trainIds:
        json2instanceImg(inJson, outImg, 'trainIds')
    else:
        json2instanceImg(inJson, outImg)


# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])
