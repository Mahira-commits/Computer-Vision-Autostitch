import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    # Get image dimensions
    rows, cols = img.shape[:2]

    # Define the corners of the original image
    corners = np.array([[0, 0, 1], [0, rows - 1, 1], [cols - 1, 0, 1], [cols - 1, rows - 1, 1]], dtype=np.float32)

    # Apply transformation to corners
    transformed_corners = np.dot(M, corners.T).T

    # Normalize transformed corners
    transformed_corners[:, 0] /= transformed_corners[:, 2]
    transformed_corners[:, 1] /= transformed_corners[:, 2]

    # Compute bounding box coordinates
    minX = int(np.min(transformed_corners[:, 0]))
    minY = int(np.min(transformed_corners[:, 1]))
    maxX = int(np.max(transformed_corners[:, 0]))
    maxY = int(np.max(transformed_corners[:, 1]))
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN

    print(f"Inside accumulateBlend \nimg.shape: {img.shape}")

    #Get image dimensions
    height, width, channels = img.shape
    invM = np.linalg.inv(M)

    minX, minY, maxX, maxY = imageBoundingBox(img, M)

    for i in range(minX, maxX):
        for j in range(minY, maxY):
            pixel = np.dot(invM, np.array([i, j, 1]))
            pixel = pixel / pixel[2]

            x, y = int(pixel[0]), int(pixel[1])

            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            x0, y0 = int(pixel[0]), int(pixel[1])
            x1, y1 = x - x0, y - y0

            if x0 < 0 or x0 >= width or y0 < 0 or y0 >= height:
                continue

            #Calculate value for linear interpolation
            interpolated_value = (1-x1) * (1-y1) * img[y0, x0] + \
                                x1 * (1-y1) * img[y0, min(x0+1, width-1)] + \
                                (1-x1) * y1 * img[min(y0+1, height-1), x0] + \
                                x1 * y1 * img[min(y0+1, height-1), min(x0+1, width-1)]
            
            #Check if pixel is black
            if np.all(interpolated_value == 0):
                continue
            
            #Calculate weight for blending
            # weight = min(min(x, width-x), blendWidth) * min(min(y, height-y), blendWidth)
            #For now, use constant weight
            weight = 1.0

            #Update accumulator
            acc[j, i, :3] += weight * interpolated_value
            acc[j, i, 3] += weight

    print(f"Exited accumulateBlend \nacc.shape: {acc.shape}")


    # Get image dimensions
    # rows, cols = acc.shape[:2]

    # # Create a grid of indices for the output image
    # y, x = np.indices((rows, cols))
    # indices = np.column_stack((x.ravel(), y.ravel(), np.ones_like(x.ravel())))

    # # Inverse warp indices to get corresponding coordinates in source image
    # inv_M = np.linalg.inv(M)
    # source_indices = np.dot(inv_M, indices.T).T

    # # Normalize coordinates
    # source_indices[:, 0] /= source_indices[:, 2]
    # source_indices[:, 1] /= source_indices[:, 2]

    # # Reshape to original shape
    # source_indices = source_indices[:, :2].reshape((rows, cols, 2))

    # # Compute weights for blending
    # weights = np.minimum(np.minimum(source_indices[:, :, 0], cols - source_indices[:, :, 0]) / blendWidth, 1)

    # # Initialize accumulator for weights
    # weight_acc = np.zeros_like(acc[:, :, :3])

    # # Loop over each pixel in the source image
    # for i in range(rows):
    #     for j in range(cols):
    #         # Skip if pixel is outside the source image or if it's black
    #         if (source_indices[i, j, 0] < 0 or source_indices[i, j, 0] >= img.shape[1] or
    #                 source_indices[i, j, 1] < 0 or source_indices[i, j, 1] >= img.shape[0] or
    #                 np.all(img[int(source_indices[i, j, 1]), int(source_indices[i, j, 0])] == 0)):
    #             continue

    #         # Compute coordinates in the source image
    #         src_x = int(source_indices[i, j, 0])
    #         src_y = int(source_indices[i, j, 1])

    #         # Blend colors using weights
    #         acc[i, j, :3] += weights[i, j] * img[src_y, src_x]
    #         weight_acc[i, j] += weights[i, j]

    # # Normalize accumulated colors by accumulated weights
    # acc[:, :, :3] /= weight_acc + 1e-6

    # # Update fourth channel of acc to record sum of weights
    # acc[:, :, 3] += weight_acc

    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    # Initialize the normalized image
    img = np.copy(acc)

    # Iterate over each pixel in the accumulated image
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            # Extract RGB values and alpha value of the pixel
            r, g, b, alpha = acc[i, j]

            # Check if alpha is greater than zero to avoid division by zero
            if alpha > 0:
                # Normalize RGB values by dividing by alpha
                img[i, j] = [r / alpha, g / alpha, b / alpha, alpha]
            else:
                # If alpha is zero, set RGB values to zero
                img[i, j] = [0, 0, 0, alpha]

            # Set alpha channel to opaque
            img[i, j, 3] = 255
    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        imgMinX, imgMinY, imgMaxX, imgMaxY = imageBoundingBox(img, M)
        minX = min(imgMinX, minX)
        minY = min(imgMinY, minY)
        maxX = max(imgMaxX, maxX)
        maxY = max(imgMaxY, maxY)
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    if is360:
        # Shift it left by the correct amount
        A[0, 2] = -width / 2

        # Then handle the vertical drift
        drift = (y_final - y_init)
        length = (x_final - x_init)
        if length != 0:
            A[1, 0] = -drift / length
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

