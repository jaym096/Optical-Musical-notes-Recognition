import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pylab as pyl
# import imageio
import sys


def getImages(fname):
    try:
        img = Image.open(fname).convert('L')
        return img
    except IOError:
        print("IOError")


def convolve1(img):
    """
    Part3: Convolution using arbitrary kernel
    """
    print("Part3: Convolution using arbitrary kernel")
    print("arbitrary kernel: blur filter")

    # Blur filter
    kernel = np.array([[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]]).reshape((3, 3))
    # kernel = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]).reshape((3, 3))

    # flip the kernel
    kernel = np.flipud(np.fliplr(kernel))

    # pad the image
    pad_image = np.ones((img.shape[0] + 2, img.shape[1] + 2))
    pad_image[1:-1, 1:-1] = img

    # output matrix
    output_con = np.ones(img.shape)

    # convolution
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output_con[i, j] = (kernel * pad_image[i:i + 3, j:j + 3]).sum()
    pyl.imshow(output_con, cmap="Greys_r")
    pyl.savefig("part3_result.png")


def convolve2(img):
    """
    Part4: Convolution using separable kernel
    """
    print("Part4: Convolution using separable kernel")
    print("arbitrary kernel: blur filter")

    # h1 = np.array([0.25, 0.5, 0.25]).reshape((3,1))
    # h2 = np.array([0.25, 0.5, 0.25]).reshape((1,3))
    h1 = np.array([0.333, 0.333, 0.333, 0.333, 0.333]).reshape((5, 1))
    h2 = np.array([0.333, 0.333, 0.333, 0.333, 0.333]).reshape((1, 5))

    # pad the image
    pad_image = np.zeros((img.shape[0] + 4, img.shape[1] + 4))
    pad_image[2:-2, 2:-2] = img

    # output matrix
    output_con = np.zeros(img.shape)

    # convolution using h1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output_con[i, j] = (h1 * pad_image[i:i + 5, j]).sum()

    # Convolution using h2
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            output_con[i, j] = (h2 * pad_image[i, j:j + 5]).sum()
    # pyl.show()
    pyl.imshow(output_con, cmap="Greys_r")
    pyl.savefig("part4_result.png")


def edge_detection(img):
    """
    detects edges of an image
    """
    # filter
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.transpose(kernel_x)

    # pad the image
    pad_image_x = np.ones((img.shape[0] + 2, img.shape[1] + 2))
    pad_image_x[1:-1, 1:-1] = img
    pad_image_y = np.ones((img.shape[0] + 2, img.shape[1] + 2))
    pad_image_y[1:-1, 1:-1] = img

    # output matrix
    output_con_x = np.ones(img.shape)
    output_con_y = np.ones(img.shape)
    output_con = np.ones(img.shape)

    # convolution
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output_con_x[i, j] = (kernel_x * pad_image_x[i:i + 3, j:j + 3]).sum()
            output_con_y[i, j] = (kernel_y * pad_image_y[i:i + 3, j:j + 3]).sum()
            output_con[i, j] = np.sqrt(np.power(output_con_x[i, j], 2) + np.power(output_con_y[i, j], 2))
            if output_con[i, j] < 2:
                output_con[i, j] = 0
            else:
                output_con[i, j] = 1
#    pyl.imshow(output_con, cmap="Greys_r")
#    pyl.show()
    return output_con

def edge_detection(img):
    """
    detects edges of an image
    """
    # filter
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.transpose(kernel_x)

    # pad the image
    pad_image_x = np.ones((img.shape[0] + 2, img.shape[1] + 2))
    pad_image_x[1:-1, 1:-1] = img
    pad_image_y = np.ones((img.shape[0] + 2, img.shape[1] + 2))
    pad_image_y[1:-1, 1:-1] = img

    # output matrix
    output_con_x = np.ones(img.shape)
    output_con_y = np.ones(img.shape)
    output_con = np.ones(img.shape)

    # convolution
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output_con_x[i, j] = (
                kernel_x * pad_image_x[i:i + 3, j:j + 3]
            ).sum()
            output_con_y[i, j] = (
                kernel_y * pad_image_y[i:i + 3, j:j + 3]
            ).sum()
            output_con[i, j] = np.sqrt(
                np.power(
                    output_con_x[i, j], 2) + np.power(output_con_y[i, j], 2
                )
            )
            if output_con[i, j] < 2:
                output_con[i, j] = 0
            else:
                output_con[i, j] = 1
    return output_con


def get_dist(i, k, a, b):
    return np.square(np.square(i - a) + np.square(k - b))

def getFulld(arr):
    """
    finds distance of nearest edge
    """
    response = np.empty(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            least_dist = float('inf')
            p, q = None, None
            for k in range(arr.shape[0]):
                for l in range(arr.shape[1]):
                    if arr[k][l] == 1:
                        d = get_dist(i, j, k, l)
                        if d < least_dist:
                            least_dist = d
                            p, q = k, l
            response[i][j] = least_dist
    return response

def getPixelsforBoxes(im):
    ggwp = np.min(im)
    th = 1.8
    boxworthypixels = []
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] < th * ggwp:
                flag = True
                # dont make double box
                for a in range(-3, 3):
                    for b in range(-3, 3):
                        if (i + a, j + b) in boxworthypixels:
                            flag = False
                if flag:
                    print(i, j)
                    print(im[i][j])
                    boxworthypixels.append((i, j))
    return boxworthypixels

def getGamma(val):
    if val:
        return 0
    return float('inf')

def getSimilarityMatrix(image, template):
    # template = np.flipud(np.fliplr(template))
    hshape = image.shape[0] - template.shape[0] + 1
    vshape = image.shape[1] - template.shape[1] + 1
    op = np.empty((hshape, vshape))

    for i in range(hshape):
        for j in range(vshape):
            op[i][j] = (
                image[i:i + template.shape[0], j:j + template.shape[1]] * template
            ).sum()
    m = np.min(op)
    return op


def getSpacing(im):
    l = im.shape[0]
    vote = np.zeros((l, l // 5 + 1))
    th = im.shape[1]
    result = list()

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] == 1:
                for spacing in range(1, l // 5 + 1):
                    for mul in range(5):
                        startLine = i - mul * spacing
                        if startLine >= 0:
                            vote[startLine][spacing] += 1

    startLine = -1
    tmpV = 0
    for i in range(vote.shape[0]):
        if np.sum(vote[i]) > tmpV:
            startLine = i
            tmpV = np.sum(vote[i])
    spacing = -1
    tmpV = 0
    for i in range(vote.shape[1]):
        if vote[startLine][i] > tmpV:
            spacing = i
            tmpV = vote[startLine][i]
    return startLine, spacing

def get_box_center(row,col,ht,wt):
    return row+int(ht)/2, col+int(wt)/2

def decide_note(center, imght, imgwt):
    pass

def pad_template(template):
    pad_temp = np.ones((template.shape[0] + 4, template.shape[1] + 4))
    pad_temp[2:-2, 2:-2] = template
    return pad_temp

def exe_omr(img, ntemp, qtemp, etemp):

    # standardize
    gray_img = np.asarray(img) / 255
    gray_ntemp = np.asarray(ntemp) / 255
    gray_qtemp = np.asarray(qtemp) / 255
    gray_etemp = np.asarray(etemp) / 255

    # pad the template
    gray_ntemp = pad_template(gray_ntemp)
    gray_qtemp = pad_template(gray_qtemp)
    gray_etemp = pad_template(gray_etemp)

    # Edge detection
    edge_map_img = edge_detection(gray_img)
    edge_map_ntemp = edge_detection(gray_ntemp)
    edge_map_qtemp = edge_detection(gray_qtemp)
    edge_map_etemp = edge_detection(gray_etemp)

    # spacing stores spaces between lines
    # startLine first line location
    startLine, spacing = getSpacing(edge_map_img)

    # kernel_ht stores sample kernel ht
    kernel_ht = gray_ntemp.shape[0]

    # resize ip image
    factor = kernel_ht / spacing
    ipimage = img.resize(
        (int(factor * img.size[0]), int(factor * img.size[1]))
    )

    img_dist_mat = getFulld(edge_map_img)

    nsim_mat = getSimilarityMatrix(img_dist_mat, edge_map_ntemp)
    npixels = getPixelsforBoxes(nsim_mat)
    qsim_mat = getSimilarityMatrix(img_dist_mat, edge_map_qtemp)
    qpixels = getPixelsforBoxes(qsim_mat)
    esim_mat = getSimilarityMatrix(img_dist_mat, edge_map_etemp)
    epixels = getPixelsforBoxes(esim_mat)

    imgc = img.convert('RGB')

    draw = ImageDraw.Draw(imgc)
    npix = ""
    qpix = ""
    epix = ""
    for pixel in npixels:
        draw.rectangle((
            (pixel[1], pixel[0]),
            (pixel[1] + gray_ntemp.shape[1], pixel[0] + gray_ntemp.shape[0])
        ), outline="red")
        npix += str(pixel[0]) + ", " + str(pixel[1]) + ", " + str(gray_ntemp.shape[0]) \
              + ", " + str(gray_ntemp.shape[1]) + ", " + "filled_note" + "\n"

    for pixel in qpixels:
        draw.rectangle([
            (pixel[1], pixel[0]),
            (pixel[1] + gray_qtemp.shape[1]),
            pixel[0] + gray_qtemp.shape[0]
        ], outline="green")
        qpix += str(pixel[0]) + ", " + str(pixel[1]) + ", " + str(gray_ntemp.shape[0]) \
                + ", " + str(gray_ntemp.shape[1]) + ", " + "Quarter_rest" + "\n"

    for pixel in epixels:
        draw.rectangle([
            (pixel[1], pixel[0]),
            (pixel[1] + gray_etemp.shape[1]),
            pixel[0] + gray_etemp.shape[0]
        ], outline="blue")
        epix += str(pixel[0]) + ", " + str(pixel[1]) + ", " + str(gray_ntemp.shape[0]) \
                + ", " + str(gray_ntemp.shape[1]) + ", " + "Eighth_rest" + "\n"

    out_file = open("detected.txt", "w")
    out_file.write(npix)
    out_file.write(qpix)
    out_file.write(epix)
    out_file.close()

    imgc.save("detected.png")


if __name__ == "__main__":
    """
    The program takes two parameters, the image and a number which represents which part of the assignment to run.
    1 ==> to run the part3 of the assignment
    2 ==> to run the part4 of the assignment
    3 ==> to run the rest of the part of the assignment (part 5 to 8).
    """
    # I/O
    part = int(sys.argv[2])
    path = "test-images/"
    filename = path + sys.argv[1]
    print("filename used: ", filename)

    # defining templates
    ntemplate = path + "template1.png"
    qtemplate = path + "template2.png"
    etemplate = path + "template3.png"

    if (part == 1):
        print("Executing part 3 of the assignment...")
        img = getImages(filename)
        gray_img = np.asarray(img) / 255  # standardize
        convolve1(gray_img)
    elif (part == 2):
        print("Executing part 4 of the assignment...")
        img = getImages(filename)
        gray_img = np.asarray(img) / 255  # standardize
        convolve2(gray_img)
    elif (part == 3):
        print("Executing OMR...")
        img = getImages(filename)
        ntemp = getImages(ntemplate)
        qtemp = getImages(qtemplate)
        etemp = getImages(etemplate)
        exe_omr(img, ntemp, qtemp, etemp)
    else:
        print("You've entered the wrong input please run the program again!!")