import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pylab as pyl
import imageio
import sys

def getImages(filename):
    path = "test-images/" + filename
    try:
        img = Image.open(path).convert('L')
        return img
    except IOError:
        print("IOError")

def convolve1(img):
    """
    Part3: Convolution using arbitrary kernel
    """
    # Blur filter
    kernel = np.array([[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]]).reshape((3, 3))

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

def pipeline(filename):

    ntemplate = "template1.png"
    qtemplate = "template2.png"
    etemplate = "template3.png"

    img = getImages(filename)
    ntemp = getImages(ntemplate)
    qtemp = getImages(qtemplate)
    etemp = getImages(etemplate)

    gray_img = np.asarray(img) / 255  # standardize
    gray_ntemp = np.asarray(ntemp) / 255
    gray_qtemp = np.asarray(qtemp) / 255
    gray_etemp = np.asarray(etemp) / 255

    # pad the template
    gray_ntemp = pad_template(gray_ntemp)
    gray_qtemp = pad_template(gray_qtemp)
    gray_etemp = pad_template(gray_etemp)

    edge_map_img = edge_detection(gray_img)

    startLine, spacing = getSpacing(edge_map_img)
    #spacing stores spaces between lines
    #startLine first line location

    kernel_ht = gray_ntemp.shape[0]
    #kernel_ht stores sample kernel ht

    #resize ip image
    factor = kernel_ht/spacing
    ipimage = img.resize(
        (int(factor*img.size[0]),int(factor*img.size[1]))
    )

    img_dist_mat = getFulld(edge_map_img)

    edge_map_ntemp = edge_detection(gray_ntemp)
    edge_map_qtemp = edge_detection(gray_qtemp)
    edge_map_etemp = edge_detection(gray_etemp)

    nsim_mat = getSimilarityMatrix(img_dist_mat, edge_map_ntemp)
    npixels = getPixelsforBoxes(nsim_mat)
    qsim_mat = getSimilarityMatrix(img_dist_mat, edge_map_qtemp)
    qpixels = getPixelsforBoxes(qsim_mat)
    esim_mat = getSimilarityMatrix(img_dist_mat, edge_map_etemp)
    epixels = getPixelsforBoxes(esim_mat)

    draw = ImageDraw.Draw(img)
    for pixel in npixels:
        draw.rectangle([
            (pixel[1], pixel[0]),
            (pixel[1] + gray_ntemp.shape[1]),
            pixel[0] + gray_ntemp.shape[0]
        ])

    for pixel in qpixels:
        draw.rectangle([
            (pixel[1], pixel[0]),
            (pixel[1] + gray_qtemp.shape[1]),
            pixel[0] + gray_qtemp.shape[0]
        ])

    for pixel in epixels:
        draw.rectangle([
            (pixel[1], pixel[0]),
            (pixel[1] + gray_etemp.shape[1]),
            pixel[0] + gray_etemp.shape[0]
        ])

    img.save("detected.png")



    # notes # array of note dicts
    # quarters # array of quareter rest dicts
    # eights # array of eight rests dicts

    # with open('detected.txt', 'w') as f:
    #     for note in notes:
    #         ll = "{row} {col} {height} {width} filled_note {pitch} {confidence}"
    #         f.write(ll.format(
    #             row=note['row'],
    #             col=note['col'],
    #             height=note['height'],
    #             width=note['width'],
    #             pitch=note['pitch'],
    #             confidence=note['confidence'],
    #         ))

    #     for quareter in quareters:
    #         ll = "{row} {col} {height} {width} quarter_rest _ {confidence}"
    #         f.write(ll.format(
    #             row=note['row'],
    #             col=note['col'],
    #             height=note['height'],
    #             width=note['width'],
    #             confidence=note['confidence'],
    #         ))

    #     for eight in eights:
    #         ll = "{row} {col} {height} {width} eight_rest _ {confidence}"
    #         f.write(ll.format(
    #             row=note['row'],
    #             col=note['col'],
    #             height=note['height'],
    #             width=note['width'],
    #             confidence=note['confidence'],
    #         ))

if __name__ == "__main__":
    filename = "music1crop.png"
    # filename2 = "template1.png"

    # img = getImages(filename)
    # temp = getImages(filename2)

    # gray_img = np.asarray(img) / 255  # standardize
    # gray_temp = np.asarray(temp) / 255

    # part3
    # convolve1(gray_img)

    # # part4
    # convolve2(gray_img)

    # part6
    pipeline(filename)
