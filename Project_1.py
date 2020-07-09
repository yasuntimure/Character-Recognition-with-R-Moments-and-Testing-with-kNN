from PIL import *
from PIL import Image
import numpy as np
import math
from PIL import Image,ImageDraw
from PIL import Image
from scipy import stats
import numpy as np


def main():
    img = Image.open('image.png')
    img_gray = img.convert('L')  # converts the image to grayscale image
    # img_bin = img.convert('1') #converts to a binary image, T=128, LOW=0, HIGH=255
    #img_gray.show()
    ONE = 150
    a = np.asarray(img_gray)  # from PIL to np array
    a_bin = threshold(a,100,ONE,0)
    im = Image.fromarray(a_bin)  # from np array to PIL format
    #im.show()
    #a_bin = binary_image(100,100,ONE)  # creates a binary image
    label, im = blob_coloring_8_connected(a_bin, ONE)
    new_img = np2PIL_color(label)
    #new_img.show()
    k_values = locateRectangle(im)
    newValues = resizeRectengle(k_values)
    #drawRectengle(k_values, new_img)
    #huMoments = cropImage(k_values,new_img)
    cropImage(k_values,new_img)



def binary_image(nrow,ncol,Value):
    x,y = np.indices((nrow,ncol))
    mask_lines = np.zeros(shape=(nrow,ncol))

    x0,y0,r0 = 30,30,10
    x1,y1,r1 = 70,30,10

    for i in range(50,70):
        mask_lines[i][i] = 1
        mask_lines[i][i + 1] = 1
        mask_lines[i][i + 2] = 1
        mask_lines[i][i + 3] = 1
        mask_lines[i][i + 6] = 1
        mask_lines[i - 20][90 - i + 1] = 1
        mask_lines[i - 20][90 - i + 2] = 1
        mask_lines[i - 20][90 - i + 3] = 1

    # mask_circle1 = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
    mask_square1 = np.fmax(np.absolute(x - x1),np.absolute(y - y1)) <= r1
    # mask_square2 = np.fmax(np.absolute( x - x2), np.absolute( y - y2)) <= r2
    # mask_square3 = np.fmax(np.absolute( x - x3), np.absolute( y - y3)) <= r3
    # mask_square4 =  np.fmax(np.absolute( x - x4), np.absolute( y - y4)) <= r4
    # imge = np.logical_or ( np.logical_or(mask_lines, mask_circle1), mask_square1) * Value
    imge = np.logical_or(mask_lines,mask_square1) * Value
    # imge = np.logical_or(mask_lines, mask_circle1) * Value

    return imge


def np2PIL(im):
    print("size of arr: ",im.shape)
    img = Image.fromarray(im,'RGB')
    return img


def np2PIL_color(im):
    print("size of arr: ",im.shape)
    img = Image.fromarray(np.uint8(im))
    return img


def threshold(im,T,LOW,HIGH):
    (nrows,ncols) = im.shape
    im_out = np.zeros(shape=im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) < T:
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out


def blob_coloring_8_connected(bim, ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    #print("nrow, ncol",nrow, ncol)
    im = np.zeros(shape=(nrow, ncol),dtype=int)
    a = np.zeros(shape=max_label,dtype=int)
    a = np.arange(0,max_label,dtype=int)
    color_map = np.zeros(shape=(max_label,3),dtype=np.uint8)
    color_im = np.zeros(shape=(nrow,ncol,3),dtype=np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0,255,1,dtype=np.uint8)
        color_map[i][1] = np.random.randint(0,255,1,dtype=np.uint8)
        color_map[i][2] = np.random.randint(0,255,1,dtype=np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1,nrow - 1):
        for j in range(1,ncol - 1):
            c = bim[i][j]
            l = bim[i][j - 1]
            u = bim[i - 1][j]
            lu = bim[i - 1][j - 1]
            ru = bim[i - 1][j + 1]
            label_u = im[i - 1][j]
            label_l = im[i][j - 1]
            label_lu = im[i - 1][j - 1]
            label_ru = im[i - 1][j + 1]
            im[i][j] = max_label
            if c == ONE:
                min_label = min(label_u, label_l, label_lu, label_ru)
                if min_label == max_label:
                    k += 1
                    im[i][j] = k
                else:
                    im[i][j] = min_label
                    if min_label != label_u and label_u != max_label:
                        update_array(a, min_label, label_u)
                    if min_label != label_l and label_l != max_label:
                        update_array(a, min_label, label_l)
                    if min_label != label_lu and label_lu != max_label:
                        update_array(a, min_label, label_lu)
                    if min_label != label_ru and label_ru != max_label:
                        update_array(a, min_label, label_ru)
            else:
                im[i][j] = max_label

    # final reduction in label array
    for i in range(k + 1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    # second pass to resolve labels and show label colors
    for i in range(nrow):
        for j in range(ncol):
            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j],0]
                color_im[i][j][1] = color_map[im[i][j],1]
                color_im[i][j][2] = color_map[im[i][j],2]
    return color_im,im


def locateRectangle(im):
    k_values=[]
    nrow=len(im)
    ncol=len(im[0])
    labels=[10000]

    for i in range(nrow):
        for j in range(ncol):
            if im[i][j] not in labels:
                mini=nrow
                minj=ncol
                maxi=0
                maxj=0
                k=im[i][j]
                for i in range (nrow):
                    for j in range (ncol):
                        if k== im[i][j]:
                            if mini>i:
                                mini=i
                            if minj>j:
                                minj=j
                            if maxi<i:
                                maxi=i
                            if maxj<j:
                                maxj=j
                k_values.append([k,mini,minj,maxi,maxj])
                labels.append(k)

    #print("Labels", labels)

    return k_values


def drawRectengle(k_values, new_img):
    # importing image object from PIL
    from PIL import Image,ImageDraw
    for i in range(len(k_values)):
        mini = k_values[i][1]
        minj = k_values[i][2]
        maxi = k_values[i][3]
        maxj = k_values[i][4]
        print("k values"," mini:",mini," minj:",minj," maxi:",maxi," maxj:",maxj)
        draw = ImageDraw.Draw(new_img)
        draw.rectangle((minj,mini,maxj,maxi),outline=(255,0,0))
        new_im=new_img
        new_im.save('image_new.png',quality=95)
        new_im.show()
    return mini

def resizeRectengle(k_values): #resizes rectengles to square
    newValues=[]
    croppedImages=[]
    for i in range(len(k_values)):
        mini = k_values[i][1]
        minj = k_values[i][2]
        maxi = k_values[i][3]
        maxj = k_values[i][4]
        if(abs(maxi-mini) > abs(maxj-minj)):
            newMinj= (minj+maxj)/2 - abs(mini-maxi)/2
            newMaxj= (minj+maxj)/2 + abs(mini-maxi)/2
            newMini=mini
            newMaxi=maxi
        elif(abs(mini-maxi) < abs(minj-maxj)):
            newMini = (mini+maxi)/2 - abs(minj - maxj)/2
            newMaxi = (mini+maxi)/2 + abs(minj - maxj)/2
            newMinj = minj
            newMaxj = maxj
        else:
            continue
        newValues.append([i,newMini,newMinj,newMaxi,newMaxj])

    return newValues

def normalizedMoments(p,q,Up0q0,Upq):
    y = (p+q)/2+1
    n = Upq/(Up0q0 ** y)
    return n

def cropImage(newValues,new_img):
    croppedImages = []
    for i in range(len(newValues)):
        mini = newValues[i][1]
        minj = newValues[i][2]
        maxi = newValues[i][3]
        maxj = newValues[i][4]
        try:
            area = (minj,mini,maxj,maxi)
            img = new_img.crop(area)
            croppedImages.append([img])
            #img.show()
            #img.save("cropped_picture.jpg")
        except IOError:
            pass
        img_gray = img.convert('L')  # converts the image to grayscale image
        img_gray.show()
        ONE = 150
        a = np.asarray(img_gray)  # from PIL to np array
        a_bin = threshold(a,100,ONE,0)
        img = Image.fromarray(a_bin)  # from np array to PIL format
    #     p0 = stats.moment(img,moment=0)
    #     p1 = stats.moment(img,moment=1)
    #     p2 = stats.moment(img,moment=2)
    #     p3 = stats.moment(img,moment=3)
    #     q0 = stats.moment(img,moment=0)
    #     q1 = stats.moment(img,moment=1)
    #     q2 = stats.moment(img,moment=2)
    #     q3 = stats.moment(img,moment=3)
    #     img = a_bin
    #     for i in range(len(img)):
    #         for j in range(len(img[0])):
    #             moments=[]
    #             Mp0q0 = (i ** p0) * (j ** q0) * img[i][j]
    #             Mp0q1 = (i ** p0) * (j ** q1) * img[i][j]
    #             Mp1q0 = (i ** p1) * (j ** q0) * img[i][j]
    #             Mp1q1 = (i ** p1) * (j ** q0) * img[i][j]
    #             moments.append([Mp0q0,Mp0q1,Mp1q0,Mp1q1])
    #
    #             Up0q0 = (i - (Mp1q0 / Mp0q0) ** p0) * (j - (Mp0q1 / Mp0q0) ** q0) * img[i][j]    #0
    #             Up2q0 = (i - (Mp1q0 / Mp0q0) ** p2) * (j - (Mp0q1 / Mp0q0) ** q0) * img[i][j]    #1
    #             Up0q2 = (i - (Mp1q0 / Mp0q0) ** p0) * (j - (Mp0q1 / Mp0q0) ** q2) * img[i][j]    #2
    #             Up1q1 = (i - (Mp1q0 / Mp0q0) ** p1) * (j - (Mp0q1 / Mp0q0) ** q1) * img[i][j]    #3
    #             Up3q0 = (i - (Mp1q0 / Mp0q0) ** p3) * (j - (Mp0q1 / Mp0q0) ** q0) * img[i][j]    #4
    #             Up0q3 = (i - (Mp1q0 / Mp0q0) ** p0) * (j - (Mp0q1 / Mp0q0) ** q3) * img[i][j]    #5
    #             Up1q2 = (i - (Mp1q0 / Mp0q0) ** p1) * (j - (Mp0q1 / Mp0q0) ** q2) * img[i][j]    #6
    #             Up2q1 = (i - (Mp1q0 / Mp0q0) ** p2) * (j - (Mp0q1 / Mp0q0) ** q1) * img[i][j]    #7
    #
    #             n20 = normalizedMoments(p2, q0, Up0q0, Up2q0)
    #             n02 = normalizedMoments(p0, q2, Up0q0, Up0q2)
    #             n11 = normalizedMoments(p1, q1, Up0q0, Up1q1)
    #             n30 = normalizedMoments(p3, q0, Up0q0, Up3q0)
    #             n03 = normalizedMoments(p0, q3, Up0q0, Up0q3)
    #             n12 = normalizedMoments(p1, q2, Up0q0, Up1q2)
    #             n21 = normalizedMoments(p2, q1, Up0q0, Up2q1)
    #             huMoments=[]
    #             H1 = n20 + n02
    #             H2 = (n20-n02) ** 2 + 4*(n11**2)
    #             H3 = (n30-(3*n12))**2 + (3*n21 - n03)**2
    #             H4 = (n30 + n12)**2 + (n21 + n03)**2
    #             #H5 = ()
    #             huMoments.append([H1,H2,H3,H4])
    #
    # return huMoments


def update_array(a,label1,label2):
    index = lab_small = lab_large = 0
    if label1 < label2:
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else:  # a[index] == lab_small
            break

    return


if __name__ == '__main__':
    main()
