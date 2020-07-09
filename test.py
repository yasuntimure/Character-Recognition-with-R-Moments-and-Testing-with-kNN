from PIL import *
from PIL import Image
import numpy as np
import math
from PIL import Image
from scipy import stats
import numpy as np
from numpy import genfromtxt
import statistics

def main():
    img = Image.open('image.png') #reads image
    img_gray = img.convert('L')  # converts the image to grayscale image
    # img_bin = img.convert('1') #converts to a binary image, T=128, LOW=0, HIGH=255
    img_gray.show()
    ONE = 150
    a = np.asarray(img_gray)  # from PIL to np array
    a_bin = threshold(a,100,ONE,0)
    im = Image.fromarray(a_bin)  # from np array to PIL format
    im.show()
    # a_bin = binary_image(100,100,ONE)  # creates a binary image
    label,img = blob_coloring_8_connected(a_bin,ONE) #obtains coloured and k valued array
    print(type(img))
    new_img = np2PIL_color(label)
    new_img.show()
    kValues = locateRectangle(img) #kValues = [k,mini,minj,maxi,maxj] values.
    drawRectengle(kValues, new_img) #draws rectengles and shows
    dataset=np.zeros((len(kValues),36)) # creates an array full of zeros
    new_img2 = np2PIL_color(label)
    for i in range(len(kValues)): # loop continues k variable times to calculate all figures in the image
        sqrImg = cropImage(kValues[i][2],kValues[i][1],kValues[i][4],kValues[i][3],new_img2) #cropes image to square sized
        H_Moments = huMoments(sqrImg)
        R_Moments = rMoments(H_Moments)
        Z_Moments = zernikeMomennts(sqrImg)
        dataset[i][:]= Z_Moments
        print("R Moments",R_Moments,"\n\nH_Moments",H_Moments)
        print("\n\nZ Moments",Z_Moments)
    dataset = np.random.rand(len(kValues),36)
    np.random.shuffle(dataset)
    percent80 = int(len(kValues)*80/100) # percent for the divide 80% of data
    training,test = dataset[:percent80,:],dataset[percent80:,:]
    print("Train Set",training)
    print("\n\nTest Set ",test)




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


def blob_coloring_8_connected(bim,ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    # print("nrow, ncol",nrow, ncol)
    im = np.zeros(shape=(nrow,ncol),dtype=int)
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
            label_lu = im[i - 1][j - 1]  # for the 8 connected system we need to add left up and right up.
            label_ru = im[i - 1][j + 1]
            im[i][j] = max_label
            if c == ONE:
                min_label = min(label_u,label_l,label_lu,label_ru)
                if min_label == max_label:
                    k += 1
                    im[i][j] = k
                else:
                    im[i][j] = min_label
                    if min_label != label_u and label_u != max_label:
                        update_array(a,min_label,label_u)
                    if min_label != label_l and label_l != max_label:
                        update_array(a,min_label,label_l)
                    if min_label != label_lu and label_lu != max_label:
                        update_array(a,min_label,label_lu)
                    if min_label != label_ru and label_ru != max_label:
                        update_array(a,min_label,label_ru)
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


def locateRectangle(im): #defines the minimum and maximum side of characters and saves it into k_values
    k_values = []
    nrow = len(im)
    ncol = len(im[0])
    labels = [10000]
    for i in range(nrow):
        for j in range(ncol):
            if im[i][j] not in labels:
                mini = nrow
                minj = ncol
                maxi = 0
                maxj = 0
                k = im[i][j]
                for i in range(nrow):
                    for j in range(ncol):
                        if k == im[i][j]:
                            if mini > i:
                                mini = i
                            if minj > j:
                                minj = j
                            if maxi < i:
                                maxi = i
                            if maxj < j:
                                maxj = j
                k_values.append([k,mini,minj,maxi,maxj])
                labels.append(k)
    return k_values


def drawRectengle(k_values,new_img):
    global drawRectengle
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
        new_img.show()


def normalizedMoments(p,q,u00,Upq): #
    y = (p + q) / 2 + 1
    n = Upq / (u00 ** y)
    return n


def make_square(im,min_size,fill_color=(0,0,0,0)): # fills with black colour the empty side og rectengles
    x,y = im.size
    size = max(min_size,x,y)
    new_im = Image.new('RGB',(size,size),fill_color)
    new_im.paste(im,(int((size - x) / 2),int((size - y) / 2)))
    return new_im


def cropImage(minj,mini,maxj,maxi,new_img):
    croppedImages = []
    try:
        area = (minj,mini,maxj,maxi)
        img = new_img.crop(area)
        croppedImages.append([img])
        #img.show()
        # img.save("cropped_picture.jpg")
    except IOError:
        pass
    x,y = img.size  # defines x or y greater than that and defines it min_size. So make_square func. fills the true are with black
    if x < y:
        min_size = x
    else:
        min_size = y
    img = make_square(img,min_size)
    img.show()
    img = img.resize((21,21))
    img_gray = img.convert('L')  # converts the image to grayscale image
    ONE = 150
    a = np.asarray(img_gray)  # from PIL to np array
    a_bin = threshold(a,100,ONE,0)
    return a_bin


def huMoments(sqrImg): # calculates Hu moments of cropped square img
    moments = []
    m00 = 0
    m01 = 0
    m10 = 0
    m11 = 0
    for i in range(len(sqrImg)):
        for j in range(len(sqrImg[0])):
            m00 = m00 + pow(i,0) * pow(j,0) * sqrImg[i][j]
    for i in range(len(sqrImg)):
        for j in range(len(sqrImg[0])):
            m01 = m01 + pow(i,0) * pow(j,1) * sqrImg[i][j]
    for i in range(len(sqrImg)):
        for j in range(len(sqrImg[0])):
            m10 = m10 + pow(i,1) * pow(j,0) * sqrImg[i][j]
    for i in range(len(sqrImg)):
        for j in range(len(sqrImg[0])):
            m11 = m11 + pow(i,1) * pow(j,1) * sqrImg[i][j]
    uMoments = []
    u = 0
    for p in range(4):
        for q in range(4):
            for i in range(len(sqrImg)):
                for j in range(len(sqrImg[0])):
                    u = u + pow((i - (m10 / m00)),p) * pow((j - (m01 / m00)),q) * sqrImg[i][j]
            uMoments.append([p,q,u])
    n20 = normalizedMoments(2,0,uMoments[0][2],uMoments[8][2])
    n02 = normalizedMoments(0,2,uMoments[0][2],uMoments[2][2])
    n11 = normalizedMoments(1,1,uMoments[0][2],uMoments[5][2])
    n30 = normalizedMoments(3,0,uMoments[0][2],uMoments[12][2])
    n03 = normalizedMoments(0,3,uMoments[0][2],uMoments[3][2])
    n12 = normalizedMoments(1,2,uMoments[0][2],uMoments[6][2])
    n21 = normalizedMoments(2,1,uMoments[0][2],uMoments[9][2])
    huMoments = []
    H1 = n20 + n02
    H2 = (n20 - n02) ** 2 + 4 * (n11 ** 2)
    H3 = (n30 - (3 * n12)) ** 2 + (3 * n21 - n03) ** 2
    H4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    H5 = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * ((n21 + n03) ** 2)) + (3 * n21 - n03) * (
                n21 + n03) * (3 * ((n30 + n12) ** 2) - ((n21 + n03) ** 2))
    H6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03)
    H7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * ((n21 + n03) ** 2)) + (3 * n12 - n30) * (
                n21 + n03) * (3 * ((n30 + n12) ** 2) - (n21 + n03) ** 2)
    huMoments.append([H1,H2,H3,H4,H5,H6,H7])
    return huMoments


def rMoments(array): # calculates R moments of cropped img by hu moments
    R_Moments=[]
    R1= math.sqrt(array[0][1]) / array[0][0]
    R2 = (array[0][0] + math.sqrt(array[0][1])) / (array[0][0] - (math.sqrt(array[0][1])))
    R3 = (math.sqrt(array[0][2])) / (math.sqrt(array[0][3]))
    R4 = (math.sqrt(array[0][2])) / math.sqrt(abs(array[0][4]))
    R5 = (math.sqrt(array[0][3])) / math.sqrt(abs(array[0][4]))
    R6 = abs(array[0][5]) / array[0][0] * array[0][2]
    R7 = abs(array[0][5]) / array[0][0] * math.sqrt(abs(array[0][4]))
    R8 = abs(array[0][5]) / array[0][2] * math.sqrt(array[0][1])
    R9 = abs(array[0][5]) / math.sqrt(array[0][1]*abs(array[0][4]))
    R10= abs(array[0][4]) / array[0][2]*array[0][3]
    R_Moments.append([R1,R2,R3,R4,R5,R6,R7,R8,R9,R10])
    return  R_Moments

def radialZernike(p,n1,m1): # calculates Radial Zernike Polynomials formula
    global r1
    r1=0
    a = int((n1 - abs(m1) / 2))
    for s in range(a):
        r1 = (pow(-1,s) * pow(p, (n1-2*s)) * math.factorial(abs(int(n1-2*s)))) / math.factorial(int(s)) * math.factorial(abs(int((n1+abs(m1))/2 - 2))) * math.factorial(abs(int((n1-abs(m1))/2 - 2 )))
    return r1

def zernikeMomennts(img): #calculates zernike moment of cropped square img
    N = len(img)
    Z_nm = []
    ZRnm=0
    ZInm=0
    Rnm=0
    Znm=0
    for n in range(6):
        for m in range(6):
            for i in range(N-1):
                for j in range(N-1):
                    xi = (math.sqrt(2)/N-1)*i - 1/math.sqrt(2)
                    yj = (math.sqrt(2)/N-1)*j - 1/math.sqrt(2)
                    dxi = 2 / N * math.sqrt(2)
                    dyj = dxi
                    Qij = math.atan(yj/xi)
                    pij = math.sqrt(pow(xi,2)+pow(yj,2))
                    Rnm = radialZernike(pij,n,m)
                    ZRnm = ZRnm + (img[i][j] * Rnm * math.cos(m*Qij) * dxi * dyj)
                    ZInm = ZInm + (img[i][j] * Rnm * math.sin(m*Qij) * dxi * dyj)
            ZRnm = ZRnm * (n+1)/math.pi
            ZInm = ZInm * -((n+1)/math.pi)
            Znm = math.sqrt(pow(ZRnm,2) + pow(ZInm,2))
            Z_nm.append(Znm)

    return Z_nm


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
