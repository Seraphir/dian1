import cv2
import numpy as np
import os

Input_Dir = "./ARJ21/"
Store_Dir = "./aug/"
filelist = os.listdir(Input_Dir)
if not os.path.exists(Store_Dir):
    os.makedirs(Store_Dir)


class augment:
    def __init__(self, filled_color=(0, 0, 0)):
        self.param = {"deg": 0, "mv": 0, "mh": 0, "ptv": 0, "pth": 0}
        self.filled_color = filled_color

    # generate rand
    def genrand(self, mean, sigmaX3, edgel=-999, edgeh=999):
        np.random.seed()
        value = mean + np.random.randn() * (sigmaX3 / 3)
        if value < edgel:
            value = edgel
        elif value > edgeh:
            value = edgeh
        return value

    def rotate(self, img, angle):
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=self.filled_color)

    def move(self, img, dv, dh):
        (h, w) = img.shape[:2]
        M = np.float32([[1, 0, dh], [0, 1, dv]])
        return cv2.warpAffine(img, M, (w, h), borderValue=self.filled_color)

    def Perspective(self, img, dv, dh):
        (h, w) = img.shape[:2]

        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[0 + dh, 0 + dv], [w - dh, 0 - dv], [0 - dh, h - dv], [w + dh, h + dv]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        return cv2.warpPerspective(img, M, (w, h), borderValue=self.filled_color)

    def genparam(self):
        # rotate
        self.param["deg"] = 360 * np.random.rand()
        # movement
        self.param["mv"] = self.genrand(0, 50)
        self.param["mh"] = self.genrand(0, 50)
        # perspective
        self.param["ptv"] = self.genrand(0, 10)
        self.param["pth"] = self.genrand(0, 10)
        return

    def param2str(self):
        string = ""
        for key in self.param:
            string += "_{:.0f}".format(self.param[key])
        print(string)
        return string

    # main transform
    def modify(self, img):
        img = self.rotate(img, angle=self.param["deg"])  # 1deg max
        img = self.move(img, dv=self.param["mv"], dh=self.param["mh"])
        img = self.Perspective(img, dv=self.param["ptv"], dh=self.param["pth"])
        return img


# main function
for filename in filelist:
    if filename[-3:] not in ["png", "jpg", "bmp", "tif"]:
        continue
    # Read Input Imagine
    img = cv2.imread(Input_Dir + filename)
    filled_color = np.where(img>0)
    Aug = augment()
    for i in range(1):
        # Generate parameter
        Aug.genparam()
        paramstr = Aug.param2str()
        # Modify Imagine
        img = Aug.modify(img)
        cv2.imwrite(Store_Dir + filename[:-4] + paramstr + ".jpg", img)
        # cv2.imshow(paramstr, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()