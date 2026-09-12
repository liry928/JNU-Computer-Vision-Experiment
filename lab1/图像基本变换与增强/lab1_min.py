import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# #解决matplotlib中文乱码
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
# plt.rcParams['axes.unicode_minus'] = False

root_dir = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(root_dir, "picture")
out_dir = os.path.join(root_dir, "output")  

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#原图路径
path_messi = os.path.join(pic_dir, "messi5.jpg")
path_sudoku = os.path.join(pic_dir, "sudoku.png")
path_lena = os.path.join(pic_dir, "lena.jpg")

def imread_cn(file_path):
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

#工具函数：保存并绘图对比
def save_show(origin, processed, save_name, title1="原图", title2="处理后"):
    out_file = os.path.join(out_dir, save_name)
    cv2.imwrite(out_file, processed)
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(origin, cv2.COLOR_BGR2RGB))
    plt.title(title1)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.title(title2)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#任务1 messi5.jpg：缩放、平移、旋转、仿射
img_messi = imread_cn(path_messi)
if img_messi is None:
    raise FileNotFoundError(f"读取失败：{path_messi}\n")
h, w = img_messi.shape[:2]

#1‑1 图像缩放
img_resize = cv2.resize(img_messi, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
save_show(img_messi, img_resize, "geo_resize.png")

#1‑2 图像平移
M_trans = np.float32([[1,0,80],[0,1,40]])
img_trans = cv2.warpAffine(img_messi, M_trans, (w, h))
save_show(img_messi, img_trans, "geo_trans.png")

#1‑3 图像旋转
M_rot = cv2.getRotationMatrix2D((w//2, h//2), 30, 1.0)
img_rot = cv2.warpAffine(img_messi, M_rot, (w, h))
save_show(img_messi, img_rot, "geo_rotate.png")

#1‑4 仿射变换：三点映射
src_pts = np.float32([[0,0],[w-1,0],[0,h-1]])
dst_pts = np.float32([[20,30],[w-40,10],[10,h-20]])
M_affine = cv2.getAffineTransform(src_pts, dst_pts)
img_affine = cv2.warpAffine(img_messi, M_affine, (w, h))
save_show(img_messi, img_affine, "geo_affine.png")

#任务2 sudoku.png：透视矫正
img_sudoku = imread_cn(path_sudoku)
if img_sudoku is None:
    raise FileNotFoundError(f"读取失败：{path_sudoku}")
h_s, w_s = img_sudoku.shape[:2]
#sudoku棋盘四点坐标
src_persp = np.float32([[73,84],[488,70],[15,510],[518,520]])
dst_persp = np.float32([[0,0],[w_s,0],[0,h_s],[w_s,h_s]])
M_persp = cv2.getPerspectiveTransform(src_persp, dst_persp)
img_persp = cv2.warpPerspective(img_sudoku, M_persp, (w_s, h_s))
save_show(img_sudoku, img_persp, "geo_perspective.png")

#任务3 lena.jpg：直方图均衡、CLAHE、滤波、锐化
img_lena = imread_cn(path_lena)
if img_lena is None:
    raise FileNotFoundError(f"读取失败：{path_lena}")
gray_lena = cv2.cvtColor(img_lena, cv2.COLOR_BGR2GRAY)

#3‑1 全局直方图均衡
equ_gray = cv2.equalizeHist(gray_lena)
cv2.imwrite(os.path.join(out_dir,"enh_hist_eq.png"), equ_gray)
plt.figure(figsize=(10,4))
plt.subplot(121);plt.imshow(gray_lena,cmap="gray");plt.title("原图灰度");plt.axis("off")
plt.subplot(122);plt.imshow(equ_gray,cmap="gray");plt.title("全局直方图均衡");plt.axis("off")
plt.tight_layout()
plt.show()

#3‑2 CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_gray = clahe.apply(gray_lena)
cv2.imwrite(os.path.join(out_dir,"enh_clahe.png"), clahe_gray)
plt.figure(figsize=(10,4))
plt.subplot(121);plt.imshow(gray_lena,cmap="gray");plt.title("原图灰度");plt.axis("off")
plt.subplot(122);plt.imshow(clahe_gray,cmap="gray");plt.title("CLAHE");plt.axis("off")
plt.tight_layout()
plt.show()

#3‑3 各类平滑滤波
img_mean = cv2.blur(img_lena, (5,5))
save_show(img_lena, img_mean, "enh_mean.png")

img_gauss = cv2.GaussianBlur(img_lena, (5,5), sigmaX=1.5)
save_show(img_lena, img_gauss, "enh_gauss.png")

img_median = cv2.medianBlur(img_lena, 5)
save_show(img_lena, img_median, "enh_median.png")

img_bilateral = cv2.bilateralFilter(img_lena, d=9, sigmaColor=75, sigmaSpace=75)
save_show(img_lena, img_bilateral, "enh_bilateral.png")

#3‑4 锐化
kernel_sharp = np.array([
    [-1,-1,-1],
    [-1, 9,-1],
    [-1,-1,-1]
])
img_sharp = cv2.filter2D(img_lena, -1, kernel_sharp)
save_show(img_lena, img_sharp, "enh_sharp.png")
