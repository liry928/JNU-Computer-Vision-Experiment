# -*- coding: utf-8 -*-
import cv2
import os
import sys
import numpy as np

# 自动获取脚本所在目录
root_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(root_dir, "output")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print(f"输出文件夹路径：{out_dir}")

# 图片路径：优先使用命令行参数，其次使用脚本目录下的 image.png
# 例如：python ppt_correct.py D:\test\image.png
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = os.path.join(root_dir, "image.png")

# 如果你仍然想用原来的绝对路径，取消下面这行注释并修改路径
# img_path = r"C:\Users\lenovo\Desktop\computer vision\lab1\数字图像处理技术应用：PPT文档拍照\image.png"


def imread_cn(file_path):
    """兼容中文路径读取图片"""
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_cn(file_path, img):
    """兼容中文路径保存图片"""
    ext = os.path.splitext(file_path)[1]
    if not ext:
        ext = ".png"

    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(file_path)
    return ok


def order_points(pts):
    """按 左上、右上、右下、左下 排序四个点"""
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


def four_point_transform(image, pts):
    """四点透视变换"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth <= 0 or maxHeight <= 0:
        raise Exception("透视变换计算得到宽或高为0，角点坐标错误！")

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def enhance_image(img):
    """CLAHE + 双边滤波 + 锐化增强"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)

    lab_enh = cv2.merge((l_enh, a, b))
    img_clahe = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

    blur = cv2.bilateralFilter(img_clahe, 9, 75, 75)

    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    sharp = cv2.filter2D(blur, -1, sharpen_kernel)
    return sharp


def resize_limit(img, max_w=1200):
    """限制最大宽度，等比例缩放"""
    hh, ww = img.shape[:2]
    if ww > max_w:
        s = max_w / ww
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def resize_to_height(img, target_h):
    """按目标高度等比例缩放，避免拉伸变形"""
    h, w = img.shape[:2]
    if h == target_h:
        return img

    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    img = imread_cn(img_path)
    if img is None:
        raise Exception(f"图片读取失败，请检查路径：{img_path}")

    # 等比例缩小图片，最大宽度 1400
    max_w = 1400
    h, w = img.shape[:2]
    print(f"原始图片宽={w}, 高={h}")

    scale = 1.0
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"图片已缩放，缩放系数 scale={scale:.3f}")

    orig = img.copy()

    # 原始大图下的四个角点
    # 顺序：左上、右上、右下、左下
    # 注意：换图片后必须重新修改这四个点
    original_pts = np.array([
        [54, 51],       # 左上
        [2594, 64],     # 右上
        [2446, 1411],   # 右下
        [119, 1369]     # 左下
    ], dtype=np.float32)

    pts = (original_pts * scale).astype(np.int32)
    print(f"缩放后使用的4个角点：\n{pts}")

    warped_img = four_point_transform(orig, pts)
    result_img = enhance_image(warped_img)

    # 保存，兼容中文路径
    path1 = os.path.join(out_dir, "ppt_corrected.png")
    ok1 = imwrite_cn(path1, warped_img)
    print(f"保存 ppt_corrected.png ：{ok1}")

    path2 = os.path.join(out_dir, "ppt_final_enh.png")
    ok2 = imwrite_cn(path2, result_img)
    print(f"保存 ppt_final_enh.png ：{ok2}")

    # 拼接对比图：按原图高度缩放，避免拉伸变形
    h_target = orig.shape[0]
    warped_resize = resize_to_height(warped_img, h_target)
    final_resize = resize_to_height(result_img, h_target)

    compare_img = np.hstack([orig, warped_resize, final_resize])
    compare_img = resize_limit(compare_img, max_w=2400)

    path3 = os.path.join(out_dir, "ppt_compare.png")
    ok3 = imwrite_cn(path3, compare_img)
    print(f"保存 ppt_compare.png ：{ok3}")

    print("\n全部保存完成！去上面打印的输出文件夹查看图片")