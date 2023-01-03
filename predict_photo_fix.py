from pathlib import Path

from deoldify.generators import gen_inference_wide
from deoldify.filters import MasterFilter, ColorizerFilter
import cv2
import numpy as np
from PIL import Image


def photo_fix(image_path, mask_path):
    # 指定模型文件
    learn = gen_inference_wide(root_folder=Path('./'), weights_name='ColorizeStable_gen')
    # 加载模型
    deoldfly_model = MasterFilter([ColorizerFilter(learn=learn)], render_factor=10)
    # 读取原图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    filtered_image = deoldfly_model.filter(
        pil_img, pil_img, render_factor=35, post_process=True
    )

    result_img = np.asarray(filtered_image)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    # 保存修复后的图像
    cv2.imwrite(mask_path, result_img)
