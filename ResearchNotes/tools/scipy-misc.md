# Scipy Misc

``scipy.misc`` 已经在版本 1.10.0 后取消，这里记一下对其中函数的替代方法：

---

```python
scipy.misc.imread(filename) # return numpy.ndarray type np.uint8
scipy.misc.imsave(filename:str, image:numpy.ndarray) # image type: np.uint8
```

替代为：

```python
import cv2
cv2.imread(filename:str) # return numpy.ndarray
cv2.imwrite(filename:str, image:numpy.ndarray) # image type: np.uint8
```

---

```python
scipy.misc.imresize(img, (self.img_height, self.img_width))
```

替代为：

```python
from PIL import Image
curr_img = np.array(
    Image.fromarray(curr_img).resize((self.img_width, self.img_height))
)
```

**注意 ``resize`` 方法参数的顺序**。
