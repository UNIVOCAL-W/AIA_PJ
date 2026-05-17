import cv2
import matplotlib.pyplot as plt
from pathlib import Path

img_path = Path(r"C:\AIA_workspace\data\raw\Leaves\1001.jpg")

img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")
plt.show()