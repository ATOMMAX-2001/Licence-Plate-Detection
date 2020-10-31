import numpy as np
import cv2
image = cv2.imread(r'D:\abilash\Desktop\hackathon\download.jpg',cv2.IMREAD_COLOR)
boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]
for (lower, upper) in boundaries:
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	print(output)
	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)
