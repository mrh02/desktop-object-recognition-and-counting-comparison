import cv2

# 直接读取桌面上的照片
image = cv2.imread(r"C:\Users\moruohan\Desktop\desk.jpg")

if image is None:
    print("没有读到照片，请检查路径和文件名。")
else:
    height, width = image.shape[:2]
    print(f"图片宽度：{width}，高度：{height}")

    # 等比例缩放，方便查看
    scale = min(800 / width, 600 / height, 1)
    preview = cv2.resize(
        image,
        (int(width * scale), int(height * scale))
    )

    cv2.imshow("Desk image", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()