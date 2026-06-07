import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def show_step(title, image, cmap=None):
    plt.figure(figsize=(10, 6))

    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap=cmap)

    plt.title(title)
    plt.axis("off")
    plt.show()



IMAGE_PATH = "gambar/parking_ori.jpg"

os.makedirs("output", exist_ok=True)
os.makedirs("output/steps", exist_ok=True)


img = cv2.imread(IMAGE_PATH)

if img is None:
    raise Exception("Gambar tidak ditemukan")

scale = 0.5

img = cv2.resize(
    img,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_AREA
)

result = img.copy()

print(f"Ukuran baru: {img.shape[1]}x{img.shape[0]}")



show_step("Step 1 - Original Image", img)

cv2.imwrite(
    "output/steps/01_original.png",
    img
)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

show_step(
    "Step 2 - HSV Image",
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
)

cv2.imwrite(
    "output/steps/02_hsv.png",
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
)


lower_asphalt = np.array([0, 0, 40])
upper_asphalt = np.array([180, 70, 200])

asphalt_mask = cv2.inRange(
    hsv,
    lower_asphalt,
    upper_asphalt
)

show_step(
    "Step 3 - Asphalt Segmentation",
    asphalt_mask,
    cmap="gray"
)

cv2.imwrite(
    "output/steps/03_asphalt_mask.png",
    asphalt_mask
)



cars_mask = cv2.bitwise_not(asphalt_mask)

show_step(
    "Step 4 - Cars Mask",
    cars_mask,
    cmap="gray"
)

cv2.imwrite(
    "output/steps/04_cars_mask.png",
    cars_mask
)



kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (5,5)
)

cars_mask = cv2.morphologyEx(
    cars_mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

cars_mask = cv2.morphologyEx(
    cars_mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)

show_step(
    "Step 5 - Morphology Result",
    cars_mask,
    cmap="gray"
)

cv2.imwrite(
    "output/steps/05_morphology.png",
    cars_mask
)



num_labels, labels, stats, centroids = \
    cv2.connectedComponentsWithStats(
        cars_mask,
        connectivity=8
    )

car_count = 0

for i in range(1, num_labels):

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    if area < 350:
        continue

    if area > 3000:
        continue

    if w < 25:
        continue

    if h < 25:
        continue

    if w > 120:
        continue

    if h > 120:
        continue

    ratio = max(w,h) / min(w,h)

    if ratio > 3.5:
        continue

    fill_ratio = area / (w*h)

    if fill_ratio < 0.40:
        continue

    car_count += 1

    cv2.rectangle(
        result,
        (x,y),
        (x+w,y+h),
        (0,255,0),
        2
    )

    cv2.putText(
        result,
        str(car_count),
        (x,y-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        1
    )


cv2.putText(
    result,
    f"Cars: {car_count}",
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,0),
    2
)

cv2.imwrite(
    "output/result.png",
    result
)

cv2.imwrite(
    "output/steps/06_final_detection.png",
    result
)

print("="*50)
print(f"TOTAL MOBIL TERDETEKSI = {car_count}")
print("="*50)

show_step(
    f"Step 6 - Final Detection ({car_count} Cars)",
    result
)



fig, ax = plt.subplots(2, 3, figsize=(16,10))

ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0,0].set_title("Original Image")

ax[0,1].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
ax[0,1].set_title("HSV")

ax[0,2].imshow(asphalt_mask, cmap="gray")
ax[0,2].set_title("Asphalt Mask")

ax[1,0].imshow(cv2.bitwise_not(asphalt_mask), cmap="gray")
ax[1,0].set_title("Cars Mask")

ax[1,1].imshow(cars_mask, cmap="gray")
ax[1,1].set_title("Morphology")

ax[1,2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
ax[1,2].set_title(f"Detected Cars = {car_count}")

for a in ax.ravel():
    a.axis("off")

plt.tight_layout()

plt.savefig(
    "output/steps/all_steps.png",
    bbox_inches="tight"
)

plt.show()
