import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import PIL

plt.style.use('dark_background')

###Read Input Image###
##이미지를 불러온 후 너비, 높이, 채널의 값 저장##

# 자동차 사진의 객체 행렬 읽어들이기
img = cv2.imread('anaconda3/images/car.jpg')

# 객체행렬 값들 부여(높이,넓이,채널)
height, width, channel = img.shape

cv2.imshow("S", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

###Convert Image to Grayscale###

# convertColor함수를 이용해 gray채널로 변경
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("T", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 가로12,세로10인치의 figure 생성
plt.figure(figsize=(12, 10))

# gray변수에 담긴 이미지를 'gray'색 colormap으로 출력
plt.imshow(gray, cmap='gray')

##자동차 이미지의 edge추출하기

# 모폴로지 변환에서 사용 하는 함수라고 하는데 정확힌 모르겠음
# 우선 구조요소를 생성하는 함수!
structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

# 두 이미지 사이의 연산 함수(add,subtract..)
imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

###가우시안 블러 작업###
##가우시안 블러는 사진의 노이즈를 없애는 작업.
##번호판을 더 잘 찾을 수 있게 해줌!
img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

###Thresholding###
##threshold값을 기준으로 정하고, 이보다 낮은 값은0, 높은 값은 255로 설정
##즉 흑과 백으로마 사진을 구성하는 방법
img_blur_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)

###Thresholding###
##threshold값을 기준으로 정하고, 이보다 낮은 값은0, 높은 값은 255로 설정
##즉 흑과 백으로마 사진을 구성하는 방법
img_thresh = cv2.adaptiveThreshold(
    gray,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)

# 블러처리하지 않은 이미지.
plt.figure(figsize=(12, 10))
plt.imshow(img_thresh, cmap='gray')

# 블러처리한 이미지.
plt.figure(figsize=(12, 10))
plt.imshow(img_blur_thresh, cmap='gray')

###contours(윤곽)찾기###
# contour란 같은 값을 가진 곳을 연결한 선이라고 생각하면 됨.
# 예를 들어 등고선, 등압선 ..
# 이미지 contour란 동일한 색 또는 동일한 색상 강도를 가진 부분의 가장자리 경계를 연결한 선
# OpenCV는 contour를 찾을때 검은 바탕에 찾는 물체는 흰색으로 설정해야한다.
# 전체 이미지에서 contour의 가로 세로 비율 값과 면적을 통해, 번호판 영역에 벗어난 걸로 추정되는 값들은 제외 시켜주었다.

# findContours() : 검은색 바탕에서 흰색 대상 찾아주는 메서드
# 그래서 위에서 Thresholding + 가우시안 블러 적용해준 것
contours, _ = cv2.findContours(img_blur_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)


temp_result = np.zeros((height, width, channel), dtype=np.uint8)

cv2.drawContours(temp_result, contours=contours, contourIdx=-1,
                 color=(255, 255, 255))

plt.figure(figsize=(12, 10))

plt.imshow(temp_result)

###Data 준비하기###
##원본 사진과 동일한 크기에다가 찾은 Contours들의 좌표를 이용해
##사각형 형태를 그려보기.
##동시에 딕셔너리를 하나 만들어 contours들의 정보를 저장.

contours_dict = []

x, y, w, h = cv2.boundingRect(contours)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h),
                  color=(255, 255, 255), thickness=2)

    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')

###Select Candidates by char size###
##번호판 글자일 것 같은 Contours추려내기##
##적당한 상수값을 지정한 후 비슷한 크기의 윤곽들만 따로 저장##

MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

# 번호판의 가능성 높은 윤곽들 담는 배열
possible_contours = []

