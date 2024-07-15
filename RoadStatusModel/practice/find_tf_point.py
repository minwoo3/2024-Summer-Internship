import cv2
import numpy as np

win_name = "scanning"
img = cv2.imread("/home/rideflux/Public/GeneralCase/Raw/31110/camera_0/100.jpg")
img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10)
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)
fixed_y = None

def sort_points(points):
    sorted_pts = sorted(points, key=lambda point: point[1])  # y좌표 기준으로 정렬
    top_pts = sorted(sorted_pts[:2], key=lambda point: point[0])  # x좌표 기준으로 정렬
    bottom_pts = sorted(sorted_pts[2:], key=lambda point: point[0])  # x좌표 기준으로 정렬
    ################ top left, top right, bottom left, bottom right
    return [top_pts[0], top_pts[1], bottom_pts[0], bottom_pts[1]]

def ptf(pts):
    global draw
    tl, tr, bl, br = pts
    pts1 = np.float32([tl, tr, bl, br])

    w1 = abs(br[0]-bl[0])
    w2 = abs(tr[0]-tl[0])
    width = int(max([w1,w2]))
    h1 = abs(br[1]-tr[1])
    h2 = abs(bl[1]-tl[1])
    height = int(max([h1,h2]))

    pts2 = np.float32([[0,0],[width-1,0],[0, height-1],[width-1,height-1]])

    transform_mat = cv2.getPerspectiveTransform(pts1,pts2)

    result = cv2.warpPerspective(draw, transform_mat, (width, height))
    return result

def onMouse(event, x, y, flags, param):
    global pts_cnt, fixed_y
    if event == cv2.EVENT_LBUTTONDOWN and fixed_y is None:
        fixed_y = y
        cv2.circle(draw, (x, fixed_y), 5, (0, 255, 0), -1)
        cv2.line(draw, (0, fixed_y), (cols, fixed_y), 5)
        cv2.imshow(win_name, draw)
        
        pts[pts_cnt] = [int(x), fixed_y]
        pts_cnt += 1

    elif event == cv2.EVENT_LBUTTONDOWN and fixed_y is not None and pts_cnt % 2 == 1:
        cv2.circle(draw, (x, fixed_y), 5, (0, 255, 0), -1)
        cv2.imshow(win_name, draw)
        pts[pts_cnt] = [int(x), fixed_y]
        fixed_y = None
        pts_cnt += 1

    elif pts_cnt > 0 and pts_cnt % 4 == 0:
        sorted_pts = sort_points(pts)
        print(sorted_pts)
        transformed = ptf(sorted_pts)
        canny = cv2.Canny(transformed,50, 150)
        canny_3d = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

        combined = np.hstack((transformed,canny_3d))
        cv2.imshow('combined', combined)
        cv2.waitKey(0)
        cv2.destroyWindow('combined')
        pts_cnt = 0
    
cv2.imshow(win_name, img)
cv2.setMouseCallback(win_name, onMouse)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키의 ASCII 코드
        break
cv2.destroyAllWindows()
