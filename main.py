import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from pyzbar import pyzbar


parser = argparse.ArgumentParser(description='DIP Final Project')
# args
parser.add_argument('--img-dir', default='train', help='diretory of images')
parser.add_argument('--tickets-dir', default='tickets', help='diretory of tickets')
parser.add_argument('--qr-dir', default='qr', help='diretory of QR code')
parser.add_argument('--output', default='predictions.txt', help='result file')

args = parser.parse_args()

shape_x, shape_y = int(856 / 1.5), int(540 / 1.5)
outer_x, outer_y = 20, 20

"""
You may define your helper functions here
"""
def solve(p1, p2, p3, p4):
    A = np.array([[p2[1]-p1[1], p1[0]-p2[0]], [p4[1]-p3[1], p3[0]-p4[0]]])
    B1 = np.array([[(p2[1]-p1[1])*p1[0]+(p1[0]-p2[0])*p1[1], p1[0]-p2[0]], 
                  [(p4[1]-p3[1])*p3[0]+(p3[0]-p4[0])*p3[1], p3[0]-p4[0]]])
    B2 = np.array([[p2[1]-p1[1], (p2[1]-p1[1])*p1[0]+(p1[0]-p2[0])*p1[1]],
                  [p4[1]-p3[1], (p4[1]-p3[1])*p3[0]+(p3[0]-p4[0])*p3[1]]])
    A, B1, B2 = map(lambda x: np.linalg.det(x), (A, B1, B2))
    return B1/A, B2/A

def handle_not_rect(points):
    new_points = np.zeros((points.shape[0] - 1, points.shape[1]))
    dist_list = [np.linalg.norm(points[i] - points[i - 1]) for i in range(points.shape[0])]
    i = np.argmin(np.array(dist_list)) # 缩点(i-2, i-1), (i, i+1)
    x, y = solve(points[i-2], points[i-1], points[i], points[(i+1)%points.shape[0]])
    # print(new_points[:i-1, :].shape, points[:i-1, :].shape, i)
    if i > 0:
        new_points[:i-1, :] = points[:i-1, :]
    new_points[i-1, :] = np.array([x, y])
    new_points[i:, :] = points[i+1:, :]
    return new_points

def rotate_if_reversed(input_img):
    img = cv2.medianBlur(input_img, 5)
    _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

    r, c = img.shape
    lu = img[outer_y>>1: int(shape_y*0.4), outer_x>>1: int(shape_x*0.25)]
    rd = img[r-int(shape_y*0.4):r-(outer_y>>1), c-int(shape_x*0.25): c-(outer_x>>1)]
    if(np.mean(lu) < np.mean(rd)):
        input_img = cv2.flip(input_img, -1)
    return input_img

def extract_ticket(att_img):
    padding = 100
    att_img = np.pad(att_img, ((padding, padding), (padding, padding)), 'constant', constant_values=(0, 0))
    blurred_img = cv2.medianBlur(att_img, 5)
    _, thres_img = cv2.threshold(blurred_img, 80, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    morphed_img = cv2.morphologyEx(thres_img, cv2.MORPH_CLOSE, kernel)

    canny = cv2.Canny(morphed_img, 40, 150)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour ,max_area = 0, 0
    for contour in contours:
        tmp = cv2.contourArea(contour)
        if tmp > max_area:
            max_contour, max_area = contour, tmp

    poly_contour = cv2.approxPolyDP(max_contour, 0.02 * cv2.arcLength(max_contour, True), True)
    points = poly_contour.squeeze()
    while points.shape[0] > 4:
        points = handle_not_rect(points)

    c = np.cross(points[1]-points[0], points[2]-points[1])
    if c > 0:   # 修正为右旋标价
        points = points[::-1, :]

    len1 = np.linalg.norm(points[0] - points[1])
    len2 = np.linalg.norm(points[1] - points[2])
    if len1 >= len2:    # 短边优先
        points = np.roll(points, 1, axis=0)

    if points[0][0] > points[2][0]: # y值小的优先
        points = np.roll(points, 2, axis=0)

    N = np.array([[outer_x, outer_y], [outer_x, shape_y - outer_y], 
                [shape_x - outer_x, shape_y - outer_y], [shape_x - outer_x, outer_y]])
    mat = cv2.getPerspectiveTransform(points.astype(np.float32), N.astype(np.float32))
    return cv2.warpPerspective(att_img, mat, (shape_x, shape_y))

def hoffle_fix(output1):
    blurred_img = cv2.medianBlur(output1, 5)
    _, thres_img = cv2.threshold(blurred_img, 80, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    morphed_img = cv2.morphologyEx(thres_img, cv2.MORPH_CLOSE, kernel)

    canny = cv2.Canny(morphed_img, 40, 150)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 100)
    lines = lines.squeeze()

    angle, cnt = 0, 0
    for _, theta in lines:
        theta = theta * 180 / np.pi - 90
        if np.abs(theta) < 10:
            angle = angle * cnt + theta
            cnt += 1
            angle /= cnt
    mat = cv2.getRotationMatrix2D((shape_x/2, shape_y/2), angle, 1)
    fixed_img = cv2.warpAffine(output1, mat, (shape_x, shape_y))

    # 裁剪一下边缘
    return fixed_img[outer_y>>1:shape_y-(outer_y>>1), outer_x>>1:shape_x-(outer_x>>1)]

def crop_all_tickets(img_dir, ticket_dir):
    '''
    Crop all tickets
    '''

    print('cropping all tickets...')
    if not os.path.exists(ticket_dir):
        os.mkdir(ticket_dir)

    for file_name in tqdm(os.listdir(img_dir)):
        try:
            img = cv2.imread(os.path.join(img_dir, file_name), cv2.IMREAD_GRAYSCALE)
            ticket_img = extract_ticket(img)
            ticket_img = hoffle_fix(ticket_img)
            ticket_img = rotate_if_reversed(ticket_img)
            cv2.imwrite(os.path.join(ticket_dir, file_name), ticket_img)
        except Exception as e:
            print("\nFail at {}: {}, skipped".format(file_name, e.args[0]))
    print('\ndone!')

# =====================Part 1 End========================
def fetch_line(lines, shape):
    lines = abs(lines) # 对rho取绝对值
    line_num = 4 # 横竖各考虑2条干扰线，太多会引入误差，太少容易漏掉真边
    standard_shape = 90 # 设定二维码标准长宽
    fetch_lines = np.zeros((2, line_num)) 
    cnt = [0, 0]
    for rho, theta in lines:
        mode = 0
        if theta < np.pi / 180 * 5 or np.pi - theta < np.pi / 180 * 5: # 竖线
            mode = 0
        elif abs(theta - np.pi / 2) < np.pi / 180 * 5: # 横线
            mode = 1
        else:
            continue
        if cnt[mode] >= line_num:
            continue

        for i in range(cnt[mode]):
            r = fetch_lines[mode][i]
            if abs(rho - r) <= 15: # 认为是同一条线，rho取更外围的值
                if rho > shape[0]/2 and rho > r:
                    fetch_lines[mode][i] = rho
                elif rho < shape[0]/2 and rho < r:
                    fetch_lines[mode][i] = rho
                break
        else:
            fetch_lines[mode][cnt[mode]] = rho
            cnt[mode] += 1

    # 取相邻两边距离最接近standard_shape的一对边
    fetch_lines = np.sort(fetch_lines, axis=1)
    # print("sorted:", fetch_lines)
    result = np.zeros((2, 2))
    for mode in range(2):
        min = standard_shape
        for i in range(line_num):
            if fetch_lines[mode, i] == 0: continue
            for j in range(i+1, line_num):
                dis = fetch_lines[mode, j] - fetch_lines[mode, i] - standard_shape 
                if dis > 0 and dis < min:
                    min = dis
                    result[mode] = [fetch_lines[mode, i], fetch_lines[mode, j]]
    
    # print(result)
    return np.uint8(result)

def extract_qr_old(att_img):
    # 裁剪，右下角约八分之一
    r, c = att_img.shape
    crop_img = att_img[int(r*0.5):r-30, int(c*0.7): c-5]

    # 滤波、二值化
    blurred_img = cv2.medianBlur(crop_img, 7)
    _, thres_img = cv2.threshold(blurred_img, max(blurred_img.reshape(-1)) * 0.5, 255, cv2.THRESH_BINARY)  

    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morphed_img = cv2.morphologyEx(thres_img, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    morphed_img = cv2.morphologyEx(morphed_img, cv2.MORPH_CLOSE, kernel)

    # 霍夫变换
    canny = cv2.Canny(morphed_img, 80, 150)
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 10) # 参数
    lines = lines.squeeze()

    # 求解四条边界
    fetch_lines = fetch_line(lines, canny.shape)
    left, right, up, down = (*fetch_lines[0], *fetch_lines[1])

    res_img = crop_img[up-2:down+2, left-2:right+2]
    return res_img

def extract_qr(att_img, img_name, fail_list):
    # 裁剪，右下角约八分之一
    r, c = att_img.shape
    crop_img = att_img[int(r*0.5):r-30, int(c*0.7): c-5]

    blurred_img = cv2.medianBlur(crop_img, 7)
    blurred_img = 255 - blurred_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morphed = cv2.morphologyEx(blurred_img, cv2.MORPH_OPEN, kernel)
    top_hat = 255 - cv2.subtract(blurred_img, morphed)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morphed_img = cv2.morphologyEx(top_hat, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morphed_img = cv2.morphologyEx(morphed_img, cv2.MORPH_CLOSE, kernel)

    _, morphed_img = cv2.threshold(morphed_img, np.max(morphed_img) * 0.8, 255, cv2.THRESH_BINARY)
    morphed_img = np.pad(morphed_img, ((0, 2), (0, 2)),'constant', constant_values=(255,255))

    canny = cv2.Canny(morphed_img, 80, 150)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour ,max_area = 0, 0
    for contour in contours:
        tmp = cv2.contourArea(contour)
        if tmp > max_area:
            max_contour, max_area = contour, tmp

    max_contour = max_contour.squeeze()
    maxx, maxy = np.max(max_contour, axis=0) + 8
    minx, miny = np.min(max_contour, axis=0) - 8

    res_img = crop_img[miny:maxy, minx:maxx]

    if not (res_img.shape[1]>0 and res_img.shape[0] / res_img.shape[1] > 0.9 and res_img.shape[0] / res_img.shape[1] < 1.1):
        fail_list.append(img_name)
        att2_img = extract_qr_old(att_img)
        if att2_img.shape[0]>0 and att2_img.shape[1]>0:
            return att2_img
    return res_img

def crop_all_qrcodes(ticket_dir, qr_dir):
    print('cropping all QR codes...')
    if not os.path.exists(qr_dir):
        os.mkdir(qr_dir)

    fail_list = []
    for file_name in tqdm(os.listdir(ticket_dir)):
        try:
            img = cv2.imread(os.path.join(ticket_dir, file_name), cv2.IMREAD_GRAYSCALE)
            qr_img = extract_qr(img, file_name, fail_list)
            cv2.imwrite(os.path.join(qr_dir, file_name), qr_img)
        except Exception as e:
            print("\nFail at {}: {} skipped".format(file_name, e.args[0]))
    print('\ndone!')
    print('Following pictures were failed with new algorithm. Old algorithm was applied:')
    for name in fail_list:
        print(name)

# =====================Part 2 End========================

def recog_qr(img, not_continue=False):
    barcodes=pyzbar.decode(img)
    if len(barcodes) > 0:
        return 0, barcodes[0].data.decode('utf-8')
    
    img = 255 - img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morphed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    output = 255 - cv2.subtract(img, morphed)
    barcodes=pyzbar.decode(output)
    if len(barcodes) > 0:
        return 1, barcodes[0].data.decode('utf-8')

    laplacian = cv2.Laplacian(output, -1)
    new_img =cv2.subtract(output, laplacian)
    barcodes=pyzbar.decode(new_img)
    if len(barcodes) > 0:
        return 2, barcodes[0].data.decode('utf-8')
    
    test = cv2.adaptiveThreshold(new_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    barcodes=pyzbar.decode(test)
    if len(barcodes) > 0:
        return 3, barcodes[0].data.decode('utf-8')
    test = cv2.adaptiveThreshold(new_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    barcodes=pyzbar.decode(test)
    if len(barcodes) > 0:
        return 3, barcodes[0].data.decode('utf-8')
    
    if not_continue:
        return -1, ''

    for i in range(127, 256):  #粗暴的阈值处理
        _, output2 = cv2.threshold(new_img, i, 255, cv2.THRESH_BINARY)
        barcodes=pyzbar.decode(output2)
        if len(barcodes) != 0:
            return 4, barcodes[0].data.decode('utf-8')
    return -1, ''

def process_qr(img):
    status, content = recog_qr(img, False)
    if status != -1:
        return status, content
    
    for rate in np.arange(0.8, 1, 0.01):
        for divide in np.arange(0.25, 0.5, 0.05):
            r, c = img.shape
            crop_x = int(c * divide)
            left_img = img[:, :crop_x]
            right_img = img[:, crop_x:]
            rl, cl = left_img.shape
            left_img2 = cv2.resize(left_img, (int(cl*rate),  rl))
            fit_img = np.hstack([left_img2, right_img])

            status, content = recog_qr(fit_img, True)
            if status != -1:
                return status, content
    return -1, ''

def recg_all_qrcodes(qr_dir, result_file):
    print('recognizing QR codes...')
    counter = [0, 0, 0, 0, 0]

    with open(result_file, 'w') as f:
        for file_name in tqdm(os.listdir(qr_dir)):
            img = cv2.imread(os.path.join(qr_dir, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            status, result = process_qr(img)
            f.write('{} {}\n'.format(file_name, result))
            if status != -1:
                counter[status] += 1

    total = sum(counter)
    print()
    print('Sum*', 'Base', 'T-hat', 'lapla', 'athrs', 'thrs', sep='\t')
    print(total, *counter, sep='\t')
    print('Results were saved to {}.'.format(result_file))


def main(img_dir, ticket_dir, qr_dir, result_file):
    crop_all_tickets(img_dir, ticket_dir)
    crop_all_qrcodes(ticket_dir, qr_dir)
    recg_all_qrcodes(qr_dir, result_file)
    print('done!')

if __name__ == "__main__":
    main(args.img_dir, args.tickets_dir, args.qr_dir, args.output)
