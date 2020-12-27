import os
import argparse
import cv2


parser = argparse.ArgumentParser(description='DIP Final Project')
# args
parser.add_argument('--img-dir', default='train', help='diretory of images')
parser.add_argument('--tickets-dir', default='tickets', help='diretory of tickets')
parser.add_argument('--qr-dir', default='train', help='diretory of QR code')

args = parser.parse_args()


"""
You may define your helper functions here
"""


def main(img_dir, ticket_dir, qr_dir):
    # ...
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        """
        Your main loop here.
        A concrete example:
        ticket = crop_ticket(img)
        qr_code = crop_qr_code(ticket)
        result = qr_recognition(qr_code)
        ...
        """



if __name__ == "__main__":
    main(args.img_dir, args.ticket_dir, args.qr_dir)
