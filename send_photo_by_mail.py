#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2

# Import smtplib for the actual sending function
import smtplib

# Here are the email package modules we'll need
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


COMMASPACE = ', '
IMAGE_FILENAME = "photo.png"

FROM = "salajka@ntis.zcu.cz"
TO = "zelezny@kky.zcu.cz"

USERNAME = "salajka"
PASSWORD = "SecreteOrionPassword"

ESC_KEY = 27
ENTER_KEY = 13


def sendmail(
        image_filename=IMAGE_FILENAME, host="smtp.zcu.cz", port=465,
        username=USERNAME, password=PASSWORD, addrs_from=FROM, 
        addrs_to=TO):
    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg['Subject'] = 'Our family reunion'
    msg['From'] = FROM
    msg['To'] = TO
    msg.preamble = 'Our family reunion'

    # Open the files in binary mode.  Let the MIMEImage class automatically
    # guess the specific image type.
    fp = open(image_filename, 'rb')
    img = MIMEImage(fp.read())
    fp.close()
    msg.attach(img)

    # Send the email via our own SMTP server.
    s = smtplib.SMTP_SSL(host, port)
    s.login(username, password)
    s.sendmail(addrs_from, TO, msg.as_string())
    s.quit()


def main():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == ESC_KEY:
            break
        elif key == ENTER_KEY:
            cv2.imwrite(IMAGE_FILENAME, frame)
            sendmail()
            print "Mail has been sent!"
    cv2.destroyWindow("preview")


if __name__ == "__main__":
    main()
