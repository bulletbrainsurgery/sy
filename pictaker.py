import requests
from PIL import Image, ImageOps
import pywin32_system32

import os, sys, io
import time, string
from timeit import default_timer as timer

savedir = 'C:\\Users\\David\\test\\syrnia\\captchas'

url = "http://www.syrnia.com/workimage.php"
data = "/theGame/ajax/centerContent.php HTTP/1.1".encode('utf-8')
headers = {
    "Host": "www.syrnia.com", "Connection": "keep-alive", "Content-Length": "20", "Origin": "http://www.syrnia.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
    "Content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Accept": "text/javascript, text/html, application/xml, text/xml, */*", "X-Prototype-Version": "1.6.1",
    "X-Requested-With": "XMLHttpRequest", "DNT": "1", "Referer": "http://www.syrnia.com/game.php",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
    "Cookie": "gamescreenwidth=1024; PHPSESSID=lq1k2oacc1ivmvd4kamc54lfe6; Syrnia=neggie333"}
suffixnum = string.digits + string.ascii_lowercase  # "0123456789abcdefghijklmnopqrstuvwxyz"


def getimg(code):
    os.chdir(savedir)
    savelist = []
    times = []
    counter = 0
    starttime = timer()
    with requests.Session() as s:
        s.headers = headers
        for i in range(0, 36):
            start = timer()
            ticked = False
            filename = "%s %s.png" % (code, suffixnum[i])
            if not os.path.isfile(filename):
                resp = s.post(url, data=data)
                # r = requests.post(url, data = data, headers = headers)
                pic = resp.content
                if 12000 < len(pic) < 20000:
                    savelist.append((filename, pic))
                else:
                    # if len(pic) == 0: 
                    # win32api.MessageBox(0, 'replace session id', 'error :(', 0x00001000)
                    # raise ValueError("Nothing returned, check session id?"%len(pic)) #  Should this happen? If it cuts off halfway it should keep going?
                    # break # for loop: line 36
                    # else:    # write contents into an error log: try bytes, then text. 
                    # errorlogname = time.time()
                    # try:
                    # with open()
                    pywin32_system32.MessageBox(0, 'didn\'t work', 'error :(', 0x00001000)
                    raise ValueError("File size is out of range: %s" % len(pic))

                counter += 1
                ticked = True
            end = timer()
            times.append(end - start)
            if not counter % 6 and counter * ticked:  # prints every 6 if counter has ticked and >0
                print("saved %s codes" % counter)
    endtime = timer()
    print('done (%s)' % code)
    # print('Total: %s. Min: %s Max: %s Avg: %s'%(endtime-starttime, min(times), max(times), sum(times)/len(times)))
    print('Total: {:.5f}. Avg: {:.5f}\nMin: {:.5f} Max: {:.5f}'.format(endtime - starttime, sum(times) / len(times),
                                                                       min(times), max(times)))

    pywin32_system32.MessageBox(0, 'do the captcha you sausage', 'pics saved', 0x00001000)
    import denoise
    for pic in savelist:
        filename = pic[0]
        picture = Image.open(io.BytesIO(pic[1]))
        picture = picture.convert("RGB")
        result = denoise.denoise(pic=picture, filename=filename)
        result = result.crop((0, 0, 150, 55))
        result.save(filename)
    if len(savelist) > 0:
        print('converted & saved')


def main():
    codeformat = True
    while codeformat:
        if len(sys.argv) > 1 and sys.argv[1]:  # is not none
            code = sys.argv[1]
        else:
            code = input("What's the code?\n> ")
        try:
            int(code)  # make sure it's a number
            if len(code) in (3, 4):  # make sure it's the right length
                codeformat = False
            else:
                sys.argv[1] = None
                raise Exception
        except Exception:
            print("Try again: code must be a 3 or 4 digit number.")
    getimg(code)


if __name__ == "__main__":
    main()
