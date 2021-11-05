__author__ = 'Frostime'

from win32 import win32api, win32gui, win32print
from win32.lib import win32con
import win32gui, win32ui, win32api, win32con
from win32.win32api import GetSystemMetrics
import tkinter as tk
from PIL import ImageGrab,Image
import cv2
from mss import mss
import numpy
import time
def get_real_resolution():

    hDC = win32gui.GetDC(0)
    # 横向分辨率
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    # 纵向分辨率
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    return w, h


def get_screen_size():

    w = GetSystemMetrics(0)
    h = GetSystemMetrics(1)
    return w, h


real_resolution = get_real_resolution()
screen_size = get_screen_size()

screen_scale_rate = round(real_resolution[0] / screen_size[0], 2)


class Box:

    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

    def isNone(self):
        return self.start_x is None or self.end_x is None

    def setStart(self, x, y):
        self.start_x = x
        self.start_y = y

    def setEnd(self, x, y):
        self.end_x = x
        self.end_y = y

    def box(self):
        lt_x = min(self.start_x, self.end_x)
        lt_y = min(self.start_y, self.end_y)
        rb_x = max(self.start_x, self.end_x)
        rb_y = max(self.start_y, self.end_y)
        return lt_x, lt_y, rb_x, rb_y

    def center(self):
        center_x = (self.start_x + self.end_x) / 2
        center_y = (self.start_y + self.end_y) / 2
        return center_x, center_y


class SelectionArea:

    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.area_box = Box()

    def empty(self):
        return self.area_box.isNone()

    def setStartPoint(self, x, y):
        self.canvas.delete('area', 'lt_txt', 'rb_txt')
        self.area_box.setStart(x, y)

        self.canvas.create_text(
            x, y - 10, text=f'({x}, {y})', fill='red', tag='lt_txt')

    def updateEndPoint(self, x, y):
        self.area_box.setEnd(x, y)
        self.canvas.delete('area', 'rb_txt')
        box_area = self.area_box.box()

        self.canvas.create_rectangle(
            *box_area, fill='black', outline='red', width=2, tags="area")
        self.canvas.create_text(
            x, y + 10, text=f'({x}, {y})', fill='red', tag='rb_txt')


class ScreenShot():

    def __init__(self, scaling_factor=2):
        self.win = tk.Tk()
        # self.win.tk.call('tk', 'scaling', scaling_factor)
        
        self.width = self.win.winfo_screenwidth()
        self.height = self.win.winfo_screenheight()

        #self.win.geometry(f"{self.width}x{self.height}+{self.width}+0")

        self.win.overrideredirect(True)
        
        self.win.attributes('-alpha', 0.25)
        #self.win.configure(bg = 'grey')
        self.is_selecting = False

        self.win.bind('<KeyPress-Escape>', self.exit)
        self.win.bind('<KeyPress-Return>', self.confirmScreenShot)
        self.win.bind('<Button-1>', self.selectStart)
        self.win.bind('<ButtonRelease-1>', self.selectDone)
        self.win.bind('<Motion>', self.changeSelectionArea)
        self.box_area =[]
        self.nonscale_barea = []
        self.canvas = tk.Canvas(self.win, width=self.width,
                                height=self.height)
        self.canvas.pack()
        self.area = SelectionArea(self.canvas)
        self.win.mainloop()

    def exit(self, event):
        self.win.destroy()

    def clear(self):
        self.canvas.delete('area', 'lt_txt', 'rb_txt')
        self.win.attributes('-alpha', 0)

    def captureImage(self):
        if self.area.empty():
            return None
        else:
            self.nonscale_barea  = [x  for x in self.area.area_box.box()]
            self.box_area = [x * screen_scale_rate for x in self.area.area_box.box()]
            self.clear()
            print(f'Grab: {self.box_area}')
            img = ImageGrab.grab(self.box_area)
            return img

    def confirmScreenShot(self, event):
        img = self.captureImage()
        #if img is not None:
        #    img.show()
        self.win.destroy()

    def selectStart(self, event):
        self.is_selecting = True
        self.area.setStartPoint(event.x, event.y)
        #print('Select', event)

    def changeSelectionArea(self, event):
        if self.is_selecting:
            self.area.updateEndPoint(event.x, event.y)
            #print(event)

    def selectDone(self, event):
        # self.area.updateEndPoint(event.x, event.y)
        self.is_selecting = False


# def main():
#     ScreenShot()

# def draw_area_box(box):
#     dc = win32gui.GetDC(0)
#     dcObj = win32ui.CreateDCFromHandle(dc)
#     hwnd = win32gui.WindowFromPoint((0,0))
#     _sx=int(box[0])
#     _sy=int(box[1])
#     _ex=int(box[2])
#     _ey=int(box[3])
   
#     print(box_cor)

#     while(True):
#         print("thread runnung?")
#         rect = win32gui.CreateRoundRectRgn(*box_cor, 4 , 4)
#         win32gui.RedrawWindow(hwnd, box_cor, rect, win32con.RDW_INVALIDATE)
#         for x in range(25):
#             win32gui.SetPixel(dc, _sx+x, _sy, red)
#             win32gui.SetPixel(dc, _sx+x, _ey, red)
#             win32gui.SetPixel(dc, _ex-x, _sy, red)
#             win32gui.SetPixel(dc, _ex-x, _ey, red)
#             for y in range(25):
#                 win32gui.SetPixel(dc, _sx, _sy+y, red)
#                 win32gui.SetPixel(dc, _ex, _sy+y, red)
#                 win32gui.SetPixel(dc, _sx, _ey-y, red)
#                 win32gui.SetPixel(dc, _ex, _ey-y, red)
def getbbox_area():
    return srs.box_area

if __name__ == '__main__':
    
    srs = ScreenShot() 
    #key = getkey()
    dc = win32gui.GetDC(0)
    dcObj = win32ui.CreateDCFromHandle(dc)
    hwnd = win32gui.WindowFromPoint((0,0))
    #monitor = GetSystemMetrics(0)
    print(srs.box_area)
    red = win32api.RGB(255, 0, 0) # Red
    #draw_cor = ()
    sx=int(srs.box_area[0])
    sy=int(srs.box_area[1])
    ex=int(srs.box_area[2])
    ey=int(srs.box_area[3])
    box_cor = (sx,sy,ex,ey)
    # nsx = int(srs.nonscale_barea[0])
    # nsy = int(srs.nonscale_barea[1])
    # nex = int(srs.nonscale_barea[2])
    # ney = int(srs.nonscale_barea[3])
    #_mon_area = [x for x in srs.area.area_box.box()]
    #mon = {'top' : _mon_area[0], 'left' : _mon_area[1], 'width' : _mon_area[2]-_mon_area[0], 'height' : _mon_area[3]-_mon_area[1]}
    mon = {'top' : sy, 'left' : sx, 'width' : ex-sx, 'height' : ey-sy}
    # print(screen_scale_rate)
    # print(mon)
    # print(box_cor)
    # print(srs.nonscale_barea)
    sct = mss()
    previous_time = 0
    #等待 thread 化
    while True:
        #print(srs.box_area)
        
        rect = win32gui.CreateRoundRectRgn(*box_cor, 4 , 4)
        win32gui.RedrawWindow(hwnd, box_cor, rect, win32con.RDW_INVALIDATE)
        #print(box_cor)
        for x in range(25):
            win32gui.SetPixel(dc, sx+x, sy, red)
            win32gui.SetPixel(dc, sx+x, ey, red)
            win32gui.SetPixel(dc, ex-x, sy, red)
            win32gui.SetPixel(dc, ex-x, ey, red)
            for y in range(25):
                win32gui.SetPixel(dc, sx, sy+y, red)
                win32gui.SetPixel(dc, ex, sy+y, red)
                win32gui.SetPixel(dc, sx, ey-y, red)
                win32gui.SetPixel(dc, ex, ey-y, red)
        # key = ord(getch())
        # if key == 27: #ESC
        #     break
        # print("draw_rect")
        #win32gui.RedrawWindow(hwnd, draw_cor, rect, win32con.RDW_INVALIDATE)
        sct_img = sct.grab(mon)
        #MSS 座標對應問題 如果解決可以提升 10 FPS
        #sct_img =ImageGrab.grab(srs.box_area)
        #frame = Image.frombytes( 'RGB', (sct.width, sct.height), sct.image )
        #frame = cv2.cvtColor(numpy.array(sct_img))
        #frame = Image.frombytes( 'RGB', (sct.width, sct.height), sct.image )
        #frame = np.array(frame)
        # image = image[ ::2, ::2, : ] # can be used to downgrade the input
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #sct_img = numpy.array(sct_img)
        tmp = numpy.array(sct_img)
        #sct_img = cv2.cvtColor(tmp)
        cv2.imshow ('frame', tmp)
        if cv2.waitKey ( 1 ) & 0xff == ord( 'q' ) :
            break
        txt1 = 'fps: %.1f' % ( 1./( time.time() - previous_time ))
        previous_time = time.time()
        print (txt1)
    #新增OPENCV 持續錄至該區域
    #之後 和 detect 做混合 
        
