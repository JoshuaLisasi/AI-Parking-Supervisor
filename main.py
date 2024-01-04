from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import Polygon

# Load the Ultralytics YOLOv8 model
model = YOLO('yolov8n.pt')

# Taking input video from file
video_path = './videos/parkingvideo.mp4'
cap = cv2.VideoCapture(video_path)

font = cv2.FONT_HERSHEY_SIMPLEX

# Get position of point when double click with mouse 
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y
        print (f'{mouseX},{mouseY}')
        return(mouseX,mouseY)
    
def draw_text(img, text,
        font=cv2.FONT_HERSHEY_PLAIN,
        pos=(0, 0),
        font_scale=3,
        font_thickness=2,
        text_color=(0, 128, 255),
        text_color_bg=(0, 0, 0)
        ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

# ParkingSpot definied by the shape (polygon) of the parking spot and the id
class ParkingSpot:
    def __init__(self,first_point,second_point, third_point,fourth_point,id):
        self.id = id
        self.points = [first_point,second_point,third_point,fourth_point]
        self.cars = []
        self.free = True

        self.polygon = Polygon(self.points)
        self.polylines =  np.array(self.points).reshape(-1,1,2)
        
    def draw(self,frame,isFree):
        color = ()
        if isFree : 
            color = (0,255,0)
            # frame = cv2.polylines(frame,[self.polylines],True,(0,255,0),thickness=2)
        else  : 
            color = (0,0,255)
            # frame = cv2.polylines(frame,[self.polylines],True,(0,0,255),thickness=2)

        frame = cv2.polylines(frame,[self.polylines],True,color,thickness=2)
        frame = cv2.putText(frame,f'{self.id}',self.points[0], font, 1,(255,255,255),2,cv2.LINE_AA)
        return frame
    
    def calculate_iou(self,box):
        box_x_min = int(box.xyxy[0][0])
        box_y_min = int(box.xyxy[0][1])
        box_x_max = int(box.xyxy[0][2])
        box_y_max = int(box.xyxy[0][3])

        polygon_box = Polygon([(box_x_min, box_y_min), (box_x_max, box_y_min),(box_x_max, box_y_max), (box_x_min, box_y_max)])
        intersect = self.polygon.intersection(polygon_box).area
        union = self.polygon.union(polygon_box).area
        iou = intersect / union
        if iou > 0.15:
            return True
        return False

# Lists of ParkingSpot defined but their polygons
p1 = ParkingSpot([331,889],[572,604],[946,603],[737,964],1)
p2 = ParkingSpot([5,818],[191,533],[467,583],[249,881],2)
p3 = ParkingSpot([828,957],[1066,578],[1410,608],[1307,988],3)
p4 = ParkingSpot([1387,969],[1481,605],[1878,629],[1877,992],4)
p5 = ParkingSpot([1326,112],[1599,114],[1557,259],[1231,263],5)
p6 = ParkingSpot([1601,258],[1626,119],[1900,124],[1886,270],6)
p7 = ParkingSpot([903,252],[1007,102],[1279,113],[1193,250],7)
p8 = ParkingSpot([547,249],[721,86],[917,99],[853,248],8)
p9 = ParkingSpot([232,245],[412,102],[627,115],[497,242],9)
p10 = ParkingSpot([1,212],[120,76],[293,113],[190,231],10)
parking = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10]

ret = True
debug = False
cv2.namedWindow('frame')
cv2.setMouseCallback('frame',draw_circle)
names = model.names

nbr_cars = 0
nbr_free_spot = 0 
# running while loop just to make sure that 
# our program keep running until we stop it 
while ret:
    ret , frame = cap.read()
    results = model.track(frame, persist=True , tracker = 'botsort.yaml',conf=0., verbose=False)
    r = results[0]

    # plotting bounding boxes
    # if debug == True:
    frame = r.plot(conf=True, line_width=None, font_size=None,
            font='Arial.ttf', pil=False, img=None,
            im_gpu=None, kpt_radius=0, kpt_line=False, 
            labels=True, boxes=False, masks=False, probs=True)
        
    
    temp_nbr_free_spot = 0
    boxes = r.boxes
    for spot in parking:
        temp_nbr_cars = 0
        # getting previous cars in the ParkingSpot
        temp_cars = spot.cars.copy()
        for box in boxes:
            if(names[box.cls.item()] != ('car' or 'truck' or 'motorbike' or 'truck' or 'bus' or 'bicyle') ):
                continue
            temp_nbr_cars +=1
            x_c = int(box.xywh[0][0])
            y_c = int(box.xywh[0][1])
            frame = cv2.putText(frame,'Car',(x_c,y_c), font, 2,(0,128,255),2,cv2.LINE_AA)
            
            # Calculate IOU beteween the BpundingBoxes and the ParkingSpots
            isTaken = spot.calculate_iou(box)

            if( isTaken== True):
                if(spot.free):
                    if debug == True:  
                        print(f"ID added {box.id.item()} to spot : {spot.id}")
                    temp_cars.append(box.id)
                    spot.free = False

            # Removing the car if the ParkingSpot is not taken (free)
            # And the car is still in the ParkingSpot's tracked car
            elif box.id in temp_cars:
                if debug == True:
                    print(f"ID removed {box.id.item()} from spot : {spot.id}")
                temp_cars.remove(box.id)
                spot.free = True

            spot.cars = temp_cars
        

        if spot.free == True:
            temp_nbr_free_spot +=1

        if debug == True:
            print(f'Number of car in spot number {spot.id} : {len(spot.cars)}')
            

        # Draws the ParkingSpot in the right color
        if  len(spot.cars):
            frame = spot.draw(frame,False)
        else : 
            frame = spot.draw(frame,True)

    nbr_cars = temp_nbr_cars       
    nbr_free_spot =  temp_nbr_free_spot
    text = f'Number of cars : {nbr_cars} | Number of free spots : {nbr_free_spot}'
    offsetX,offsetY = cv2.getTextSize(text,font,2,3)
    draw_text(frame, text)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


