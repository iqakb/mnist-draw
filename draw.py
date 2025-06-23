import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np

class DigitRecognition(nn.Module):
    def __init__(self):
        super(DigitRecognition, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.d1 = nn.Linear(26 * 26 * 32, 128)
        self.d2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        
        x = x.flatten(start_dim=1)

        x = self.d1(x)
        x = F.relu(x)

        x = self.d2(x)
        x = F.softmax(x,dim=1)
        return x


model = DigitRecognition()    
model.load_state_dict(torch.load('./mnist-draw/saved_weights.pth'))
model.eval()  # Put model in eval mode (important!)

drag = False; predicted_class=-1; pred_text=''

def show_mouse_position(event, x, y, flags, param):
    global img_copy,drag,img_src,img,predicted_class,pred_text
    
    if event == cv.EVENT_LBUTTONDOWN:
        drag= True

    elif event == cv.EVENT_MOUSEMOVE:
        if drag:
            cv.circle(img_src,(x//10,y//10),1,255,-1,cv.LINE_AA)
            img = cv.resize(img_src, (280, 280), interpolation=cv.INTER_NEAREST)
            img_copy = img.copy()
            text = f"({x}, {y})"
            
            input = torch.from_numpy(img_src).unsqueeze(0).unsqueeze(0)
            input = input.to(torch.float32)/255

            with torch.no_grad():
                output = model(input)  # shape: [1, 10]
                predicted_class = output.argmax(dim=1).item()
                pred_text = f'{predicted_class}: {output[0][predicted_class]:.2f}'

            cv.putText(img_copy, text, (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(img_copy,pred_text,(5,30),cv.FONT_HERSHEY_SIMPLEX,1,(255),1)
            cv.imshow("Image", img_copy)
             
    elif event == cv.EVENT_LBUTTONUP:
        drag = False
        img_copy = img.copy()
        cv.putText(img_copy,pred_text,(5,30),cv.FONT_HERSHEY_SIMPLEX,1,(255),1)
        cv.imshow("Image", img_copy)

cv.namedWindow("Image")
cv.setMouseCallback("Image", show_mouse_position)

img_src = (np.zeros((28,28))).astype(np.uint8)
img = cv.resize(img_src, (280, 280), interpolation=cv.INTER_NEAREST)
img_copy = img.copy()
while True:
    cv.imshow("Image", img)
    if cv.waitKey(0) & 0xFF == 27:  # Press Esc to exit
        break
    if cv.waitKey(0) & 0xFF == 114:  # Press r to reset drawing
        img_src = (np.zeros((28,28))).astype(np.uint8)
        img = cv.resize(img_src, (280, 280), interpolation=cv.INTER_NEAREST)
        img_copy = img.copy()

cv.destroyAllWindows()