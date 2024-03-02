import cv2
import mediapipe as mp
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import time

cap = cv2.VideoCapture(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('/Users/borosabel/Documents/Uni/Projects/Signlanguage/PyCharm-mediapipe/model.pth', map_location=torch.device('cpu'))
model.eval()
model = model.to(device)
#
class_names = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z', 26:'del', 27:'nothing', 28:'space'}
#
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5188, 0.4989, 0.5142], [0.2023, 0.2314, 0.2404])
    ])
}
mpHands = mp.solutions.hands
mpHands.Hands()
hands = mpHands.Hands(False, 1)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

counter = 0
prediction = ""

while True:
    success, img = cap.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    h, w, c = img.shape
    index_min = 0
    index_max = 0
    if(results.multi_hand_landmarks):
        for hand_landmark in results.multi_hand_landmarks:
            print(type(hand_landmark))
            mincx, mincy = h + 1, w + 1
            maxcx, maxcy = -1, -1
            for id, lm in enumerate(hand_landmark.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if(cx < mincx):
                    mincx = cx
                    index_min = id
                if(cx > maxcx):
                    index_max = id
                    maxcx = cx
                if(cy < mincy):
                    index_min = id
                    mincy = cy
                if(cy > maxcy):
                    index_max = id
                    maxcy = cy
            cv2.rectangle(img, (mincx - 25, mincy - 25), (maxcx + 25, maxcy + 25), (0, 255, 0), 1)
            image = Image.fromarray(img)
            image1 = image.crop((int(mincx) - 125, int(mincy) - 135, int(maxcx) + 135, int(maxcy) + 125))
            input = data_transforms['test'](image1)
            input = input.view(1, 3, 224, 224)
            input = input.to(device)
            output = model(input)
            prediction = torch.max(output, 1)
            prediction = output.max(dim=1)[1].numpy()[0]
            cv2.putText(img, class_names[prediction], (40, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
            mpDraw.draw_landmarks(img, hand_landmark, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
