import cv2
from yolov5_detect import load_model, parse_opt, predictImg


def InitNet():
    model = load_model()
    opt = parse_opt()
    return model, opt


def showCamera(saveVideo=False):
    model, opt = InitNet()

    if saveVideo:
        # fourcc = cv2.VideoWriter.fourcc('X','2','6','4')
        # fourcc = cv2.VideoWriter.fourcc('v','p','8','0')
        fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter('output.mp4', fourcc, 25.0, (640, 480))

    cap = cv2.VideoCapture(0)  # CAP_DSHOW
    # cap.set(cv2.CAP_PROP_FPS, 20)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps=', fps)

    while(True):
        ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        frame = predictImg(model, frame, opt)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if saveVideo:
            out.write(frame)

    if saveVideo:
        out.release()

    cap.release()
    cv2.destroyAllWindows()


def main():
    showCamera()


if __name__ == '__main__':
    main()
