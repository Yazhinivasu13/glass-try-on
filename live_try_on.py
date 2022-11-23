import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def generate_frames(img_url):
    while True:
        success, img = camera.read()
        if not success:
            break
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True).process(img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for landmark in results.multi_face_landmarks:
                    left_eye_xs = []
                    left_eye_ys = []
                    right_eye_xs = []
                    right_eye_ys = []

                    for source_id, target_id in mp_face_mesh.FACEMESH_LEFT_EYE:
                        source = landmark.landmark[source_id]

                        left_eye_xs.append(int(source.x * img.shape[1]))
                        left_eye_ys.append(int(source.y * img.shape[0]))

                    for source_id, target_id in mp_face_mesh.FACEMESH_RIGHT_EYE:
                        source = landmark.landmark[source_id]

                        right_eye_xs.append(int(source.x * img.shape[1]))
                        right_eye_ys.append(int(source.y * img.shape[0]))

                    x_point = min(right_eye_xs)
                    y_point = min(right_eye_ys)

                    x, y = ((x_point - 50), (y_point - 50))
                    x_width = (max(left_eye_xs) - min(right_eye_xs)) + 100
                    y_height = (max(left_eye_ys) - min(left_eye_ys)) + 100

                    replace_img_base = cv2.imread(img_url)
                    replace_img_base = image_resize(replace_img_base, width=x_width, height=y_height)

                    h2, w2 = replace_img_base.shape[:2]

                    roi = img[y:y + h2, x:x + w2]
                    gray = cv2.cvtColor(replace_img_base, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    img_bg = cv2.bitwise_and(roi, roi, mask=mask)
                    img_fg = cv2.bitwise_and(replace_img_base, replace_img_base, mask=mask_inv)
                    final = cv2.add(img_bg, img_fg)
                    img[y:y + h2, x:x + w2] = final

                    ret, buffer = cv2.imencode('.jpg', img)
                    img = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    wr = width / float(w)
    dim = (int(w * wr), height)
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized
