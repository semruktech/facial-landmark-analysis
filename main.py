import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os

# Define line names and point names
line_names = ["Middle face height", "Lower face height", "Morphologic face height", "Upper face height", "Nose height",
              "Upper lip height", "Philtrum length", "Upper vermilion height", "Lower face height",
              "Lower lip height", "Cutaneous lower lip height", "Lower vermilion height", "Chin height",
              "Nasal bridge length", "Minimum frontal breadth", "Supraorbital breadth", "Nasal root width",
              "Biocular width", "Eye fissure width (right)", "Eye fissure width (left)",
              "Interocular distance (intercanthal width)", "Distance from the lower point of circumference of pupils",
              "Maximum facial breadth", "Bitragal width", "Nose width", "Nostril floor width (right)",
              "Nostril floor width (left)", "Philtrum width", "Mouth width (labial fissure width)",
              "Lower face width (mandible width)", "The distance between the subnasale and pronasale",
              "The distance between the tragion and nasion (right)",
              "The distance between the tragion and nasion (left)",
              "Middle face depth (maxillary depth) (right)", "Middle face depth (maxillary depth) (left)",
              "Lower face depth (mandibular depth) (right)", "Lower face depth (mandibular depth) (left)"]

line_names_points = ["gl-sn", "sn-gn", "n-gn", "n-st", "n-sn", "sn-st", "sn-ls", "ls-st", "st-gn", "st-sl",
                     "li-sl", "li-st", "sl-gn", "n-prn", "ft-ft", "fz-fz", "mf-mf", "ex-ex", "ex-en (right)",
                     "ex-en (left)", "en-en", "p-p", "z-z", "tr-tr", "al-al", "sa-sn (right)", "sa-sn (left)",
                     "cp-cp", "ch-ch", "go-go", "sn-prn", "tr-n (right)", "tr-n (left)", "tr-sn (right)",
                     "tr-sn (left)", "tr-gn (right)", "tr-gn (left)"]


def get_image(image_path, new_width=1280):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    new_height = int(new_width * (h / w))
    return cv2.resize(image, (new_width, new_height))


def calibrate_image(image):
    calibration_points = []

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            calibration_points.append((x, y))
            if len(calibration_points) == 2:
                cv2.destroyAllWindows()

    cv2.imshow("Select two points for calibration", image)
    cv2.setMouseCallback("Select two points for calibration", select_points)
    cv2.waitKey(0)

    if len(calibration_points) != 2:
        raise ValueError("Two points were not selected!")

    (x1, y1), (x2, y2) = calibration_points
    pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_distance_mm = float(input("Enter the actual distance (in mm) between two points: "))
    return real_distance_mm / pixel_distance


def get_landmark_pairs(facial_landmarks):
    # Define pairs of landmarks based on their indices in the Mediapipe model
    return [
        (facial_landmarks.landmark[9], facial_landmarks.landmark[94]),  # gl-sn
        (facial_landmarks.landmark[94], facial_landmarks.landmark[175]),  # sn-gn
        (facial_landmarks.landmark[8], facial_landmarks.landmark[175]),  # n-gn
        (facial_landmarks.landmark[8], facial_landmarks.landmark[13]),  # n-st
        (facial_landmarks.landmark[8], facial_landmarks.landmark[94]),  # n-sn
        (facial_landmarks.landmark[94], facial_landmarks.landmark[13]),  # sn-st
        (facial_landmarks.landmark[94], facial_landmarks.landmark[0]),  # sn-ls
        (facial_landmarks.landmark[0], facial_landmarks.landmark[13]),  # ls-st
        (facial_landmarks.landmark[13], facial_landmarks.landmark[175]),  # st-gn
        (facial_landmarks.landmark[13], facial_landmarks.landmark[18]),  # st-sl
        (facial_landmarks.landmark[17], facial_landmarks.landmark[18]),  # li-sl
        (facial_landmarks.landmark[17], facial_landmarks.landmark[13]),  # li-st
        (facial_landmarks.landmark[18], facial_landmarks.landmark[175]),  # sl-gn
        (facial_landmarks.landmark[8], facial_landmarks.landmark[4]),  # n-prn
        (facial_landmarks.landmark[67], facial_landmarks.landmark[297]),  # ft-ft
        (facial_landmarks.landmark[139], facial_landmarks.landmark[368]),  # fz-fz
        (facial_landmarks.landmark[122], facial_landmarks.landmark[351]),  # mf-mf
        (facial_landmarks.landmark[130], facial_landmarks.landmark[359]),  # ex-ex
        (facial_landmarks.landmark[130], facial_landmarks.landmark[243]),  # ex-en (right)
        (facial_landmarks.landmark[359], facial_landmarks.landmark[463]),  # ex-en (left)
        (facial_landmarks.landmark[243], facial_landmarks.landmark[463]),  # en-en
        (facial_landmarks.landmark[468], facial_landmarks.landmark[473]),  # p-p
        (facial_landmarks.landmark[117], facial_landmarks.landmark[346]),  # z-z
        (facial_landmarks.landmark[234], facial_landmarks.landmark[454]),  # tr-tr
        (facial_landmarks.landmark[64], facial_landmarks.landmark[294]),  # al-al
        (facial_landmarks.landmark[240], facial_landmarks.landmark[94]),  # sa-sn (right)
        (facial_landmarks.landmark[460], facial_landmarks.landmark[94]),  # sa-sn (left)
        (facial_landmarks.landmark[37], facial_landmarks.landmark[267]),  # cp-cp
        (facial_landmarks.landmark[61], facial_landmarks.landmark[308]),  # ch-ch
        (facial_landmarks.landmark[58], facial_landmarks.landmark[288]),  # go-go
        (facial_landmarks.landmark[94], facial_landmarks.landmark[4]),  # sn-prn
        (facial_landmarks.landmark[234], facial_landmarks.landmark[8]),  # tr-n (right)
        (facial_landmarks.landmark[454], facial_landmarks.landmark[8]),  # tr-n (left)
        (facial_landmarks.landmark[234], facial_landmarks.landmark[94]),  # tr-sn (right)
        (facial_landmarks.landmark[454], facial_landmarks.landmark[94]),  # tr-sn (left)
        (facial_landmarks.landmark[234], facial_landmarks.landmark[175]),  # tr-gn (right)
        (facial_landmarks.landmark[454], facial_landmarks.landmark[175])  # tr-gn (left)
    ]


def calculate_distances(facial_landmarks, image_width, image_height, mm_per_pixel, points):
    distances = []
    for pt1, pt2 in points:
        x1, y1 = int(pt1.x * image_width), int(pt1.y * image_height)
        x2, y2 = int(pt2.x * image_width), int(pt2.y * image_height)
        distance_in_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distances.append(round(distance_in_pixels * mm_per_pixel, 2))
    return distances


def save_and_display_distances(image, output_folder, points, distances, line_names_points):
    os.makedirs(output_folder, exist_ok=True)

    for idx, ((pt1, pt2), distance) in enumerate(zip(points, distances)):
        x1, y1 = int(pt1.x * image.shape[1]), int(pt1.y * image.shape[0])
        x2, y2 = int(pt2.x * image.shape[1]), int(pt2.y * image.shape[0])
        image_copy = image.copy()
        cv2.line(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(image_copy, f"{distance:.2f} mm", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imwrite(os.path.join(output_folder, f"{line_names_points[idx]}.jpg"), image_copy)


def save_results(output_folder, distances, line_names, line_names_points, image_name):
    data = {"Line Names": line_names, "Line Point Names": line_names_points, "Distances (mm)": distances}
    df = pd.DataFrame(data)
    df.to_excel(os.path.join(output_folder, f"{image_name}_results.xlsx"), index=False)


def main():
    image_path = input("Enter the image name (in images folder): ")
    image_name = os.path.splitext(image_path)[0]
    output_folder = os.path.join("./results", image_name)
    image = get_image(f"./images/{image_path}")

    mm_per_pixel = calibrate_image(image)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            print("No face detected.")
            return

        facial_landmarks = results.multi_face_landmarks[0]
        points = get_landmark_pairs(facial_landmarks)

        distances = calculate_distances(facial_landmarks, image.shape[1], image.shape[0], mm_per_pixel, points)
        save_and_display_distances(image, output_folder, points, distances, line_names_points)
        save_results(output_folder, distances, line_names, line_names_points, image_name)


if __name__ == "__main__":
    main()
