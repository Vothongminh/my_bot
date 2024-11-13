import cv2
import numpy as np
import mediapipe as mp
import time
import torch
from torchreid.utils import FeatureExtractor

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load the OSNet model from Torchreid
extractor = FeatureExtractor(
    model_name='osnet_x0_25',
    model_path='/home/minh/Minh_code/models/osnet_x0_25_market1501.pth',  # Update to the correct path
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Đặt tp_features thành None khi khởi động
tp_features = None  # Placeholder for TP's features

def create_particles(x_range, y_range, N):
    particles = np.empty((N, 4))  # x, y, vx, vy
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2:4] = 0  # Initial velocities
    return particles

def predict(particles, std):
    particles[:, 0:2] += particles[:, 2:4] + np.random.normal(0, std, size=particles[:, 0:2].shape)
    particles[:, 2:4] += np.random.normal(0, std / 2, size=particles[:, 2:4].shape)

def update(particles, weights, target, R):
    dists = np.linalg.norm(particles[:, 0:2] - target, axis=1)
    weights *= np.exp(-(dists**2) / (2 * R**2))
    weights += 1.e-300
    weights /= np.sum(weights)

def check_hand_raised(landmarks, shoulder_idx, wrist_idx):
    shoulder = landmarks[shoulder_idx]
    wrist = landmarks[wrist_idx]
    return shoulder.y > wrist.y

def get_target_position(results, frame):
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    
    # Kiểm tra xem cả hai vai đều có thể nhìn thấy không
    if right_shoulder.visibility > 0.5 and left_shoulder.visibility > 0.5:
        target_position = [
            (right_shoulder.x + left_shoulder.x) / 2 * frame.shape[1],
            (right_shoulder.y + left_shoulder.y) / 2 * frame.shape[0]
        ]
    else:
        # Nếu không có vai, sử dụng trung điểm của hai hông
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        
        if right_hip.visibility > 0.5 and left_hip.visibility > 0.5:
            target_position = [
                (right_hip.x + left_hip.x) / 2 * frame.shape[1],
                (right_hip.y + left_hip.y) / 2 * frame.shape[0]
            ]
        else:
            # Nếu cả hai vai và hai hông đều không được phát hiện, quay về mũi làm dự phòng cuối cùng
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            target_position = [
                nose.x * frame.shape[1],
                nose.y * frame.shape[0]
            ]
    
    return target_position


def extract_osnet_features(model, frame, target_position):
    x, y = int(target_position[0]), int(target_position[1])
    crop = frame[y-50:y+50, x-50:x+50]
    if crop.size > 0:
        crop = cv2.resize(crop, (128, 256))  # Resize to match model input requirements
        crop = torch.Tensor(crop).permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize
        features = model(crop).detach().cpu().numpy()
        return features
    return None

def calculate_similarity(features1, features2):
    return np.dot(features1, features2.T) / (np.linalg.norm(features1) * np.linalg.norm(features2))

def detect_persons(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    persons = []
    if results.pose_landmarks:
        print("Detected landmarks:", len(results.pose_landmarks.landmark))
        center_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * frame.shape[1]
        center_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame.shape[0]
        persons.append((center_x, center_y))
    else:
        print("No persons detected.")
    return persons


def main():
    global tp_features
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    N = 1500
    std_pos = 5
    R = 30
    hand_raised_start = None
    required_duration = 5
    tp_identified = False
    target_position = None
    last_seen = time.time()
    lost_tp = False
    reidentifying = False
    similarity_threshold = 0.95  # Tăng ngưỡng để yêu cầu độ tương đồng cao hơn
    consecutive_match_count = 0  # Đếm số lần liên tiếp vượt ngưỡng để xác nhận TP
    match_requirement = 5  # Số khung hình liên tiếp cần đạt ngưỡng để xác nhận TP

    if not ret:
        print("Failed to grab frame")
        return

    particles = create_particles((0, frame.shape[1]), (0, frame.shape[0]), N)
    weights = np.ones(N) / N

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            last_seen = time.time()
            reidentifying = False  # Dừng ReID khi TP xuất hiện trong khung
            lost_tp = False

            if not tp_identified:
                # Giơ tay lần đầu để xác định TP
                if check_hand_raised(results.pose_landmarks.landmark, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_WRIST.value):
                    if hand_raised_start is None:
                        hand_raised_start = time.time()
                    elif time.time() - hand_raised_start >= required_duration:
                        target_position = get_target_position(results, frame)
                        tp_features = extract_osnet_features(extractor, frame, target_position)
                        tp_identified = True
                        print("TP identified with hand raised.")
                else:
                    hand_raised_start = None
            else:
                # Đã xác định TP ban đầu, không cần kiểm tra giơ tay nữa
                target_position = get_target_position(results, frame)

        elif tp_identified and (time.time() - last_seen <= 5):
            predict(particles, std_pos)

        else:
            if tp_identified:
                reidentifying = True
                print("Starting Re-identification...")

                persons = detect_persons(frame)
                if persons:
                    print(f"Detected {len(persons)} potential persons for ReID.")
                else:
                    print("No persons detected for ReID.")

                found_tp = False
                for person_position in persons:
                    person_features = extract_osnet_features(extractor, frame, person_position)
                    if person_features is not None:
                        similarity = calculate_similarity(tp_features, person_features)
                        print(f"Similarity: {similarity:.2f} (Threshold: {similarity_threshold})")
                        
                        if similarity > similarity_threshold:
                            consecutive_match_count += 1
                            if consecutive_match_count >= match_requirement:
                                print("ReID Successful!")
                                target_position = person_position
                                tp_identified = True
                                reidentifying = False
                                found_tp = True
                                consecutive_match_count = 0  # Reset đếm
                                break
                        else:
                            consecutive_match_count = 0  # Reset nếu không đạt ngưỡng
                            
                if not found_tp:
                    lost_tp = True

        if tp_identified:
            predict(particles, std_pos)
            update(particles, weights, target_position, R)
            indices = np.random.choice(N, N, p=weights)
            particles = particles[indices]
            weights.fill(1.0 / N)
            
            min_x, min_y = np.min(particles[:, 0]), np.min(particles[:, 1])
            max_x, max_y = np.max(particles[:, 0]), np.max(particles[:, 1])
            cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255), 2)

        for part in particles:
            cv2.circle(frame, (int(part[0]), int(part[1])), 2, (0, 255, 0), -1)

        if lost_tp:
            cv2.putText(frame, "Lose Target Person", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif reidentifying:
            cv2.putText(frame, "Re-identifying Target Person...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Particle Filter Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
