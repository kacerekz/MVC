import cv2
import numpy as np
from collections import defaultdict

SAMPLE_COUNT = 20
DISPLAY_SCALE = 0.5

full_img = cv2.imread(r"Run1024/frame_0000.png")
sub_img = cv2.imread(r"Arm L.png")

if full_img is None or sub_img is None:
    raise ValueError("Could not load images. Check file paths.")

sub_channels = cv2.split(sub_img)
full_channels = cv2.split(full_img)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# Store matches: (channel, queryIdx) -> list of (match, full_kp)
match_groups = defaultdict(list)
# Map (channel, queryIdx) -> kp1 for sub-image keypoints
keypoints_map = {}

# For each channel, run SIFT and store matches
for c in range(3):
    gray_sub = sub_channels[c]
    gray_full = full_channels[c]

    kp1, des1 = sift.detectAndCompute(gray_sub, None)
    kp2, des2 = sift.detectAndCompute(gray_full, None)

    if des1 is None or des2 is None:
        continue

    matches = bf.knnMatch(des1, des2, k=2)

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            match_groups[(c, m.queryIdx)].append((m, kp2))
            keypoints_map[(c, m.queryIdx)] = kp1

# === Find queryIdx with matches in all 3 channels ===
# Build map: queryIdx -> set of channels where it appears
queryidx_channels = defaultdict(set)
for (ch, qidx) in match_groups.keys():
    queryidx_channels[qidx].add(ch)

# Filter keys to only those present in all 3 channels
triple_channel_queryidxs = [qidx for qidx, chset in queryidx_channels.items() if len(chset) == 3]

# Build final samples: for each queryIdx in triple_channel_queryidxs,
# collect all matches from all 3 channels combined into one list
samples = []
for qidx in triple_channel_queryidxs:
    combined_matches = []
    combined_kp1 = None
    for c in range(3):
        key = (c, qidx)
        if key in match_groups:
            combined_matches.extend(match_groups[key])
            combined_kp1 = keypoints_map[key]  # all share same queryIdx in sub-image channel
    if combined_matches and combined_kp1 is not None:
        samples.append((qidx, combined_matches, combined_kp1))

# Shuffle and sample from these
np.random.shuffle(samples)
samples = samples[:min(SAMPLE_COUNT, len(samples))]

current_index = 0

def draw_window(index, scale=DISPLAY_SCALE):
    qidx, matches, kp1 = samples[index]

    sub_pt = tuple(map(int, kp1[qidx].pt))

    sub_disp = cv2.resize(sub_img, (0, 0), fx=scale, fy=scale)
    full_disp = cv2.resize(full_img, (0, 0), fx=scale, fy=scale)

    sub_pt_scaled = tuple(int(p * scale) for p in sub_pt)
    cv2.circle(sub_disp, sub_pt_scaled, 6, (0, 255, 0), -1)

    # Draw all full-image matches (across channels)
    for m, kp2 in matches:
        pt_full = tuple(map(int, kp2[m.trainIdx].pt))
        pt_full_scaled = tuple(int(p * scale) for p in pt_full)
        cv2.circle(full_disp, pt_full_scaled, 5, (0, 0, 255), -1)

    height = max(sub_disp.shape[0], full_disp.shape[0])
    total_width = sub_disp.shape[1] + full_disp.shape[1]
    combined = np.zeros((height, total_width, 3), dtype=np.uint8)
    combined[:sub_disp.shape[0], :sub_disp.shape[1]] = sub_disp
    combined[:full_disp.shape[0], sub_disp.shape[1]:] = full_disp

    cv2.putText(combined, f"Sample {index+1}/{len(samples)} (Triple-channel matches)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("SIFT Multi-Channel Consensus Viewer", combined)

while True:
    draw_window(current_index)
    key = cv2.waitKey(0)
    if key == 27:
        break
    elif key == ord('d') or key == 83:
        current_index = (current_index + 1) % len(samples)
    elif key == ord('a') or key == 81:
        current_index = (current_index - 1) % len(samples)

cv2.destroyAllWindows()
