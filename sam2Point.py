import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor

# Global variables to store clicked points
positive_points = []
negative_points = []
frame = None

# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        positive_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green for positive points
        cv2.imshow('First Frame', frame)
    elif event == cv2.EVENT_RBUTTONDOWN:
        negative_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red for negative points
        cv2.imshow('First Frame', frame)

def process_video_with_sam2(
    video_path: str,
    sam2_checkpoint: str = "D:\\AI_Graph\\sam2\\checkpoints\\sam2.1_hiera_base_plus.pt",
    model_cfg: str = "D:\\AI_Graph\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_b+.yaml"
):
    """
    Process a video with SAM2 to generate segmentation masks.
    
    Args:
        video_path: Path to the input video file
        sam2_checkpoint: Path to SAM2 checkpoint file
        model_cfg: Path to SAM2 model config file
    """
    global positive_points, negative_points, frame
    
    # Read video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # Display the first frame for point selection
    cv2.imshow('First Frame', frame)
    cv2.setMouseCallback('First Frame', click_event)
    print("Left-click for positive points, Right-click for negative points. Press 'q' to finish.")

    # Wait for user to finish clicking
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

    # Load SAM2 model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)

    # Prepare input points and labels
    input_points = np.array(positive_points + negative_points, dtype=np.float32)
    input_labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)

    # Process the first frame
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=input_points,
        labels=input_labels,
    )

    # Get the mask for the first frame
    first_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    if first_mask.ndim == 3:
        first_mask = first_mask.squeeze(0)

    # Prepare video writer for the mask video
    output_video_path = video_path.replace('.mp4', '_mask_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Write the first frame mask
    if first_mask.ndim == 2:
        first_mask = (first_mask * 255).astype(np.uint8)
        out_video.write(first_mask)
    else:
        print(f"Error: Invalid mask dimensions. Expected 2D, got {first_mask.ndim}D.")
        cap.release()
        return

    # Process the rest of the frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    frames_processed = 0
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                
                if mask.ndim != 2:
                    print(f"Skipping frame {out_frame_idx}: Invalid mask shape {mask.shape}")
                    continue
                
                if mask.shape != (frame_height, frame_width):
                    mask = cv2.resize(mask, (frame_width, frame_height))
                
                mask_uint8 = (mask * 255).astype(np.uint8)
                if mask_uint8.shape == (frame_height, frame_width):
                    out_video.write(mask_uint8)
                    frames_processed += 1
                else:
                    print(f"Skipping frame {out_frame_idx}: Final shape mismatch {mask_uint8.shape}")
                
                if frames_processed >= max_frames:
                    break
                    
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving partial results...")
            break
        except Exception as e:
            print(f"Critical error at frame {frames_processed}: {str(e)}")
            break

    # Release resources
    cap.release()
    out_video.release()
    print(f"Mask video saved as {output_video_path}")

if __name__ == "__main__":
    # Example usage with hardcoded path
    process_video_with_sam2(video_path=r"D:\\AI_Graph\\sam2output\\3.mp4")