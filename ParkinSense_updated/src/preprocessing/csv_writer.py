import csv
import logging
import os
from typing import List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# List of MediaPipe Pose landmark names in order of their indices (0 to 32)
LANDMARK_NAMES: List[str] = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

class PoseCSVWriter:
    """Writes MediaPipe landmarks to a CSV file for gait analysis."""
    
    def __init__(self, filepath: str = "data/features/pose_landmarks.csv") -> None:
        """
        Initializes the writer and creates the output file/directory structure.
        
        Args:
            filepath: Target file path to write landmarks.
        """
        self.filepath = filepath
        self.file = None
        self.writer = None
        self._initialize_file()

    def _initialize_file(self) -> None:
        """Creates the directory and initial CSV file with headers if it doesn't exist."""
        try:
            # Ensure the directory exists
            directory = os.path.dirname(self.filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)

            # Open file for writing
            self.file = open(self.filepath, mode="w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            
            # Write header
            self.writer.writerow([
                "frame",
                "landmark_name",
                "landmark_index",
                "x",
                "y",
                "z",
                "visibility"
            ])
            logger.info(f"Initialized CSV writer at: {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV file: {e}")
            raise

    def write_frame_landmarks(self, frame_no: int, landmarks: Any) -> None:
        """
        Writes all landmarks of a single frame to the CSV file.
        
        Args:
            frame_no: The frame index.
            landmarks: MediaPipe normalized landmarks list.
        """
        if not self.writer:
            raise RuntimeError("CSV writer is not initialized or has been closed.")
        
        try:
            for idx, landmark in enumerate(landmarks.landmark):
                name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"UNKNOWN_{idx}"
                self.writer.writerow([
                    frame_no,
                    name,
                    idx,
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])
        except Exception as e:
            logger.error(f"Error writing frame landmarks for frame {frame_no}: {e}")

    def close(self) -> None:
        """Closes the CSV file."""
        if self.file:
            try:
                self.file.close()
                logger.info(f"Closed CSV writer. Landmarks saved to: {self.filepath}")
            except Exception as e:
                logger.error(f"Error closing CSV writer file: {e}")
            finally:
                self.file = None
                self.writer = None
