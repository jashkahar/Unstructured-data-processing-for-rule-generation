"""
Video Parser module for the Pharmaceutical Compliance Analysis System.
Extracts frames from videos and optionally transcribes audio.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pytesseract
from PIL import Image
from loguru import logger

from utils.types import Document, DocumentType


class VideoParser:
    """Parses video files to extract frames and optionally transcribe audio."""

    def __init__(self, config: Dict = None):
        """Initialize the video parser with configuration.

        Args:
            config: Configuration dictionary for the video parser
        """
        self.config = config or {}
        
        # Set pytesseract path if specified in config
        if "tesseract_path" in self.config:
            pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_path"]
            
        # Configure frame extraction settings
        self.frame_interval = self.config.get("frame_interval_seconds", 5)
        self.max_frames = self.config.get("max_frames", 10)
        self.transcribe_audio = self.config.get("transcribe_audio", False)
        
    def parse(self, video_path: str) -> Document:
        """Parse a video file, extract frames, and optionally transcribe audio.

        Args:
            video_path: Path to the video file

        Returns:
            Document object containing extracted content and metadata
        """
        logger.info(f"Parsing video file: {video_path}")
        
        path = Path(video_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        try:
            # Extract basic file metadata
            file_size = path.stat().st_size
            file_id = self._generate_document_id(path)
            
            # Open video file
            video = cv2.VideoCapture(str(path))
            
            # Get video metadata
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Extract frames and perform OCR
            extracted_text, frames_extracted = self._extract_frames_and_ocr(video, fps)
            
            # Transcribe audio if enabled
            transcription = ""
            if self.transcribe_audio:
                transcription = self._transcribe_audio(str(path))
                if transcription:
                    extracted_text += f"\n\nAUDIO TRANSCRIPTION:\n{transcription}"
            
            # Release the video capture object
            video.release()
            
            # Combine metadata
            metadata = {
                "format": "VIDEO",
                "fps": fps,
                "duration": duration,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "frames_extracted": frames_extracted,
                "transcription_available": bool(transcription),
                "timestamp": datetime.now().isoformat()
            }
            
            # Create document object
            document = Document(
                id=file_id,
                path=path,
                doc_type=DocumentType.VIDEO,
                content=extracted_text,
                metadata=metadata,
                processed_date=datetime.now(),
                file_size=file_size,
                duration=duration
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error parsing video {video_path}: {str(e)}")
            raise
    
    def _extract_frames_and_ocr(self, video: cv2.VideoCapture, fps: float) -> tuple[str, int]:
        """Extract frames from the video and perform OCR.
        
        Args:
            video: OpenCV video capture object
            fps: Frames per second of the video
            
        Returns:
            Tuple of (extracted text, number of frames processed)
        """
        extracted_text = ""
        frames_extracted = 0
        
        # Calculate frame interval based on FPS
        frame_interval_frames = int(fps * self.frame_interval)
        if frame_interval_frames <= 0:
            frame_interval_frames = 30  # Default to every 30 frames
        
        current_frame = 0
        ocr_config = self.config.get("ocr_config", "")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            while frames_extracted < self.max_frames:
                # Set video position to the target frame
                video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                
                # Read the frame
                success, frame = video.read()
                if not success:
                    break
                
                # Save frame to temporary file
                frame_path = os.path.join(temp_dir, f"frame_{frames_extracted}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Perform OCR on the frame
                try:
                    image = Image.open(frame_path)
                    frame_text = pytesseract.image_to_string(image, config=ocr_config)
                    
                    # Add frame text to the overall extracted text
                    timestamp = current_frame / fps if fps > 0 else 0
                    extracted_text += f"\n--- FRAME {frames_extracted} (Time: {timestamp:.2f}s) ---\n"
                    extracted_text += frame_text.strip() + "\n"
                    
                    frames_extracted += 1
                except Exception as e:
                    logger.warning(f"Error performing OCR on frame {current_frame}: {str(e)}")
                
                # Move to the next frame interval
                current_frame += frame_interval_frames
        
        return extracted_text, frames_extracted
    
    def _transcribe_audio(self, video_path: str) -> str:
        """Transcribe audio from the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Transcribed text or empty string if transcription failed
        """
        # This is a placeholder for actual audio transcription logic
        # In a real system, you would integrate with a speech-to-text service
        # like Google Speech-to-Text, AWS Transcribe, or similar
        
        logger.info(f"Audio transcription requested for {video_path}")
        logger.warning("Audio transcription not implemented - would integrate with a speech-to-text service")
        
        # Return a placeholder message
        return "[Audio transcription would be provided here in a real implementation]"
    
    def _generate_document_id(self, path: Path) -> str:
        """Generate a unique document ID based on the filename.

        Args:
            path: Path to the video file

        Returns:
            String ID for the document
        """
        # Extract base name without extension
        base_name = path.stem
        
        # Add timestamp suffix for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        return f"VID_{base_name}_{timestamp}"
