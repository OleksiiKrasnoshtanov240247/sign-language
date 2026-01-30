"""
Extract individual sign language letters from video and convert to GIFs

Edit the TIMESTAMPS list below with your letter boundaries.
"""

import subprocess
import json
from pathlib import Path
from typing import List, Tuple


# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

VIDEO_URL = "https://www.youtube.com/watch?v=GMi9qDSw2o8"
OUTPUT_DIR = "letters"

# Format: (letter_name, start_time_seconds, end_time_seconds)
TIMESTAMPS = [
    ("A", 15.0, 17.5),
    ("B", 18.0, 20.0),
    ("C", 20.0, 22.0),
    ("D", 22.0, 25.0),
    ("E", 25.5, 28.0),
    ("F", 28.0, 30.5),
    ("G", 31.0, 33.5),
    ("H", 33.5, 36.0),
    ("I", 36.0, 38.5),
    ("J", 38.5, 42.5),
    ("K", 42.5, 45.5),
    ("L", 46.0, 49.0),
    ("M", 49.0, 51.5),
    ("N", 51.5, 54.5),
    ("O", 55.0, 57.5),
    ("P", 57.5, 60.5),
    ("Q", 60.5, 63.5),
    ("R", 64.0, 66.5),
    ("S", 67.0, 69.0),
    ("T", 69.5, 71.5),
    ("U", 71.5, 73.5),
    ("V", 73.5, 76.0),
    ("W", 76.5, 78.5),
    ("X", 78.5, 80.0),
    ("Y", 80.5, 82.0),
    ("Z", 82.5, 84.0),
]

# GIF quality settings
GIF_FPS = 10        # Frame rate (8-15 recommended)
GIF_WIDTH = 480     # Width in pixels (height auto-scales)

# ============================================================================


class LetterExtractor:
    def __init__(self, video_url: str, output_dir: str):
        self.video_url = video_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.video_path = None
        
    def download_video(self) -> Path:
        """Download video using yt-dlp, or use local file if VIDEO_URL is a path"""
        
        # Check if VIDEO_URL is a local file path
        potential_path = Path(self.video_url)
        if potential_path.exists() and potential_path.is_file():
            print(f"Using local video file: {potential_path}")
            self.video_path = potential_path
            return self.video_path
        
        # Otherwise, download from URL
        print(f"Downloading video from {self.video_url}")
        
        output_template = str(self.output_dir / "source_video.%(ext)s")
        
        # Use default best format without restrictions
        # Works even without JavaScript runtime
        cmd = [
            "yt-dlp",
            "-o", output_template,
            self.video_url
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("\nError downloading video with yt-dlp.")
            print("\nAlternative: Download manually and update VIDEO_URL to local path:")
            print(f"  1. Go to: {self.video_url}")
            print("  2. Download video manually")
            print("  3. Change VIDEO_URL to: 'path/to/your/video.mp4'")
            raise
        
        self.video_path = list(self.output_dir.glob("source_video.*"))[0]
        print(f"Video downloaded: {self.video_path}")
        return self.video_path
    
    def extract_letter(self, letter: str, start: float, end: float):
        """
        Extract single letter segment and convert to GIF
        Uses two-pass approach for optimal quality:
        1. Generate color palette from video segment
        2. Create GIF using that palette
        """
        output_file = self.output_dir / f"{letter}.gif"
        palette_file = self.output_dir / f"{letter}_palette.png"
        
        duration = end - start
        print(f"Extracting {letter}: {start:.2f}s - {end:.2f}s ({duration:.2f}s)")
        
        # Pass 1: Generate optimal color palette
        palette_cmd = [
            "ffmpeg",
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(self.video_path),
            "-vf", f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,palettegen",
            "-y",
            str(palette_file)
        ]
        
        subprocess.run(palette_cmd, check=True, capture_output=True)
        
        # Pass 2: Create GIF using the palette
        gif_cmd = [
            "ffmpeg",
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(self.video_path),
            "-i", str(palette_file),
            "-filter_complex", f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse",
            "-y",
            str(output_file)
        ]
        
        subprocess.run(gif_cmd, check=True, capture_output=True)
        
        # Clean up palette file
        palette_file.unlink()
        
        print(f"  Created: {output_file}")
        return output_file
    
    def extract_all(self, timestamps: List[Tuple[str, float, float]]):
        """Extract all letters based on timestamps"""
        print(f"\nExtracting {len(timestamps)} letters...")
        print("=" * 50)
        
        for letter, start, end in timestamps:
            self.extract_letter(letter, start, end)
        
        print("=" * 50)
        print(f"Done! {len(timestamps)} GIFs created in {self.output_dir}/")


def main():
    extractor = LetterExtractor(VIDEO_URL, OUTPUT_DIR)
    
    # Download video
    extractor.download_video()
    
    # Extract all letters
    extractor.extract_all(TIMESTAMPS)


if __name__ == "__main__":
    main()