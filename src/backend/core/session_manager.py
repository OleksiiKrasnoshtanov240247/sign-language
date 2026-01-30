"""
User session management for sign language learning.
Updated for recording-based workflow.
"""
import time
import uuid
from typing import Optional, Dict
from collections import Counter
from src.backend.core.config import (
    MAX_ATTEMPT_TIME,
    HINT_THRESHOLD_ATTEMPTS,
    MAX_HINTS,
    CONFIDENCE_THRESHOLD,
    RECORDING_DURATION
)
from src.backend.core.letter_sequence import LetterSequence
from src.backend.core.tutorial_manager import TutorialManager


class UserSession:
    """Manages state for a single user learning session."""
    
    def __init__(self, session_id: Optional[str] = None, mode: str = "sequential"):
        """Initialize new session."""
        self.id = session_id or str(uuid.uuid4())
        self.mode = mode  # "sequential" or "random"
        self.letter_sequence = LetterSequence(mode=mode, include_dynamic=True)
        self.tutorial_manager = TutorialManager()
        
        # Current state
        self.current_letter = self.letter_sequence.get_next_letter()
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0
        
        # Recording state
        self.is_recording = False
        self.recording_start_time = None
        self.recording_predictions = []  # Collect predictions during recording
        
        # Statistics
        self.total_correct = 0
        self.total_attempts = 0
        self.completed_letters = []
        
        # Flags
        self.is_active = True
        self.show_hint = False
        self.hint_message = ""
        
    def start_recording(self):
        """Start a new recording attempt."""
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recording_predictions = []
        print(f"📹 Started recording for letter '{self.current_letter}'")
        
    def add_prediction(self, predicted_letter: str, confidence: float) -> Optional[Dict]:
        """
        Add a prediction during recording.
        
        Args:
            predicted_letter: Letter predicted by model
            confidence: Confidence score (0-1)
            
        Returns:
            None during recording, dict with result when recording complete
        """
        if not self.is_recording:
            return None
        
        # Store prediction
        self.recording_predictions.append({
            'letter': predicted_letter,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Check if recording duration reached
        elapsed = time.time() - self.recording_start_time
        if elapsed >= RECORDING_DURATION:
            return self.finish_recording()
        
        return None
    
    def finish_recording(self) -> Dict:
        """
        Finish recording and determine result using majority vote.
        
        Returns:
            dict with match status and next actions
        """
        self.is_recording = False
        self.attempt_count += 1
        self.total_attempts += 1
        
        if not self.recording_predictions:
            return {
                'match': False,
                'success': False,
                'timeout': False,
                'message': 'No hand detected during recording',
                'show_hint': False
            }
        
        # Filter by confidence threshold
        valid_predictions = [
            p for p in self.recording_predictions 
            if p['confidence'] >= CONFIDENCE_THRESHOLD
        ]
        
        if not valid_predictions:
            return self._handle_failed_attempt("Low confidence predictions")
        
        # Majority vote
        letters = [p['letter'] for p in valid_predictions]
        letter_counts = Counter(letters)
        most_common_letter, count = letter_counts.most_common(1)[0]
        
        # Calculate average confidence for the most common letter
        avg_confidence = sum(
            p['confidence'] for p in valid_predictions 
            if p['letter'] == most_common_letter
        ) / count
        
        print(f" Predictions: {dict(letter_counts)} | Winner: {most_common_letter} ({count}/{len(valid_predictions)})")
        
        # Check if it matches target
        match = (most_common_letter == self.current_letter)
        
        if match:
            return self._handle_success(most_common_letter, avg_confidence)
        else:
            return self._handle_failed_attempt(
                f"Detected '{most_common_letter}' instead of '{self.current_letter}'"
            )
    
    def _handle_success(self, predicted_letter: str, confidence: float) -> Dict:
        """Handle successful letter recognition."""
        self.total_correct += 1
        self.completed_letters.append(self.current_letter)
        self.letter_sequence.mark_completed(self.current_letter)
        
        print(f" Success! Correctly signed '{self.current_letter}' (confidence: {confidence:.2f})")
        
        # Advance to next letter
        next_letter = self.letter_sequence.get_next_letter(self.current_letter)
        self.current_letter = next_letter
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0
        self.show_hint = False
        
        return {
            'match': True,
            'success': True,
            'timeout': False,
            'predicted_letter': predicted_letter,
            'confidence': confidence,
            'message': f'Correct! Moving to letter {next_letter}',
            'next_letter': next_letter,
            'show_hint': False
        }
    
    def _handle_failed_attempt(self, reason: str) -> Dict:
        """Handle failed recognition attempt."""
        # Check timeout
        elapsed = time.time() - self.letter_start_time
        if elapsed >= MAX_ATTEMPT_TIME:
            print(f" Timeout on letter '{self.current_letter}'")
            return self._handle_timeout()
        
        # Check if hint should be shown
        show_hint = self.tutorial_manager.should_show_hint(
            self.attempt_count, self.hints_shown, MAX_HINTS
        )
        
        if show_hint:
            self.hints_shown += 1
            self.show_hint = True
            self.hint_message = self.tutorial_manager.get_hint_message(
                self.current_letter, self.hints_shown
            )
        else:
            self.show_hint = False
        
        print(f" Failed attempt {self.attempt_count}: {reason}")
        
        return {
            'match': False,
            'success': False,
            'timeout': False,
            'message': reason,
            'show_hint': self.show_hint,
            'hint_message': self.hint_message if self.show_hint else ''
        }
    
    def _handle_timeout(self) -> Dict:
        """Handle timeout - move to next letter."""
        next_letter = self.letter_sequence.get_next_letter(self.current_letter)
        self.current_letter = next_letter
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0
        self.show_hint = False
        
        return {
            'match': False,
            'success': False,
            'timeout': True,
            'message': f'Time up! Moving to letter {next_letter}',
            'next_letter': next_letter,
            'show_hint': False
        }
    
    def get_tutorial_url(self) -> Optional[str]:
        """Get tutorial GIF URL for current letter."""
        return self.tutorial_manager.get_tutorial_url(self.current_letter)
    
    def get_time_remaining(self) -> float:
        """Get seconds remaining for current letter."""
        elapsed = time.time() - self.letter_start_time
        return max(0, MAX_ATTEMPT_TIME - elapsed)
    
    def get_progress(self) -> dict:
        """Get current session progress."""
        accuracy = (self.total_correct / self.total_attempts * 100) if self.total_attempts > 0 else 0
        
        return {
            'current_letter': self.current_letter,
            'total_correct': self.total_correct,
            'total_attempts': self.total_attempts,
            'accuracy': accuracy,
            'completed_letters': self.completed_letters,
            'attempt_count': self.attempt_count,
            'time_remaining': self.get_time_remaining(),
            'tutorial_url': self.get_tutorial_url(),
            'is_recording': self.is_recording,
            'mode': self.mode
        }
    
    def set_mode(self, mode: str):
        """Change letter sequence mode (sequential or random)."""
        if mode not in ["sequential", "random"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sequential' or 'random'.")
        
        self.mode = mode
        self.letter_sequence.mode = mode
        print(f"📝 Mode changed to: {mode}")
    
    def skip_letter(self) -> Dict:
        """
        Skip the current letter and move to the next one.
        
        Returns:
            Dict with skip result information
        """
        skipped_letter = self.current_letter
        
        # Move to next letter
        self.current_letter = self.letter_sequence.get_next_letter(self.current_letter)
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0
        self.is_recording = False
        self.recording_predictions = []
        self.show_hint = False
        
        # Mark as attempt (not counted as correct)
        self.total_attempts += 1
        
        print(f"⏭️ Skipped letter '{skipped_letter}' -> '{self.current_letter}'")
        
        return {
            "match": False,
            "success": False,
            "timeout": False,
            "skipped": True,
            "message": f"Skipped {skipped_letter}, now showing {self.current_letter}",
            "next_letter": self.current_letter,
            "show_hint": False
        }
    
    def reset(self):
        """Reset session to beginning."""
        self.letter_sequence.reset()
        self.current_letter = self.letter_sequence.get_next_letter()
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0
        self.is_recording = False
        self.recording_predictions = []
        self.total_correct = 0
        self.total_attempts = 0
        self.completed_letters = []
        self.is_active = True


class SessionManager:
    """Manages multiple user sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
    
    def create_session(self, session_id: Optional[str] = None, mode: str = "sequential") -> UserSession:
        """Create new user session."""
        session = UserSession(session_id, mode=mode)
        self.sessions[session.id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get existing session by ID."""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_inactive_sessions(self, max_age: int = 3600):
        """Remove inactive sessions older than max_age seconds."""
        current_time = time.time()
        to_remove = []
        
        for sid, session in self.sessions.items():
            if not session.is_active:
                elapsed = current_time - session.letter_start_time
                if elapsed > max_age:
                    to_remove.append(sid)
        
        for sid in to_remove:
            del self.sessions[sid]
        
        return len(to_remove)
