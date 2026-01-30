"""
Start script with proper error handling and logging.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    print("=" * 60)
    print("Starting Sign Language Learning App")
    print("=" * 60)
    
    try:
        import uvicorn
        from src.backend.api.routes import app
        
        print("\n Imports successful")
        print(f" Project root: {project_root}")
        print("\nStarting server on http://localhost:8000")
        print("Press Ctrl+C to stop\n")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"\n Import error: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
