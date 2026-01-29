# Letter GIFs

This directory should contain tutorial GIF files for each sign language letter.

## File Naming Convention
- `A.gif` - Letter A
- `B.gif` - Letter B
- ... and so on

## Requirements
- GIF should clearly show the hand gesture
- Include both start and end positions for dynamic letters (J, Z)
- Recommended size: 400x400 pixels or similar
- Keep file size reasonable (< 2MB per GIF)

## Dynamic Letters (J, Z)
For J and Z, the GIF should show the complete movement sequence:
- **J**: Hook motion downward
- **Z**: Z-shape in the air

## Static Letters
For all other letters (A-Y excluding J, Z), show a clear static pose of the hand gesture.

## Sources
You can create these GIFs from:
1. Video recordings of proper ASL signs
2. Existing ASL reference materials (ensure licensing)
3. Custom recordings demonstrating each letter

## Usage
The `TutorialManager` (to be implemented) will load these GIFs and display them to users:
- Initially when they start practicing a new letter
- After multiple failed attempts (hint system)
- On-demand when user requests help
