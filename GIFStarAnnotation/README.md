GIF Animation Star Annotation
===========================

This tool imports a gif file, finds a star based on a 2MASS ID and applies a small circle to the gif file. The purpose of this is simply to highlight a reigon of interest within a gif file.

The script leverages astrometry.net to perform the plate solving, so an API key from there is required. API keys for astrometry.net is only available when you sign in. They have various OAuth login options so it's straightforeward. 

How to use this tool
--------------------------
This is a simple python script and doesn't require compilation.

Script arguments by position:
1. Input file path
2. Output file path
3. Star designation (2MASS format)
4. Radius of circle annotation
5. Thickness of circle annotation
6. API Key

Example:
`python3 StarAnnotation.py input.gif out.gif J03293430+3117433 --radius 30 --thickness 3 --api-key xxxxx`

