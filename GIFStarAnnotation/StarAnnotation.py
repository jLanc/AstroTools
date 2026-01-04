#!/usr/bin/env python3
"""
Annotates a GIF animation with a circle around a specified star using 2MASS coordinates.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.astrometry_net import AstrometryNet
import tempfile
import warnings

warnings.filterwarnings('ignore')


def parse_2mass_id(twomass_id):
    """Parse 2MASS identifier to extract RA and Dec.
    
    Format: J[RA][Dec] where RA is HHMMSS.s and Dec is ±DDMMSS
    Example: J03293430+3117433 -> RA=03:29:34.30, Dec=+31:17:43.3
    """
    if not twomass_id.startswith('J'):
        raise ValueError("2MASS ID must start with 'J'")
    
    coords = twomass_id[1:]
    
    # RA: first 8 characters (HHMMSS.s or HHMMSSs)
    ra_str = coords[:8]
    hh = int(ra_str[:2])
    mm = int(ra_str[2:4])
    ss = float(ra_str[4:6] + '.' + ra_str[6:8])
    
    # Dec: remaining characters
    dec_str = coords[8:]
    sign = 1 if dec_str[0] == '+' else -1
    dec_str = dec_str[1:]
    dd = int(dec_str[:2])
    dm = int(dec_str[2:4])
    ds = float(dec_str[4:6] + '.' + dec_str[6:]) if len(dec_str) > 6 else float(dec_str[4:6])
    
    ra_deg = (hh + mm/60 + ss/3600) * 15  # Convert hours to degrees
    dec_deg = sign * (dd + dm/60 + ds/3600)
    
    return ra_deg, dec_deg


def extract_first_frame(gif_path):
    """Extract first frame from GIF as numpy array."""
    img = Image.open(gif_path)
    img.seek(0)
    return np.array(img.convert('L'))


def plate_solve_image(image_array, api_key=None):
    """Plate solve image using astrometry.net.
    
    Returns WCS object or None if solving fails.
    """
    print("Starting plate solving...")
    
    ast = AstrometryNet()
    if api_key:
        ast.api_key = api_key
    
    # Create temporary FITS file
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        hdu = fits.PrimaryHDU(image_array)
        hdu.writeto(tmp.name, overwrite=True)
        tmp_path = tmp.name
    
    try:
        # Submit to astrometry.net
        wcs_header = ast.solve_from_image(tmp_path, 
                                          force_image_upload=True,
                                          solve_timeout=300)
        
        if wcs_header:
            print("Plate solving successful!")
            return WCS(wcs_header)
        else:
            print("Plate solving failed - could not determine WCS")
            return None
            
    except Exception as e:
        print(f"Error during plate solving: {e}")
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def sky_to_pixel(wcs, ra, dec):
    """Convert sky coordinates to pixel coordinates."""
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    x, y = wcs.world_to_pixel(coord)
    return float(x), float(y)


def draw_circle_on_frame(frame, x, y, radius, thickness, color=None):
    """Draw circle on PIL Image frame."""
    img = frame.convert('RGB') if frame.mode != 'RGB' else frame.copy()
    draw = ImageDraw.Draw(img)
    
    # Use white or red depending on image
    if color is None:
        color = (255, 0, 0)  # Red
    
    # Draw circle with thickness
    for i in range(thickness):
        r = radius + i - thickness // 2
        draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=1)
    
    return img


def process_gif(input_path, output_path, twomass_id, radius=20, thickness=2, 
                api_key=None):
    """Main processing function."""
    print(f"Processing GIF: {input_path}")
    print(f"Target star: {twomass_id}")
    
    # Parse 2MASS coordinates
    try:
        ra, dec = parse_2mass_id(twomass_id)
        print(f"Parsed coordinates: RA={ra:.6f}°, Dec={dec:.6f}°")
    except Exception as e:
        print(f"Error parsing 2MASS ID: {e}")
        return False
    
    # Extract first frame and plate solve
    first_frame = extract_first_frame(input_path)
    wcs = plate_solve_image(first_frame, api_key)
    
    if wcs is None:
        print("Failed to plate solve image. Cannot proceed.")
        return False
    
    # Convert sky coordinates to pixel coordinates
    try:
        x, y = sky_to_pixel(wcs, ra, dec)
        print(f"Star location in image: x={x:.1f}, y={y:.1f}")
    except Exception as e:
        print(f"Error converting coordinates: {e}")
        return False
    
    # Process all frames
    print("Drawing circles on all frames...")
    img = Image.open(input_path)
    frames = []
    durations = []
    
    try:
        while True:
            # Get frame duration
            duration = img.info.get('duration', 100)
            durations.append(duration)
            
            # Draw circle on frame
            frame_with_circle = draw_circle_on_frame(img, x, y, radius, thickness)
            frames.append(frame_with_circle)
            
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    
    print(f"Processed {len(frames)} frames")
    
    # Save output GIF
    print(f"Saving to: {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=img.info.get('loop', 0)
    )
    
    print("Done!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Mark a star in a GIF using plate solving and 2MASS coordinates'
    )
    parser.add_argument('input', help='Input GIF file')
    parser.add_argument('output', help='Output GIF file')
    parser.add_argument('star', help='2MASS star ID (e.g., J03293430+3117433)')
    parser.add_argument('--radius', type=int, default=20, 
                       help='Circle radius in pixels (default: 20)')
    parser.add_argument('--thickness', type=int, default=2,
                       help='Circle line thickness in pixels (default: 2)')
    parser.add_argument('--api-key', help='Astrometry.net API key (optional)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Process the GIF
    success = process_gif(
        args.input,
        args.output,
        args.star,
        radius=args.radius,
        thickness=args.thickness,
        api_key=args.api_key
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()