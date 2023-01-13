import argparse
import os

import numpy as np

from lu_vp_detect import VPDetection

# Set up argument parser + options
parser = argparse.ArgumentParser(
    description="Main script for Lu's Vanishing Point Algorithm"
)
parser.add_argument("-i", "--image-path", help="Path to the input image", required=True)
parser.add_argument(
    "-lt",
    "--length-thresh",
    default=30,
    type=float,
    help="Minimum line length (in pixels) for detecting lines",
)
parser.add_argument(
    "-pp",
    "--principal-point",
    default=None,
    nargs=2,
    type=float,
    help="Principal point of the camera (default is image centre)",
)
parser.add_argument(
    "-f",
    "--focal-length",
    default=1500,
    type=float,
    help="Focal length of the camera (in pixels)",
)
parser.add_argument(
    "-at",
    "--angle-tol",
    default=np.pi / 3,
    type=float,
    help="Minimum angle tolerance (in radians) for detecting lines",
)
parser.add_argument(
    "-d", "--debug", action="store_true", help="Turn on debug image mode"
)
parser.add_argument(
    "-ds",
    "--debug-show",
    action="store_true",
    help="Show the debug image in an OpenCV window",
)
parser.add_argument(
    "-dp", "--debug-path", default=None, help="Path for writing the debug image"
)
parser.add_argument(
    "-s",
    "--seed",
    default=None,
    type=int,
    help="Specify random seed for reproducible results",
)
args = parser.parse_args()


def main():
    # Extract command line arguments
    input_path = args.image_path
    length_thresh = args.length_thresh
    principal_point = args.principal_point
    focal_length = args.focal_length
    angle_tol = args.angle_tol
    debug_mode = args.debug
    debug_show = args.debug_show
    debug_path = args.debug_path
    seed = args.seed

    print("Input path: {}".format(input_path))
    print("Seed: {}".format(seed))
    print("Line length threshold: {}".format(length_thresh))
    print("Focal length: {}".format(focal_length))
    print("Angle tolerance: {}".format(angle_tol))

    # Create object
    vpd = VPDetection(length_thresh, principal_point, focal_length, angle_tol, seed)

    if os.path.isdir(input_path):
        print("Input path is a directory")
        # Add only files to the list
        images = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f))
        ]
    if os.path.isfile(input_path):
        images = [input_path]

    for image in images:
        # Run VP detection algorithm
        vps = vpd.find_vps(image)
        print("Principal point: {}".format(vpd.principal_point))

        # Show VP information
        print("The vanishing points in 3D space are: ")
        for i, vp in enumerate(vps):
            print("Vanishing Point {:d}: {}".format(i + 1, vp))

        vp2D = vpd.vps_2D
        print("\nThe vanishing points in image coordinates are: ")
        for i, vp in enumerate(vp2D):
            print("Vanishing Point {:d}: {}".format(i + 1, vp))

        # Extra stuff
        debug_image = None
        if debug_path is not None:
            debug_image, ext = os.path.splitext(debug_path)
            if len(ext) == 0:
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)
                debug_image = os.path.join(debug_path, os.path.basename(image))
            else:
                debug_image = debug_path

        if debug_mode or debug_show:
            st = "Creating debug image"
            if debug_show:
                st += " and showing to the screen"
            if debug_path is not None:
                st += "\nAlso writing debug image to: {}".format(debug_image)

            if debug_show or debug_path is not None:
                print(st)
                vpd.create_debug_VP_image(debug_show, debug_image)


if __name__ == "__main__":
    main()
