from pathlib import Path
import shutil
import subprocess


INPUT_VIDEO = Path(
    r"C:/ros2/ros_ws/src/varjo_ros2_example/RealData/User99/Video/Task_1_sbs_20260608145052.mp4"
)
OUTPUT_VIDEO = INPUT_VIDEO.with_name(
    f"{INPUT_VIDEO.stem}_left_00m00s_to_00m15s_640x480.mp4"
)

START_TIME = "00:00:00"
DURATION_SECONDS = 15


def create_left_eye_clip() -> None:
    """Extract by start time and duration, crop the left SBS frame, and resize to 640x480."""
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Input video was not found: {INPUT_VIDEO}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and make sure 'ffmpeg' is on PATH."
        )

    command = [
        ffmpeg,
        "-y",
        "-ss",
        START_TIME,
        "-i",
        str(INPUT_VIDEO),
        "-t",
        str(DURATION_SECONDS),
        "-map",
        "0:v:0",
        "-vf",
        "crop=1280:1024:0:0,scale=640:480,setsar=1",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUTPUT_VIDEO),
    ]

    subprocess.run(command, check=True)
    print(f"Created: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    create_left_eye_clip()
