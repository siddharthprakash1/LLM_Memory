#!/usr/bin/env python3
"""
Graph Timelapse Recorder.

Takes periodic screenshots of the Knowledge Graph from the enhanced
LongMemEval visualizer, then stitches them into a 10-second timelapse
video (MP4) and optionally a GIF.

Prerequisites:
  pip install playwright
  playwright install chromium
  brew install ffmpeg   (or apt-get install ffmpeg)

Usage:
  1. Start the visualizer:  ./venv/bin/python longmemeval_viz_enhanced.py
  2. Click "Start Benchmark" in the UI  (or curl -X POST http://127.0.0.1:5003/api/start_benchmark)
  3. Run this script:       ./venv/bin/python record_graph_timelapse.py

Options:
  --url          URL of the visualizer (default: http://127.0.0.1:5003)
  --duration     How many seconds to record (default: 300 = 5 min)
  --interval     Seconds between screenshots (default: 3)
  --output-dir   Where to save frames and final video (default: ./timelapse_output)
  --output-mp4   Final MP4 filename (default: graph_timelapse.mp4)
  --output-gif   Final GIF filename (default: graph_timelapse.gif)
  --target-secs  Target duration of the output video (default: 10)
  --no-gif       Skip GIF generation
  --graph-only   Capture only the graph area (clips the sidebars)
"""

import argparse
import os
import sys
import time
import shutil
import subprocess
import glob


def check_ffmpeg():
    """Check if ffmpeg is available."""
    if shutil.which("ffmpeg") is None:
        print("[ERROR] ffmpeg not found. Install it:")
        print("  macOS:  brew install ffmpeg")
        print("  Linux:  sudo apt-get install ffmpeg")
        sys.exit(1)


def capture_frames(url, duration, interval, output_dir, graph_only=False):
    """Use Playwright to screenshot the page at regular intervals."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[ERROR] playwright not installed. Run:")
        print("  pip install playwright && playwright install chromium")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Clean old frames
    for f in glob.glob(os.path.join(output_dir, "frame_*.png")):
        os.remove(f)

    print(f"\n[RECORDER] Starting capture")
    print(f"  URL:        {url}")
    print(f"  Duration:   {duration}s")
    print(f"  Interval:   {interval}s")
    print(f"  Graph only: {graph_only}")
    print(f"  Output:     {output_dir}/")
    print()

    frame_count = 0
    total_frames = duration // interval

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(url)
        # Wait for page to fully load
        page.wait_for_timeout(3000)

        start = time.time()
        while (time.time() - start) < duration:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.png")

            if graph_only:
                # Capture only the graph-wrapper element
                try:
                    el = page.query_selector("#graph-wrapper")
                    if el:
                        el.screenshot(path=frame_path)
                    else:
                        page.screenshot(path=frame_path)
                except Exception:
                    page.screenshot(path=frame_path)
            else:
                page.screenshot(path=frame_path)

            frame_count += 1
            elapsed = int(time.time() - start)
            pct = min(100, int(frame_count / max(total_frames, 1) * 100))
            print(f"  [{pct:3d}%] Frame {frame_count:4d}  |  {elapsed}s / {duration}s", end="\r")

            # Wait for next interval
            next_capture = start + frame_count * interval
            sleep_time = next_capture - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        browser.close()

    print(f"\n\n[RECORDER] Captured {frame_count} frames")
    return frame_count


def stitch_video(output_dir, output_mp4, target_secs, frame_count, interval):
    """Use ffmpeg to create an MP4 from the frames."""
    actual_duration = frame_count * interval
    # Calculate the FPS needed to compress into target_secs
    fps = max(1, frame_count / target_secs)

    input_pattern = os.path.join(output_dir, "frame_%05d.png")
    output_path = os.path.join(output_dir, output_mp4)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-vf", f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    print(f"\n[FFMPEG] Stitching {frame_count} frames -> {target_secs}s MP4 (fps={fps:.1f})")
    print(f"  {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed:\n{result.stderr}")
        return None

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[DONE] MP4 saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def create_gif(output_dir, output_gif, mp4_path, target_secs):
    """Convert MP4 to an optimized GIF."""
    output_path = os.path.join(output_dir, output_gif)
    palette_path = os.path.join(output_dir, "_palette.png")

    # Generate palette for better quality
    cmd_palette = [
        "ffmpeg", "-y",
        "-i", mp4_path,
        "-vf", f"fps=15,scale=960:-1:flags=lanczos,palettegen=stats_mode=diff",
        palette_path
    ]

    cmd_gif = [
        "ffmpeg", "-y",
        "-i", mp4_path,
        "-i", palette_path,
        "-lavfi", f"fps=15,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
        output_path
    ]

    print(f"\n[FFMPEG] Creating optimized GIF...")
    subprocess.run(cmd_palette, capture_output=True, text=True)
    result = subprocess.run(cmd_gif, capture_output=True, text=True)

    # Cleanup palette
    if os.path.exists(palette_path):
        os.remove(palette_path)

    if result.returncode != 0:
        print(f"[ERROR] GIF creation failed:\n{result.stderr}")
        return None

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[DONE] GIF saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Record Knowledge Graph timelapse")
    parser.add_argument("--url", default="http://127.0.0.1:5003", help="Visualizer URL")
    parser.add_argument("--duration", type=int, default=300, help="Recording duration in seconds")
    parser.add_argument("--interval", type=int, default=3, help="Seconds between screenshots")
    parser.add_argument("--output-dir", default="./timelapse_output", help="Output directory")
    parser.add_argument("--output-mp4", default="graph_timelapse.mp4", help="MP4 filename")
    parser.add_argument("--output-gif", default="graph_timelapse.gif", help="GIF filename")
    parser.add_argument("--target-secs", type=int, default=10, help="Target video duration in seconds")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--graph-only", action="store_true", help="Capture only the graph area")

    args = parser.parse_args()

    check_ffmpeg()

    print("=" * 55)
    print("  Knowledge Graph Timelapse Recorder")
    print("=" * 55)
    print(f"  Will record for {args.duration}s, then compress to {args.target_secs}s")
    print(f"  Make sure the visualizer is running at {args.url}")
    print(f"  and the benchmark has been started!")
    print("=" * 55)

    # 1. Capture frames
    frame_count = capture_frames(
        url=args.url,
        duration=args.duration,
        interval=args.interval,
        output_dir=args.output_dir,
        graph_only=args.graph_only,
    )

    if frame_count == 0:
        print("[ERROR] No frames captured. Is the server running?")
        sys.exit(1)

    # 2. Stitch into MP4
    mp4_path = stitch_video(
        output_dir=args.output_dir,
        output_mp4=args.output_mp4,
        target_secs=args.target_secs,
        frame_count=frame_count,
        interval=args.interval,
    )

    if not mp4_path:
        sys.exit(1)

    # 3. Create GIF (optional)
    if not args.no_gif:
        create_gif(
            output_dir=args.output_dir,
            output_gif=args.output_gif,
            mp4_path=mp4_path,
            target_secs=args.target_secs,
        )

    print("\n" + "=" * 55)
    print("  ALL DONE!")
    print(f"  MP4: {os.path.join(args.output_dir, args.output_mp4)}")
    if not args.no_gif:
        print(f"  GIF: {os.path.join(args.output_dir, args.output_gif)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
