**Agentic Object Detection on YouTube Video Frames by LandingAI**

This Streamlit application enables users to download YouTube videos, extract frames, run object detection using LandingAI’s Agentic API, and visualize results—all in an interactive web interface.

---

## 📝 Features

- **Download & Extract Frames**: Fetch a YouTube video via `yt-dlp`, extract evenly spaced frames at a user-defined count and resolution.  
- **Agentic Object Detection**: Send frames to LandingAI’s Agentic API, save raw JSON responses, and display progress with a countdown timer.  
- **Interactive Visualization**:  
  - View raw frames alongside JSON detection output.  
  - Overlay bounding boxes and labels on frames.  
  - Display per-frame object counts.  
- **Summary Metrics**:  
  - Total objects detected across all frames.  
  - Percentage of frames containing at least one object.  
  - Average objects per frame (all vs non-zero).  
  - Bar chart of object counts per frame (powered by Altair).  
  - Downloadable CSV of frame-by-frame object counts.  

## 🚀 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/agentic-yt-object-detection.git
   cd agentic-yt-object-detection

2. **Create a virtual environment** (recommended)  
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows

3. **Install dependencies**
  ```bash
  pip install streamlit opencv-python pillow requests altair pandas yt-dlp
 