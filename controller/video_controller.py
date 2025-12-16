import os
import cv2
import uuid
import time
import base64
import tempfile
import json
import numpy as np
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form, Depends, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Import services and models
from service.model_manager import get_model_manager
from infrastructure.redis import getRedisRepository
from inference.inference import Inference

video_router = APIRouter(tags=["video"])

# Mapping for common layer names (Chinese to English)
LAYER_NAME_MAPPING = {
    "肌肉层": "muscle layer",
    "肌肉": "muscle",
    "皮下脂肪": "subcutaneous fat",
    "脂肪": "fatty layer",
    "皮肤": "skin",
    "乳腺结节": "breast tumors",
    "甲状腺结节": "thyroid tumors",
    "腺体": "mammary gland",
    "真皮层": "dermis",
    "表皮层": "epidermis",
    "筋膜层": "fascia",
    "骨骼": "bone",
    "血管": "blood vessel",
    "神经": "nerve",
    "淋巴结": "lymph node",
    "甲状腺": "thyroid",
    "乳腺": "breast",
    "肝脏": "liver",
    "肾脏": "kidney",
    "脾脏": "spleen",
    "胰腺": "pancreas",
    "胆囊": "gallbladder",
    "膀胱": "bladder",
    "子宫": "uterus",
    "卵巢": "ovary",
    "前列腺": "prostate"
}

# Visualization Palette (Same as Inference)
PALETTE = [
    (0, 0, 0),       # 0 Background
    (255, 0, 0),     # 1 Red
    (0, 255, 0),     # 2 Green
    (0, 0, 255),     # 3 Blue
    (255, 255, 0),   # 4 Yellow
    (255, 0, 255),   # 5 Magenta
    (0, 255, 255),   # 6 Cyan
    (255, 128, 0),   # 7 Orange
    (128, 0, 255),   # 8 Purple
]

def draw_legend(image_bgr, labels):
    """
    Draw legend on the image.
    image_bgr: OpenCV image (BGR)
    labels: List of Chinese labels corresponding to indices 1, 2, ...
    """
    if not labels:
        return image_bgr
        
    # Convert to PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a font that supports Chinese
    font = None
    font_paths = [
        "msyh.ttc",     # Microsoft YaHei
        "simhei.ttf",   # SimHei
        "simsun.ttc",   # SimSun
        "arialuni.ttf", # Arial Unicode MS
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf"
    ]
    
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, 20)
            break
        except:
            continue
            
    if font is None:
        font = ImageFont.load_default()

    # Start position
    x = 10
    y = 10
    line_height = 30
    box_size = 20
    
    # Draw background for legend (optional, for better visibility)
    # max_width = 200
    # total_height = len(labels) * line_height + 10
    # draw.rectangle([x-5, y-5, x+max_width, y+total_height], fill=(0, 0, 0, 128))

    for i, label in enumerate(labels):
        if i + 1 >= len(PALETTE): break
        
        color = PALETTE[i+1] # RGB
        
        # Draw color box
        draw.rectangle([x, y + i * line_height, x + box_size, y + i * line_height + box_size], fill=tuple(color))
        
        # Draw text (Shadow for visibility)
        text_x = x + box_size + 10
        text_y = y + i * line_height - 2
        
        # Shadow
        draw.text((text_x + 1, text_y + 1), str(label), font=font, fill=(0, 0, 0))
        # Text
        draw.text((text_x, text_y), str(label), font=font, fill=(255, 255, 255))
        
    # Convert back to BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Redis keys
REDIS_KEY_VIDEO_LABELS = "video_segmentation:labels"

def cleanup_file(path: str):
    """清理临时文件"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Error cleaning up file {path}: {e}")

def get_labels_from_config():
    """从配置文件读取标签"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('video_segmentation', {}).get('labels', [])
    except Exception as e:
        print(f"Error reading config: {e}")
        return []

def get_video_labels():
    """获取视频分割标签，优先从Redis获取，没有则从配置读取并缓存"""
    redis_repo = getRedisRepository()
    try:
        # Try to get from Redis
        cached_labels = redis_repo.get(REDIS_KEY_VIDEO_LABELS)
        if cached_labels:
            # RedisRepository automatically deserializes JSON if configured, 
            # but let's check if it returns string or object.
            # Based on RedisRepository implementation, it uses json.loads if possible.
            if isinstance(cached_labels, str):
                 return json.loads(cached_labels)
            return cached_labels
        
        # If not in Redis, load from config
        labels = get_labels_from_config()
        
        # Cache in Redis (expire in 24 hours)
        if labels:
            # RedisRepository.set handles serialization
            redis_repo.set(REDIS_KEY_VIDEO_LABELS, labels, ex=86400)
        
        return labels
    except Exception as e:
        print(f"Error getting labels: {e}")
        return get_labels_from_config() # Fallback

def refresh_video_labels_cache():
    """强制刷新视频标签缓存（从配置读取并写入Redis）"""
    redis_repo = getRedisRepository()
    try:
        labels = get_labels_from_config()
        if labels:
            redis_repo.set(REDIS_KEY_VIDEO_LABELS, labels, ex=86400)
            print(f"已刷新视频标签缓存: {len(labels)} 个标签")
    except Exception as e:
        print(f"刷新视频标签缓存失败: {e}")

@video_router.get("/video/labels")
async def get_labels():
    """获取可用的视频分割标签"""
    labels = get_video_labels()
    return {"success": True, "data": labels}

def crop_ultrasound_roi(frame):
    """
    自动裁剪超声图像的 ROI 区域
    去除周围的黑色边框或 UI 元素
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find non-black regions
    # Use a low threshold to catch the ultrasound cone/rect
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return frame, (0, 0, frame.shape[1], frame.shape[0])
    
    # Find the largest contour which should be the ultrasound image
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Add a small padding if possible, but stay within bounds
    padding = 0
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2*padding)
    h = min(frame.shape[0] - y, h + 2*padding)
    
    # Crop
    roi = frame[y:y+h, x:x+w]
    return roi, (x, y, w, h)

# In-memory store for video file paths (for demo purposes)
# In production, use Redis or database with expiration
VIDEO_STORE = {}

@video_router.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """上传视频文件，返回 file_id 用于流式处理"""
    try:
        filename = file.filename or "video.mp4"
        ext = os.path.splitext(filename)[1].lower()
        
        # Save to D drive
        upload_dir = "D:\\uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            
        file_id = str(uuid.uuid4())
        # Use file_id in filename to prevent conflicts
        save_path = os.path.join(upload_dir, f"{file_id}{ext}")
        
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
            
        VIDEO_STORE[file_id] = save_path
        print(f"Video saved to: {save_path}")
        
        return {"success": True, "file_id": file_id}
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@video_router.websocket("/ws/video/{file_id}")
async def video_websocket(websocket: WebSocket, file_id: str):
    await websocket.accept()
    
    if file_id not in VIDEO_STORE:
        await websocket.close(code=4004, reason="Video file not found")
        return
        
    file_path = VIDEO_STORE[file_id]
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        await websocket.close(code=4004, reason="Cannot open video file")
        return

    try:
        # 1. Send Metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps <= 0: fps = 25.0
        
        await websocket.send_json({
            "type": "metadata",
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height
        })
        
        print(f"WS[{file_id}]: Metadata sent.")
        
        # Model will be lazy loaded when needed
        inference_model = None
        cached_text_feature = None
        current_label = ""
        original_label = None # Store Chinese labels for legend
        
        # Inner block for loop
        if True:
            while True:
                # 2. Receive Command
                print(f"WS[{file_id}]: Waiting for command...")
                data = await websocket.receive_json()
                action = data.get("action")
                print(f"WS[{file_id}]: Received action: {action}, data: {data}")
                
                if action == "seek":
                    index = int(data.get("index", 0))
                    mode = data.get("mode", "manual")
                    label = data.get("label", "")
                    
                    # Ensure index is within bounds
                    index = max(0, min(index, total_frames - 1))
                    
                    # Seek to frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Prepare Original Image (Base64)
                        _, buffer = cv2.imencode('.jpg', frame)
                        original_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        result_b64 = None
                        
                        # Process Segmentation if not in preview mode
                        if mode != 'preview':
                            try:
                                # Lazy load model
                                if inference_model is None:
                                    print(f"WS[{file_id}]: Loading model for the first time...")
                                    try:
                                        model_manager = get_model_manager()
                                        inference_model = model_manager.get_inference_model()
                                        print(f"WS[{file_id}]: Model loaded.")
                                    except Exception as e:
                                        print(f"WS[{file_id}]: Model load failed: {e}")
                                        await websocket.send_json({
                                            "type": "error",
                                            "message": f"Model load failed: {str(e)}"
                                        })
                                        continue
                                
                                roi, (rx, ry, rw, rh) = crop_ultrasound_roi(frame)
                                if roi.size > 0:
                                    # RAG Logic
                                    if mode == 'rag':
                                        try:
                                            faiss_service = get_model_manager().get_faiss_service()
                                            roi_rgb_search = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                                            rag_result = faiss_service.get_best_match_label(roi_rgb_search, mapping=LAYER_NAME_MAPPING)
                                            if rag_result:
                                                rag_label, original_label = rag_result
                                                print(f"RAG Recall Label: {original_label}")
                                                description = rag_label
                                        except Exception as e:
                                            print(f"RAG search error: {e}")

                                    # Update cached text feature if label changed
                                    # description is initialized above loop or defaulted?
                                    # We need to ensure description is set correctly for each frame if RAG mode
                                    if mode == 'manual':
                                         description = label
                                    elif mode == 'rag' and 'description' not in locals():
                                         description = "ultrasound image" # Default if RAG fails first time

                                    if description != current_label:
                                        try:
                                            cached_text_feature = inference_model.encode_text(description)
                                            current_label = description
                                        except Exception as e:
                                            print(f"Error encoding text: {e}")
                                            cached_text_feature = None

                                    # Pass numpy array directly (convert BGR to RGB)
                                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                                    
                                    start_infer = time.time()
                                    result_overlay = inference_model.interface(
                                        roi_rgb, 
                                        description, 
                                        visualize=True, 
                                        text_feature=cached_text_feature
                                    )
                                    # print(f"Inference time: {time.time() - start_infer:.4f}s")
                                    
                                    # Resize and embed
                                    result_overlay_resized = cv2.resize(result_overlay, (rw, rh))
                                    result_overlay_bgr = cv2.cvtColor(result_overlay_resized, cv2.COLOR_RGB2BGR)
                                    frame[ry:ry+rh, rx:rx+rw] = result_overlay_bgr

                                    # Draw Legend if RAG mode
                                    if mode == 'rag' and original_label:
                                        frame = draw_legend(frame, original_label)
                                else:
                                    print(f"Empty ROI for frame {index}")
                                
                                # Encode Result Image
                                _, res_buffer = cv2.imencode('.jpg', frame)
                                result_b64 = base64.b64encode(res_buffer).decode('utf-8')
                                
                            except Exception as e:
                                print(f"WS Frame processing error: {e}")
                                # If error, return original as result or None
                                result_b64 = original_b64
                        else:
                            result_b64 = original_b64 # Preview mode: result is same as original or empty
                        
                        await websocket.send_json({
                            "type": "frame",
                            "index": index,
                            "original": original_b64,
                            "result": result_b64
                        })
                    else:
                        # End of stream or error
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to read frame"
                        })

    except WebSocketDisconnect:
        print(f"Client disconnected from video {file_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        cap.release()
        # 释放模型以回收显存
        if inference_model:
            try:
                get_model_manager().release_inference_model()
            except Exception as e:
                print(f"Error releasing model in websocket: {e}")

@video_router.get("/video/stream/{file_id}")
async def stream_video(
    file_id: str,
    mode: str = "manual", # 'manual', 'rag', 'preview'
    label: Optional[List[str]] = Query(None)
):
    """
    流式返回视频分割结果 (MJPEG stream)
    mode='preview' 时仅返回原视频流（用于解决格式兼容性问题）
    降低采样频率：1秒采样5帧
    """
    if file_id not in VIDEO_STORE:
        raise HTTPException(status_code=404, detail="Video file not found")
        
    file_path = VIDEO_STORE[file_id]
    
    # Check label for manual mode
    if mode == 'manual' and not label:
        # If no label provided, we can default or error. 
        pass
        
    # Prepare text feature for manual mode to avoid re-encoding every frame
    text_feature = None
    label_str = "ultrasound image"
    
    if mode == 'manual' and label:
        # Join labels with space if it's a list
        if isinstance(label, list):
            label_str = " ".join(label)
        else:
            label_str = str(label)

    def generate_frames():
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            yield b''
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25.0
        
        # Configure frame skipping and delay based on mode
        # mode='preview': Original video, no skipping (step=1), real-time speed (sleep)
        # mode!='preview': Segmentation, skip frames (5 FPS sampling), fast processing
        
        if mode == 'preview':
             step = 1
             frame_delay = 1.0 / fps
        else:
             target_fps = 20
             step = max(1, int(fps / target_fps))
             frame_delay = 0 # No artificial delay for segmentation
        
        # Initialize model only if not in preview mode
        model_manager = None
        inference_model = None
        
        # Pre-compute text feature if in manual mode
        cached_text_feature = None
        cached_label = label_str
        if mode != 'preview':
            model_manager = get_model_manager()
            inference_model = model_manager.get_inference_model()
            
            if mode == 'manual':
                # Encode once
                try:
                    cached_text_feature = inference_model.encode_text(label_str)
                except Exception as e:
                    print(f"Error encoding text: {e}")
        
        frame_count = 0
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames logic
                if frame_count % step != 0:
                    frame_count += 1
                    continue
                
                frame_count += 1
                
                # Process frame (only if not preview)
                if mode != 'preview':
                    try:
                        roi, (rx, ry, rw, rh) = crop_ultrasound_roi(frame)
                        if roi.size > 0:
                            target_label = label_str
                            
                            # RAG Logic
                            if mode == 'rag':
                                try:
                                    faiss_service = get_model_manager().get_faiss_service()
                                    roi_rgb_search = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                                    rag_result = faiss_service.get_best_match_label(roi_rgb_search, mapping=LAYER_NAME_MAPPING)
                                    if rag_result:
                                        rag_label, original_label = rag_result
                                        print(f"RAG Recall Label: {original_label}")
                                        target_label = rag_label
                                    else:
                                        # Reset original_label if no match to avoid showing old legend
                                        original_label = None 
                                except Exception as e:
                                    print(f"RAG search error: {e}")
                                    original_label = None
                            
                            # Update cached feature if label changed
                            if target_label != cached_label or cached_text_feature is None:
                                try:
                                    cached_text_feature = inference_model.encode_text(target_label)
                                    cached_label = target_label
                                except Exception as e:
                                    print(f"Error encoding text: {e}")
                            
                            # Pass numpy array directly (convert BGR to RGB)
                            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            
                            print(f"Target Label: {target_label}")
                            result_overlay = inference_model.interface(
                                roi_rgb, 
                                target_label, 
                                visualize=True,
                                text_feature=cached_text_feature
                            )
                            
                            result_overlay_resized = cv2.resize(result_overlay, (rw, rh))
                            result_overlay_bgr = cv2.cvtColor(result_overlay_resized, cv2.COLOR_RGB2BGR)
                            frame[ry:ry+rh, rx:rx+rw] = result_overlay_bgr
                            
                            # Draw Legend if RAG mode
                            if mode == 'rag' and original_label:
                                frame = draw_legend(frame, original_label)
                            
                    except Exception as e:
                        print(f"Frame processing error: {e}")
                
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Control playback speed for preview mode
                if mode == 'preview':
                    elapsed = time.time() - start_time
                    wait_time = frame_delay - elapsed
                    if wait_time > 0:
                        time.sleep(wait_time)
                       
        finally:
            cap.release()
            # 释放模型以回收显存
            if mode != 'preview' and inference_model:
                 try:
                     get_model_manager().release_inference_model()
                 except Exception as e:
                     print(f"Error releasing model: {e}")
            
            # Don't delete file_path here as user might re-stream
            # Cleanup should be handled by a background cleaner or expiry
            
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@video_router.post("/video/segment")
async def segment_video(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    mode: str = Form(...), # 'manual' or 'rag'
    label: Optional[List[str]] = Form(None)
):
    """
    接收视频文件，进行 ROI 裁剪、模型分割，并将结果嵌入回视频
    """
    if mode == 'manual' and not label:
         raise HTTPException(status_code=400, detail="Label is required for manual mode")
         
    # Prepare label string
    label_str = "ultrasound image"
    if mode == 'manual' and label:
        if isinstance(label, list):
            label_str = " ".join(label)
        else:
            label_str = str(label)

    # 检查文件类型
    filename = file.filename or "video.mp4"
    ext = os.path.splitext(filename)[1].lower()
    
    # 创建临时文件
    temp_input_fd, temp_input_path = tempfile.mkstemp(suffix=ext)
    os.close(temp_input_fd)
    
    # Output file (always MP4 for browser compatibility)
    temp_output_path = temp_input_path.rsplit('.', 1)[0] + '_segmented.mp4'
    
    inference_model = None
    try:
        # Save uploaded file
        content = await file.read()
        with open(temp_input_path, "wb") as f:
            f.write(content)
            
        # Get Inference Model
        model_manager = get_model_manager()
        inference_model = model_manager.get_inference_model()
        
        # Pre-compute text feature
        cached_text_feature = None
        cached_label = label_str
        original_label = None
        if mode == 'manual':
            try:
                cached_text_feature = inference_model.encode_text(label_str)
            except Exception as e:
                print(f"Error encoding text: {e}")

        # Open video
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25.0
        
        # Prepare Video Writer
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        except:
            out = None
            
        if not out or not out.isOpened():
            print("Falling back to mp4v encoder")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            
        if not out.isOpened():
             raise HTTPException(status_code=500, detail="Could not initialize video writer")

        # Process frame by frame
        # Optimization: Process every Nth frame if needed, but for smooth video we need every frame.
        # However, inference might be slow. For demo, we process all frames.
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 1. Crop ROI (Remove useless borders)
            roi, (rx, ry, rw, rh) = crop_ultrasound_roi(frame)
            
            if roi.size == 0:
                out.write(frame)
                continue

            target_label = label_str
            
            # RAG Logic
            if mode == 'rag':
                try:
                    faiss_service = get_model_manager().get_faiss_service()
                    roi_rgb_search = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    rag_result = faiss_service.get_best_match_label(roi_rgb_search, mapping=LAYER_NAME_MAPPING)
                    if rag_result:
                        rag_label, original_label = rag_result
                        print(f"RAG Recall Label: {original_label}")
                        target_label = rag_label
                    else:
                        original_label = None
                except Exception as e:
                    print(f"RAG search error: {e}")
                    original_label = None
            
            # Update cached feature if label changed
            if target_label != cached_label or cached_text_feature is None:
                 try:
                     cached_text_feature = inference_model.encode_text(target_label)
                     cached_label = target_label
                 except Exception as e:
                     print(f"Error encoding text: {e}")

            # 2. Inference on ROI
            try:
                # Pass numpy array directly (convert BGR to RGB)
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                
                # Run inference
                result_overlay = inference_model.interface(
                    roi_rgb, 
                    target_label, 
                    visualize=True,
                    text_feature=cached_text_feature
                )
                
                # result_overlay is a numpy array (RGB) of size 256x256 (from `conv_to_mask`)
                # We need to resize it back to ROI size (rw, rh)
                result_overlay_resized = cv2.resize(result_overlay, (rw, rh))
                
                # Convert back to BGR for OpenCV
                result_overlay_bgr = cv2.cvtColor(result_overlay_resized, cv2.COLOR_RGB2BGR)
                
                # 3. Embed back to original frame
                # Replace the ROI area with the result
                frame[ry:ry+rh, rx:rx+rw] = result_overlay_bgr
                
                # Draw Legend if RAG mode
                if mode == 'rag' and original_label:
                    frame = draw_legend(frame, original_label)
                
            except Exception as e:
                print(f"Frame {frame_count} processing failed: {e}")
                # If inference fails, just show original frame
                pass
                
            out.write(frame)
            
        cap.release()
        out.release()
        
        # 释放模型以回收显存
        if inference_model:
            try:
                get_model_manager().release_inference_model()
            except Exception as e:
                print(f"Error releasing model: {e}")

        # Cleanup intermediate ROI file
        if os.path.exists(temp_input_path + "_roi.png"):
            os.remove(temp_input_path + "_roi.png")

        # Register cleanup tasks
        background_tasks.add_task(cleanup_file, temp_input_path)
        background_tasks.add_task(cleanup_file, temp_output_path)
        
        return FileResponse(
            temp_output_path, 
            media_type="video/mp4", 
            filename=f"segmented_{os.path.splitext(filename)[0]}.mp4"
        )

    except Exception as e:
        # 释放模型以回收显存
        if inference_model:
            try:
                get_model_manager().release_inference_model()
            except:
                pass
            
        cleanup_file(temp_input_path)
        cleanup_file(temp_output_path)
        print(f"Video segmentation error: {e}")
        raise HTTPException(status_code=500, detail=f"Video segmentation failed: {str(e)}")

@video_router.post("/video/convert")
async def convert_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    接收视频文件，将其转换为 MP4 格式并返回
    主要用于解决浏览器不支持 AVI 播放的问题
    """
    # 检查文件类型
    filename = file.filename or "video.avi"
    ext = os.path.splitext(filename)[1].lower()
    
    # 创建临时文件路径
    temp_input_fd, temp_input_path = tempfile.mkstemp(suffix=ext)
    os.close(temp_input_fd)
    
    temp_output_path = temp_input_path.rsplit('.', 1)[0] + '_converted.mp4'
    
    try:
        # 保存上传的文件
        content = await file.read()
        with open(temp_input_path, "wb") as f:
            f.write(content)
            
        # 使用 OpenCV 进行转换
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
            
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25.0
        
        # 尝试使用 avc1 (H.264) 编码，这是浏览器支持最好的格式
        # 注意：这需要系统安装了相应的编码器库 (如 openh264 或 ffmpeg)
        # 如果 avc1 失败，OpenCV 通常不会抛出异常而是生成无法播放的文件或大小为0的文件
        # 这里我们优先尝试 avc1
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        except:
            out = None

        if not out or not out.isOpened():
            # 如果 avc1 失败，回退到 mp4v (部分浏览器可能不支持，但兼容性较好)
            print("Falling back to mp4v encoder")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            
        if not out.isOpened():
             raise HTTPException(status_code=500, detail="Could not initialize video writer")

        # 逐帧转换
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            
        cap.release()
        out.release()
        
        # 注册清理任务
        background_tasks.add_task(cleanup_file, temp_input_path)
        background_tasks.add_task(cleanup_file, temp_output_path)
        
        # 返回转换后的文件
        return FileResponse(
            temp_output_path, 
            media_type="video/mp4", 
            filename=f"converted_{os.path.splitext(filename)[0]}.mp4"
        )

    except Exception as e:
        # 出错时立即清理
        cleanup_file(temp_input_path)
        cleanup_file(temp_output_path)
        print(f"Video conversion error: {e}")
        raise HTTPException(status_code=500, detail=f"Video conversion failed: {str(e)}")
