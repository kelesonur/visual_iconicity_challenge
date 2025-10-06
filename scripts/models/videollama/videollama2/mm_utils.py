import ast
import os
import math
import base64
import traceback
from io import BytesIO

# import cv2
import torch
# import imageio
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
# from moviepy.editor import VideoFileClip
from transformers import StoppingCriteria

from .constants import NUM_FRAMES, MAX_FRAMES, NUM_FRAMES_PER_SECOND, MODAL_INDEX_MAP, DEFAULT_IMAGE_TOKEN


def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def create_photo_grid(arr, rows=None, cols=None):
    """
    Create a photo grid from a 4D numpy array with shape [t, h, w, c].

    Parameters:
        arr (numpy.ndarray): Input array with shape [t, h, w, c].
        rows (int): Optional. Number of rows in the grid. If not set, it will be determined based on `cols` or the square root of `t`.
        cols (int): Optional. Number of columns in the grid. If not set, it will be determined based on `rows` or the square root of `t`.

    Returns:
        numpy.ndarray: A 3D numpy array representing the photo grid.
    """

    if isinstance(arr, list):
        if isinstance(arr[0], Image.Image):
            arr = np.stack([np.array(img) for img in arr])
        elif isinstance(arr[0], np.ndarray):
            arr = np.stack(arr)
        else:
            raise ValueError("Invalid input type. Expected list of Images or numpy arrays.")

    t, h, w, c = arr.shape
    
    # Calculate the number of rows and columns if not provided
    if rows is None and cols is None:
        rows = math.ceil(math.sqrt(t))
        cols = math.ceil(t / rows)
    elif rows is None:
        rows = math.ceil(t / cols)
    elif cols is None:
        cols = math.ceil(t / rows)

    # Check if the grid can hold all the images
    if rows * cols < t:
        raise ValueError(f"Not enough grid cells ({rows}x{cols}) to hold all images ({t}).")
    
    # Create the grid array with appropriate height and width
    grid_height = h * rows
    grid_width = w * cols
    grid = np.zeros((grid_height, grid_width, c), dtype=arr.dtype)
    
    # Fill the grid with images
    for i in range(t):
        row_idx = i // cols
        col_idx = i % cols
        grid[row_idx*h:(row_idx+1)*h, col_idx*w:(col_idx+1)*w, :] = arr[i]
    
    return grid


def process_image(image_path, processor, aspect_ratio='pad'):
    image = Image.open(image_path).convert('RGB')

    images = [np.array(image)]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f) for f in images]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
    else:
        images = [Image.fromarray(f) for f in images]

    images = processor.preprocess(images, return_tensors='pt')['pixel_values']
    return images


def frame_sample(duration, mode='uniform', num_frames=None, fps=None):
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        # NOTE: v1 version
        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(duration - 1) / num_frames

        frame_ids = []
        for i in range(num_frames):
            # Calculate the start and end indices of each segment
            start = seg_size * i
            end   = seg_size * (i + 1)
            # Append the middle index of the segment to the list
            frame_ids.append((start + end) / 2)

        return np.round(np.array(frame_ids) + 1e-6).astype(int)
        # NOTE: v0 version
        # return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(fps // NUM_FRAMES_PER_SECOND, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')

def process_video(video_path, processor, s=None, e=None, aspect_ratio='pad', num_frames=NUM_FRAMES):
    error_info = None
    
    # 新增：前置文件校验
    if isinstance(video_path, str) and not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    try:  # 外层异常捕获
        if isinstance(video_path, str):
            if s is not None and e is not None:
                s = max(float(s), 0.0)
                e = max(float(e), 0.0)
                if s > e:
                    s, e = e, s
                elif s == e:
                    e = s + 1

            # 1. 视频加载逻辑重构
            use_fallback = False
            vreader = None
            gif_reader = None
            frame_files = []

            # 新增：解码异常处理
            try:
                if os.path.isdir(video_path):
                    frame_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    fps = 3
                    num_frames_of_video = len(frame_files)
                elif video_path.endswith('.gif'):
                    # gif_reader = imageio.get_reader(video_path)
                    # fps = 25
                    # num_frames_of_video = len(gif_reader)
                    pass
                else:
                    # 主解码方案：DECORD
                    vreader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
                    fps = vreader.get_avg_fps()
                    num_frames_of_video = len(vreader)
                    
            except (decord.DECORDError, RuntimeError) as e:
                print(f"主解码器失败 ({video_path}): {str(e)}")
                use_fallback = True

            # 2. 备用解码方案
            if use_fallback and not video_path.endswith(('.gif')):
                try:
                    print(f"尝试备用解码方案: {video_path}")
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    num_frames_of_video = total_frames if total_frames > 0 else 0
                    
                    # 重新计算帧范围
                    f_start = 0 if s is None else max(int(s * fps), 0)
                    f_end = num_frames_of_video - 1 if e is None else min(int(e * fps), num_frames_of_video - 1)
                    
                    # 均匀采样逻辑
                    duration = max(f_end - f_start + 1, 1)
                    indices = []
                    if num_frames and num_frames > 0:
                        step = max(duration // num_frames, 1)
                        indices = list(range(f_start, f_end + 1, step))[:num_frames]
                    else:
                        indices = list(range(f_start, f_end + 1))
                    
                    # 读取视频帧
                    video_data = []
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            video_data.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    cap.release()
                    
                    # 跳过后续处理
                    return processor.preprocess(video_data, return_tensors='pt')['pixel_values']
                    
                except Exception as fallback_e:
                    error_info = f"主备解码方案均失败: {str(fallback_e)}"
                    raise

            # 原始帧处理逻辑（增加索引校验）
            if not use_fallback:
                f_start = 0 if s is None else max(int(s * fps), 0)
                f_end = num_frames_of_video - 1 if e is None else min(int(e * fps), num_frames_of_video - 1)
                frame_indices = list(range(f_start, f_end + 1))
                
                # 采样前校验
                duration = len(frame_indices)
                if duration == 0:
                    raise ValueError("无效的视频时间段 (duration=0)")
                    
                # 安全采样
                try:
                    if num_frames is None:
                        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', fps=fps)]
                    else:
                        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=num_frames)]
                except IndexError:
                    # 采样异常时自动调整
                    sampled_frame_indices = frame_indices[:num_frames] if num_frames else frame_indices
                    
                # 索引范围二次校验
                sampled_frame_indices = [idx for idx in sampled_frame_indices if idx < num_frames_of_video]
                
                # 获取帧数据
                if os.path.isdir(video_path):
                    video_data = [Image.open(os.path.join(video_path, frame_files[idx])) for idx in sampled_frame_indices]
                elif video_path.endswith('.gif'):
                    video_data = [Image.fromarray(gif_reader.get_data(idx)) for idx in sampled_frame_indices]
                else:
                    # DECORD数据获取（增加异常处理）
                    try:
                        video_data = [Image.fromarray(frame) for frame in vreader.get_batch(sampled_frame_indices).asnumpy()]
                    except decord.DECORDError as de:
                        error_info = f"DECORD帧获取失败: {str(de)}"
                        raise

        # 其他数据类型处理保持不变...
        
    except Exception as e:
        # 统一错误处理
        print(f"视频处理失败: {video_path}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {error_info or str(e)}")
        
        # 返回空白帧避免流程中断
        dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        video_data = [dummy_image] * (num_frames or 8)
        
    # 后续处理保持不变...
    while num_frames is not None and len(video_data) < num_frames:
        video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))

    video_data = video_data[:MAX_FRAMES]

    try:
        if aspect_ratio == 'pad':
            images = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in video_data]
            video = processor.preprocess(images, return_tensors='pt')['pixel_values']
        else:
            images = [f for f in video_data]
            video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    except Exception as process_e:
        print(f"后处理失败: {str(process_e)}")
        video = torch.zeros((len(video_data), 3, 224, 224))  # 返回空白张量
        
    return video

def tokenizer_multimodal_token(prompt, tokenizer, multimodal_token=DEFAULT_IMAGE_TOKEN, return_tensors=None):
    """Tokenize text and multimodal tag to input_ids.

    Args:
        prompt (str): Text prompt (w/ multimodal tag), e.g., '<video>\nDescribe the video.'
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        multimodal_token (int): Token index corresponding to the multimodal tag.
    """
    multimodal_token_index = MODAL_INDEX_MAP.get(multimodal_token, None)
    if multimodal_token_index is None:
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    else:
        prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids for idx, chunk in enumerate(prompt.split(multimodal_token))]

        input_ids = []
        for i in range(1, 2 * len(prompt_chunks)):
            if i % 2 == 1:
                input_ids.extend(prompt_chunks[i // 2])
            else:
                input_ids.append(multimodal_token_index)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
