import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, Menu, ttk
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
import sys
import os
import datetime
import warnings
import queue
import re
from torchvision.ops import nms
from urllib3.exceptions import SystemTimeWarning
warnings.filterwarnings("ignore", category=SystemTimeWarning)
import time
# 忽略 libpng 的 iCCP 警告
warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*", category=UserWarning)

# 设置matplotlib的全局字体参数以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

class TextRedirector:
    def __init__(self, text_widget, original_stream):
        self.text_widget = text_widget
        self.original_stream = original_stream

    def write(self, str):
        # 将文本插入到文本框
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, str)
        self.text_widget.configure(state='disabled')
        self.text_widget.see(tk.END)
        # 同时输出到原始命令行
        self.original_stream.write(str)

    def flush(self):
        pass  # 兼容性方法

class VideoApp:
    def __init__(self, root):

        self.root = root
        self.root.title("学生课堂行为识别系统")

        # 定义独立的跳帧参数
        self.BEHAVIOR_FRAME_SKIP = 1 

        # 设置窗口大小和位置
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.9)
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height - 80) // 2
        root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        self.default_confidence = {  # 默认置信度
            'hand-raising': 0.1,
            'reading': 0.1,
            'writing': 0.1,
            'using phone': 0.1,
            'bowing head': 0.05,
        }
        # 主框架
        self.main_frame = tk.Frame(root, bg='white')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建菜单栏
        menu_bar = tk.Menu(root)
        root.config(menu=menu_bar)

        # 创建菜单项
        self.video_menu = tk.Menu(menu_bar, tearoff=0)
        self.video_menu.add_command(label="上传视频", command=self.upload_video)
        self.video_menu.add_command(label="播放/暂停", command=self.toggle_play_pause)
        self.video_menu.add_command(label="停止", command=self.stop_playback)
        self.process_menu = tk.Menu(menu_bar, tearoff=0)
        self.process_menu.add_command(label="开始行为识别", command=self.toggle_behavior_detection)


        # 将菜单项添加到菜单栏
        menu_bar.add_cascade(label="视频源", menu=self.video_menu)
        menu_bar.add_cascade(label="视频处理", menu=self.process_menu)

        # 创建一个标题区域（位于按钮区域和菜单区域之间）
        self.title_frame = tk.Frame(self.main_frame, bg='white')
        self.title_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        # 在标题区域添加标题文本
        self.title_label = tk.Label(self.title_frame, text="学生课堂行为识别系统", font=("SimHei", 20, "bold"),
                                    bg='white')
        self.title_label.pack()

        # 创建一个新框架来放置按钮和控制台
        self.control_frame = tk.Frame(self.main_frame, bg='white')
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 状态标签
        self.status_label = tk.Label(self.control_frame, text="未加载视频", width=20, height=2)
        self.status_label.grid(row=0, column=1, padx=5, pady=5)
        # 返回按钮
        self.back_button = tk.Button(self.control_frame, text="返回", command=self.go_back, width=15, height=2)
        self.back_button.grid(row=0, column=0, padx=5, pady=5)

        # 控制台输出文本框
        self.console_text = scrolledtext.ScrolledText(self.control_frame, width=60, height=30, state='disabled',
                                                      bg='black', fg='white', wrap='word')
        self.console_text.grid(row=3, column=0, columnspan=2, padx=5, pady=10, sticky='nsew')
        sys.stdout = TextRedirector(self.console_text, sys.stdout)
        sys.stderr = TextRedirector(self.console_text, sys.stderr)

        # 5个框架的显示区域（初始化为frame0）
        self.frame0 = tk.Frame(self.control_frame)
        self.frame1 = tk.Frame(self.control_frame)
        self.frame2 = tk.Frame(self.control_frame)
        self.frame4 = tk.Frame(self.control_frame) 

        # 状态变量，用于保存按钮状态
        self.is_behavior_processing = False

        self.create_frame0()

        # 视频显示框架
        self.video_frame = tk.Frame(self.main_frame, bg='white')
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 设置画布尺寸
        self.canvas_width = int(window_width * 0.5)
        self.canvas_height = int(window_height * 0.42)

        # 原始和处理后的视频画布
        self.canvas_original = tk.Canvas(self.video_frame, bg='white', width=self.canvas_width, height=self.canvas_height)
        self.canvas_original.pack(side=tk.TOP, padx=10, pady=10, expand=True)

        self.canvas_processed = tk.Canvas(self.video_frame, bg='white', width=self.canvas_width, height=self.canvas_height)
        self.canvas_processed.pack(side=tk.BOTTOM, padx=10, pady=10, expand=True)

        # 初始化图像ID和ImageTk对象
        self.image_id_original = None
        self.image_tk_original = None
        self.image_id_processed = None
        self.image_tk_processed = None

        # 视频路径
        self.video_path = None

        # 统计数据
        self.action_stats = {}

        # 视频播放控制
        self.playing = False
        self.cap_1 = None
        self.delay = 1  # 约为30 FPS

        # 初始化处理状态标志
        self.processing = False  # 用于指示是否正在处理

        # 统计数据锁
        self.stats_lock = threading.Lock()

        # 当前播放帧数
        self.current_frame = 0

        # 总帧数
        self.total_frames = 0

        # 视频帧率
        self.fps = 30  # 默认帧率

        # 存储最新帧用于录入人脸
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # 创建队列用于传递原始和处理后的帧
        # 修改队列大小为1，以确保始终处理最新的帧
        self.raw_frame_queue = queue.Queue(maxsize=1)         
        self.processed_frame_queue = queue.Queue(maxsize=10)  

        # 启动一个后台线程处理帧
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

        # 启动一个定期检查队列的主线程任务
        self.root.after(30, self.update_processed_canvas)

        # 初始化模型
        self.init_models()

        # 处理程序退出时，确保后台线程停止
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.running = True  

    def create_frame0(self):
        """创建frame0并显示菜单按钮"""
        # 清空frame0中的内容
        for widget in self.frame0.winfo_children():
            widget.destroy()

        # 视频源、视频处理按钮
        button_video_source = tk.Button(self.frame0, text="视频源", width=15, height=2, command=self.show_video_frame)
        button_video_source.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        button_process = tk.Button(self.frame0, text="视频处理", width=15, height=2, command=self.show_process_frame)
        button_process.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
     
        self.frame0.grid_columnconfigure(0, weight=1)
        self.frame0.grid_columnconfigure(1, weight=1)
        self.frame0.grid(row=2, column=0, columnspan=2) 

    def show_video_frame(self):
        """显示视频源相关按钮"""
        self.hide_all_frames()
        self.create_video_frame()
        self.frame1.grid(row=2, column=0, columnspan=2)

    def create_video_frame(self):
        """创建视频源相关的按钮"""
        for widget in self.frame1.winfo_children():
            widget.destroy()

        # 视频源相关操作按钮
        self.button_upload_video = tk.Button(self.frame1, text="上传视频", command=self.upload_video, width=15, height=2)
        self.button_upload_video.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.button_play = tk.Button(self.frame1, text="播放/暂停", command=self.toggle_play_pause, width=15, height=2)
        self.button_play.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.button_stop = tk.Button(self.frame1, text="停止", command=self.stop_playback, width=15, height=2)
        self.button_stop.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # 配置frame1的列权重
        self.frame1.grid_columnconfigure(0, weight=1)
        self.frame1.grid_columnconfigure(1, weight=1)

    def show_process_frame(self):
        self.hide_all_frames()
        self.create_process_frame()
        self.frame2.grid(row=2, column=0, columnspan=2)

    def create_process_frame(self):
        for widget in self.frame2.winfo_children():
            widget.destroy()

        # 行为识别按钮
        behavior_text = "停止行为识别" if self.is_behavior_processing else "开始行为识别"
        self.button_behavior_recognition = tk.Button(self.frame2, text=behavior_text,
                                                command=self.toggle_behavior_detection, width=15, height=2)
        self.button_behavior_recognition.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # 显示行为统计按钮
        self.button_show_statistics = tk.Button(self.frame2, text="显示行为统计", 
                                                 command=self.show_statistics, width=15, height=2)
        self.button_show_statistics.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # 配置frame2的列权重
        self.frame2.grid_columnconfigure(0, weight=1)
        self.frame2.grid_columnconfigure(1, weight=1)

    def hide_all_frames(self):
        self.frame0.grid_forget()
        self.frame1.grid_forget()
        self.frame2.grid_forget()
        if hasattr(self, 'frame4'): 
             self.frame4.grid_forget()

    def go_back(self):

        self.hide_all_frames()
        self.create_frame0()
        self.frame0.grid(row=2, column=0, columnspan=2)

    def on_closing(self):
        # 停止播放
        self.playing = False

        # 释放视频捕获对象
        if self.cap_1 is not None and self.cap_1.isOpened():
            self.cap_1.release()
  
        # 关闭主窗口
        self.root.destroy()

    def init_models(self):
        # 初始化所需的模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 实例化 YOLO 模型，用于行为检测
        try:
            print("加载行为检测 YOLO 模型...")
            self.yolo_model = YOLO('model_data/best.pt')
            self.yolo_model.to(self.device)
            print("行为检测 YOLO 模型加载成功。")
        except Exception as e:
            print(f"加载行为检测 YOLO 模型时发生错误: {e}")
            messagebox.showerror("错误", f"加载行为检测 YOLO 模型失败：{e}")

        # 实例化 YOLO 模型，用于人物检测
        try:
            print("加载人物检测 YOLO 模型...")
            self.person_model = YOLO('model_data/yolov8l.pt') 
            self.person_model.to(self.device)
            print("人物检测 YOLO 模型加载成功。")
        except Exception as e:
            print(f"加载人物检测 YOLO 模型时发生错误: {e}")
            messagebox.showerror("错误", f"加载人物检测 YOLO 模型失败：{e}")

        self.update_status_label()

    def check_processing(self):
        with self.stats_lock:
            if self.processing:
                messagebox.showinfo("处理中", "视频正在处理，请稍后再操作。")
                return True
        return False

    def upload_video(self):
        if self.check_processing():
            return

        # 选择视频文件
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if not self.video_path:
            return

        # 释放之前的cap_1
        if self.cap_1 is not None:
            if self.cap_1.isOpened():
                self.cap_1.release()
            self.cap_1 = None

        # 获取视频帧率和总帧数
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开视频。")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 显示第一帧
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开视频。")
            return

        ret, frame = cap.read()
        if ret:
            self.display_frame(frame, self.canvas_original)
            with self.frame_lock:
                self.latest_frame = frame.copy()
        cap.release()
        self.cap_1 = None  # 确保在播放时重新创建捕获对象
        self.playing = False
        self.current_frame = 0  # 重置当前帧数
        self.update_status_label()

    def toggle_behavior_detection(self):
        if self.is_behavior_processing:
            # 停止行为识别
            self.is_behavior_processing = False
            # 使用整数索引更新菜单项
            self.process_menu.entryconfig(0, label="开始行为识别")
            self.button_behavior_recognition.config(text="开始行为识别")
            print("停止行为识别")
            self.update_status_label()
        else:
            # 启动行为识别
            if not self.video_path:
                messagebox.showerror("错误", "请先上传视频。") 
                return

            self.is_behavior_processing = True
            # 使用整数索引更新菜单项
            self.process_menu.entryconfig(0, label="停止行为识别")
            self.button_behavior_recognition.config(text="停止行为识别")
            print("开始行为识别")
            self.update_status_label()

    def show_statistics(self):
        with self.stats_lock:
            if self.processing:
                messagebox.showinfo("处理中", "视频正在处理，请稍后再操作。")
                return

            if not self.action_stats:
                messagebox.showerror("错误", "暂无统计数据。请先处理视频。")
                return

            # 过滤掉次数为0的行为类别
            filtered_action_stats = {k: v for k, v in self.action_stats.items() if v > 0}
            if not filtered_action_stats:
                messagebox.showinfo("无数据", "没有检测到任何行为。")
                return
                
            labels = list(filtered_action_stats.keys())
            values = list(filtered_action_stats.values())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 条形图
        axes[0].bar(labels, values, color='skyblue')
        axes[0].set_title('行为频率（条形图）') 
        axes[0].set_xlabel('行为')
        axes[0].set_ylabel('频率') 
        axes[0].tick_params(axis='x', rotation=45)
        for label in axes[0].get_xticklabels(): 
            label.set_fontfamily('SimHei')


        # 饼图
        textprops = {'fontfamily': 'SimHei', 'fontsize': 10}
        wedges, texts, autotexts = axes[1].pie(values, autopct='%1.1f%%', startangle=140, textprops=textprops)
        axes[1].set_title('行为频率（饼图）') 
        axes[1].legend(wedges, labels, title="行为", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), prop={'family': 'SimHei'})


        plt.tight_layout()
        plt.show()

    def toggle_play_pause(self):
        if self.playing:
            self.playing = False
            print("暂停播放")
            self.update_status_label()
        else:
            if not self.video_path:
                messagebox.showerror("错误", "请先上传至少一个视频或连接摄像头。")
                return

            self.playing = True
            print("开始播放")
            # 重置当前帧数以确保重新播放
            self.current_frame = 0
            self.play_video()

    def stop_playback(self):
        if self.playing:
            self.playing = False
            print("停止播放")

        # 释放视频捕获对象
        if self.cap_1 is not None:
            if self.cap_1.isOpened():
                self.cap_1.release()
            self.cap_1 = None

        # 重置视频显示到第一帧
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.display_frame(frame, self.canvas_original)
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
            cap.release()

            # 更新处理后的视频显示到第一帧
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.display_frame(frame, self.canvas_processed)
            cap.release()

        # 清空队列
        with self.raw_frame_queue.mutex:
            self.raw_frame_queue.queue.clear()
        with self.processed_frame_queue.mutex:
            self.processed_frame_queue.queue.clear()

        # 更新播放按钮文本
        self.update_status_label()
        self.current_frame = 0
        print("视频已重置到开始位置")

    def play_video(self):
        if not self.playing:
            return

        # 处理视频
        if self.cap_1 is None:
            self.cap_1 = cv2.VideoCapture(self.video_path)
            if not self.cap_1.isOpened():
                messagebox.showerror("错误", "无法打开视频。")
                self.cap_1 = None
                self.playing = False
                return
            # 重置当前帧数
            self.current_frame = 0

        ret, frame = self.cap_1.read()
        if ret:
            self.display_frame(frame, self.canvas_original)
            with self.frame_lock:
                self.latest_frame = frame.copy()
            self.current_frame += 1
            self.update_status_label()

            # 将帧放入raw_frame_queue进行处理
            try:
                self.raw_frame_queue.put_nowait(frame.copy())
            except queue.Full:
                try:
                    self.raw_frame_queue.get_nowait()  # 移除旧帧
                    self.raw_frame_queue.put_nowait(frame.copy())  # 放入新帧
                except queue.Empty:
                    pass

            if self.current_frame <= self.total_frames:
                self.root.after(int(1000 / self.fps), self.play_video)
            else:
                self.playing = False  # 视频播放完毕
                print("视频播放完毕")
                self.update_status_label()
                self.cap_1.release()
                self.cap_1 = None
        else:
            self.playing = False  # 播放完毕
            print("视频播放完毕")
            self.update_status_label()
            if self.cap_1 is not None:
                self.cap_1.release()
                self.cap_1 = None

    def process_frames(self):
        while True:
            try:
                frame = self.raw_frame_queue.get(timeout=1)
            except queue.Empty:
                continue
    
            processed_frame = frame.copy()
            action_stats = {}
    
            with self.stats_lock:
                self.processing = True
    
            try:
                current_conf = self.default_confidence
    
                # 行为识别
                if self.is_behavior_processing:
                    frame_after_behavior, action_stats = process_behavior_detection(
                        frame, 
                        self.yolo_model,
                        self.person_model,
                        self.device,
                        confidence_config=current_conf,
                        IOU_THRESHOLD=0.1
                    )
    
                    # 累积行为统计
                    for k, v in action_stats.items():
                        self.action_stats[k] = self.action_stats.get(k, 0) + v
                    
                    # 输出检测到的行为
                    detected_behaviors = {k: v for k, v in action_stats.items() if v > 0}
                    if detected_behaviors:
                        behaviors_str = ', '.join([f"{k}: {v}" for k, v in detected_behaviors.items()])
                        print(f" {behaviors_str}")
                    processed_frame = frame_after_behavior
    
                try:
                    self.processed_frame_queue.put_nowait(processed_frame.copy())
                except queue.Full:
                    pass
    
            except Exception as e:
                print(f"处理帧时发生错误: {e}")
            finally:
                with self.stats_lock:
                    self.processing = False

    def update_processed_canvas(self):
        try:
            while not self.processed_frame_queue.empty():
                frame = self.processed_frame_queue.get_nowait()
                self.display_frame(frame, self.canvas_processed)
        except queue.Empty:
            pass
        finally:
            self.root.after(30, self.update_processed_canvas)

    def display_frame(self, frame, canvas):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # 使用预设的画布尺寸
            canvas_width = self.canvas_width
            canvas_height = self.canvas_height

            # 计算缩放比例，保持原始比例
            img_width, img_height = img.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_size = (int(img_width * ratio), int(img_height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 计算图像在画布中的位置，使其居中显示
            x = (canvas_width - new_size[0]) // 2
            y = (canvas_height - new_size[1]) // 2

            img_tk = ImageTk.PhotoImage(image=img)

            if canvas == self.canvas_original:
                if self.image_id_original is None:
                    self.image_id_original = canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
                else:
                    canvas.coords(self.image_id_original, x, y)
                    canvas.itemconfig(self.image_id_original, image=img_tk)
                self.image_tk_original = img_tk
            elif canvas == self.canvas_processed:
                if self.image_id_processed is None:
                    self.image_id_processed = canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
                else:
                    canvas.coords(self.image_id_processed, x, y)
                    canvas.itemconfig(self.image_id_processed, image=img_tk)
                self.image_tk_processed = img_tk
        except Exception as e:
            print(f"显示帧时发生错误: {e}")

    def update_status_label(self):
        if self.video_path:
            if self.total_frames > 0:
                self.status_label.config(text=f"当前帧: {self.current_frame} / {self.total_frames}")
            else:
                self.status_label.config(text=f"当前帧: {self.current_frame}")
        else:
            self.status_label.config(text="未加载视频")

def process_behavior_detection(frame, behavior_model, person_model, device, confidence_config, IOU_THRESHOLD):
   
    # 定义行为类别
    BEHAVIOR_CLASSES = [
        'hand-raising',
        'reading',
        'writing',
        'using phone',
        'bowing head',
        'normal'
    ]
    
    behavior_adjustment = {
        'hand-raising': 1,
        'reading': 1.2,
        'writing': 1.4,
        'using phone': 1,
        'bowing head': 1.5,
    }

    BEHAVIOR_COLORS = {
        'hand-raising': (0, 255, 0),
        'reading': (255, 255, 0),
        'writing': (255, 0, 0),
        'using phone': (0, 0, 255),
        'bowing head': (0, 255, 255),
    }

    action_stats = {behavior: 0 for behavior in BEHAVIOR_CLASSES if behavior in BEHAVIOR_COLORS}

    # 1. 行为检测
    behavior_results = behavior_model(frame, device=device, verbose=False)
    raw_behavior_detections = []
    for result in behavior_results:
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            conf = box.conf[0].item()
            class_name = behavior_model.names.get(cls_idx)
            
            if class_name and class_name in BEHAVIOR_CLASSES:
                adjusted_conf = conf
                if class_name in behavior_adjustment:
                    adjusted_conf *= behavior_adjustment[class_name]
                
                if adjusted_conf > confidence_config.get(class_name, 0.1):
                    raw_behavior_detections.append({
                        'box': box.xyxy[0].tolist(),
                        'class': class_name,
                        'conf': adjusted_conf
                    })

    # 2. 人物检测
    person_results = person_model(frame, device=device, verbose=False)
    detected_persons_boxes = []
    for result in person_results:
        for box_obj in result.boxes: 
            if int(box_obj.cls[0]) == 0: 
                 if box_obj.conf[0].item() > 0.3: 
                    person_box_coords = box_obj.xyxy[0].tolist()
                    detected_persons_boxes.append(person_box_coords)
                    plot_one_box(person_box_coords, frame, label="person", color=(255, 255, 255), line_thickness=2)

    final_behavior_detections = []
    if detected_persons_boxes:
        for behavior_det in raw_behavior_detections:
            behavior_box = behavior_det['box']
            max_iou_with_person = 0
            for person_box in detected_persons_boxes:
                iou = compute_iou(behavior_box, person_box)
                if iou > max_iou_with_person:
                    max_iou_with_person = iou
            
            if max_iou_with_person > IOU_THRESHOLD:
                final_behavior_detections.append(behavior_det)

    # 3. 绘制与人物关联的行为检测框并统计
    for detection in final_behavior_detections:
        behavior_class = detection['class']
        box = detection['box']
        label_text = f"{behavior_class}: {detection['conf']:.2f}" 
        color = BEHAVIOR_COLORS.get(behavior_class, (0, 255, 255))
        plot_one_box(box, frame, label=label_text, color=color, line_thickness=2)
        if behavior_class in action_stats:
            action_stats[behavior_class] += 1
            
    return frame, action_stats

def compute_iou(box1, box2):

    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # 计算交集
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # 计算并集
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    # 计算IOU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def plot_one_box(x, img, color=(0, 255, 0), label=None, line_thickness=2):

    # 解析坐标
    x1, y1, x2, y2 = map(int, x)
    # 绘制矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=line_thickness, lineType=cv2.LINE_AA)

    if label:
        # 获取文本大小
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 绘制标签背景
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        # 绘制标签文本
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1,
                    lineType=cv2.LINE_AA)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
