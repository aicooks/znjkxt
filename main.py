import sys
import cv2
import os
import time
import json
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
from ultralytics import YOLO
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QComboBox, QTextEdit, QFileDialog, QHBoxLayout, QVBoxLayout, QFrame, QMessageBox, QDateTimeEdit, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer, QDateTime
from PySide6.QtGui import QImage, QPixmap, QTextCursor, QIcon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))
# 默认配置
DEFAULT_CONFIG = {
    "model": "yolov8n.pt",
    "conf_thres": 0.5,
    "save_dir": "detected_objects",
    "target_classes": ["person"],
    "cooldown": 2,
    "camera_id": 0,
    "use_gpu": False,
    "show_fps": True
}


class VideoThread(QThread):
    update_frame = Signal(QImage)
    update_fps = Signal(int)
    log_signal = Signal(str)
    update_target_count = Signal(int)

    def __init__(self, camera_id=0, model=None, config=None, detect_mode=True):
        super().__init__()
        self.cap = None  # 视频捕获对象属性
        self.frame_size = (640, 480)  # 默认帧尺寸
        self.camera_id = camera_id
        self.running = False
        self.fps_counter = 0
        self.start_time = time.time()
        self.model = model
        self.config = config
        self.last_save_time = {}

        self.detect_mode = detect_mode

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.frame_size = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 转换为RGB格式
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.detect_mode:
                # 检测目标
                results = self.model.predict(
                    source=rgb_image,
                    conf=self.config["conf_thres"],
                    verbose=False,
                )
                # 绘制检测结果
                self.process_detections(rgb_image, results)

            # 更新帧
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.update_frame.emit(qt_image)

            # 计算FPS
            self.fps_counter += 1
            if time.time() - self.start_time >= 1:
                self.update_fps.emit(self.fps_counter)
                self.fps_counter = 0
                self.start_time = time.time()

        self.cap.release()

    def process_detections(self, frame, results):

        current_time = time.time()
        target_count = 0  # 用于统计目标数量

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]
                conf = box.conf.item()
                if class_name in self.config["target_classes"]:
                    # 绘制检测框
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    # 确保坐标有效性
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    if x1 >= x2 or y1 >= y2:  # 跳过无效区域
                        continue

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # 保存检测到的对象（新增异常处理）
                    if (current_time - self.last_save_time.get(class_name, 0)) > self.config["cooldown"]:
                        cropped = frame[y1:y2, x1:x2]
                        save_dir = self.config["save_dir"]
                        try:
                            os.makedirs(save_dir, exist_ok=True)
                            # 格式化置信度为两位小数
                            conf_str = f"{conf:.2f}".replace('.', '_')
                            filename = os.path.join(
                                save_dir,
                                f"{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{conf_str}.jpg"
                            )
                            # 保存
                            cv2.imwrite(filename, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                            self.log_signal.emit(f"检测到 {class_name}, 已保存：{filename}")

                            self.last_save_time[class_name] = current_time
                        except Exception as e:
                            self.log_signal.emit(f"保存失败：{str(e)}")

                    target_count += 1  # 增加目标数量计数

        self.update_target_count.emit(target_count)  # 发送目标数量信号


    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        # 写入日志文件
        log_file_path = os.path.join(self.log_dir, f"log_{datetime.now().strftime('%Y%m%d')}.txt")
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(log_entry)

        # 显示在QTextEdit中
        self.log_edit.append(log_entry)
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("安防监控系统")
        self.setGeometry(100, 100, 1000, 600)
        self.is_recording = False
        self.recording_writer = None
        self.recording_start_time = None
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_time)

        # 创建日志文件夹
        self.log_dir = os.path.join(os.getcwd(), "log")
        os.makedirs(self.log_dir, exist_ok=True)

        self.config = self.load_config()
        self.video_thread = None

        self.model = None

        # 创建 QTextEdit 作为历史记录显示区域
        self.history_window = QTextEdit()
        self.history_window.setReadOnly(True)

        # 创建一个新的 QWidget 作为 history_window 的容器
        self.history_container = QWidget()
        self.history_container.setWindowTitle("历史记录")
        self.history_container.setGeometry(100, 100, 600, 400)

        # 创建检测类型下拉组件
        self.history_class_combo = QComboBox()
        self.history_class_combo.addItems([""] + ["person", "car", "dog", "cat", "bicycle", "bird", "traffic light"])
        self.history_class_combo.setCurrentText("")  # 初始值为空
        self.history_class_name = QLabel("类型：")

        # 创建时间范围选择组件
        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setCalendarPopup(True)
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(time.time())))  # 初始值为当前时间
        self.start_time_edit.setReadOnly(False)  # 设置为只读
        self.history_start_time_name = QLabel("开始日期：")

        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setCalendarPopup(True)
        self.end_time_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(time.time())))  # 初始值为当前时间
        self.end_time_edit.setReadOnly(False)  # 设置为只读
        self.history_end_time_name = QLabel("结束日期：")

        # 创建导出按钮
        self.export_btn = QPushButton("导出历史记录")
        self.export_btn.clicked.connect(self.export_history)

        self.report_type_name = QLabel("报告类型：")
        report_type_label = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.report_type_name.setSizePolicy(report_type_label)

        self.report_type_combo = QComboBox()
        self.report_type_combo.addItems(["HTML 报告", "PDF 报告"])
        self.generate_report_btn = QPushButton("生成统计报告")
        self.generate_report_btn.clicked.connect(self.generate_statistical_report)

        # 日志区域
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)

        self.init_ui()
        self.init_model()

        self.raise_()
        self.activateWindow()

    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # 视频显示区域
        video_widget = QWidget()
        video_layout = QVBoxLayout()

        self.video_label = QLabel()

        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)  # 左上对齐

        self.status_label = QLabel("状态: 待机")
        self.fps_label = QLabel("FPS: 0")
        self.target_count_label = QLabel("目标数量: 0")

        video_layout.addWidget(self.video_label)
        # 视频显示区域布局
        video_info_layout = QHBoxLayout()
        video_info_layout.addWidget(self.status_label)
        video_info_layout.addWidget(self.fps_label)
        video_info_layout.addWidget(self.target_count_label)

        # 添加摄像头选择下拉菜单
        self.camera_combo = QComboBox()
        self.populate_camera_list()  # 填充摄像头列表
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)  # 连接信号
        video_info_layout.addWidget(self.camera_combo)

        # 添加录制按钮
        self.record_btn = QPushButton("开始录制")
        self.record_btn.clicked.connect(self.toggle_recording)
        video_info_layout.addWidget(self.record_btn)

        video_layout.addLayout(video_info_layout)
        video_widget.setLayout(video_layout)

        # 控制面板
        control_widget = QFrame()
        control_widget.setFrameShape(QFrame.Shape.StyledPanel)
        control_layout = QVBoxLayout()

        # 添加仅打开摄像头按钮
        self.open_camera_btn = QPushButton("开启摄像头")
        self.open_camera_btn.clicked.connect(self.open_camera)


        # 控制按钮
        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self.toggle_camera)

        # 重置按钮
        self.reset_btn = QPushButton("重置配置")
        self.reset_btn.clicked.connect(self.reset_to_default_config)
        # 导入配置按钮
        self.import_config_btn = QPushButton("导入配置")
        self.import_config_btn.clicked.connect(self.import_config)
        # 添加导出配置按钮
        self.export_config_btn = QPushButton("导出配置")
        self.export_config_btn.clicked.connect(self.export_config)

        reset_import_export_layout = QHBoxLayout()
        reset_import_export_layout.addWidget(self.reset_btn)
        reset_import_export_layout.addWidget(self.import_config_btn)
        reset_import_export_layout.addWidget(self.export_config_btn)

        control_layout.addLayout(reset_import_export_layout)

        # 配置参数
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.config["conf_thres"] * 100))
        self.conf_slider.valueChanged.connect(self.update_conf_thres)

        self.conf_value_label = QLabel(f"{self.config['conf_thres']:.2f}")  # 添加置信度阈值标签

        # 创建一个水平布局来放置标签和值标签
        conf_layout = QHBoxLayout()
        self.conf_value = QLabel(f"置信度阈值: {self.config['conf_thres']:.2f}")
        conf_layout.addWidget(self.conf_value)
        control_layout.addLayout(conf_layout)

        self.class_combo = QComboBox()
        self.class_combo.addItems(["person", "car", "dog", "cat", "bicycle", "bird", "traffic light"])
        self.class_combo.setCurrentText(self.config["target_classes"][0])
        self.class_combo.currentTextChanged.connect(self.update_target_classes)
        self.save_dir_edit = QTextEdit(self.config["save_dir"])
        self.save_dir_edit.setMaximumHeight(30)
        self.save_dir_edit.setReadOnly(True)
        self.browse_btn = QPushButton("修改监测目标保存路径")
        self.browse_btn.clicked.connect(self.select_save_dir)

        self.cooldown_edit = QTextEdit(str(self.config["cooldown"]))
        self.cooldown_edit.setMaximumHeight(30)



        history_view_layout = QHBoxLayout()
        # 历史记录查看按钮
        self.history_btn = QPushButton("查看历史记录")
        self.history_btn.clicked.connect(self.show_history)
        # 查看日志文件按钮
        self.view_log_btn = QPushButton("查看日志文件")
        self.view_log_btn.clicked.connect(self.view_log_files)
        history_view_layout.addWidget(self.history_btn)
        history_view_layout.addWidget(self.view_log_btn)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_camera_btn)
        button_layout.addWidget(self.start_btn)

        self.class_combo_layout = QHBoxLayout()

        # 设置 QLabel 的 sizePolicy
        self.class_label = QLabel("目标类别:")
        size_policy_label = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.class_label.setSizePolicy(size_policy_label)
        self.class_combo_layout.addWidget(self.class_label)

        # 设置 QComboBox 的 sizePolicy
        size_policy_combo = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.class_combo.setSizePolicy(size_policy_combo)
        self.class_combo_layout.addWidget(self.class_combo)

        # 布局组装
        control_layout.addWidget(self.conf_slider)
        control_layout.addLayout(self.class_combo_layout)
        control_layout.addWidget(QLabel("监测目标保存路径:"))
        control_layout.addWidget(self.save_dir_edit)
        control_layout.addWidget(self.browse_btn)
        control_layout.addWidget(QLabel("冷却时间(秒):"))
        control_layout.addWidget(self.cooldown_edit)
        control_layout.addLayout(button_layout)
        control_layout.addWidget(QLabel("事件日志:"))
        control_layout.addWidget(self.log_edit)
        control_layout.addLayout(history_view_layout)
        control_widget.setLayout(control_layout)

        # 主布局组合
        main_layout.addWidget(video_widget)
        main_layout.addWidget(control_widget)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def generate_statistical_report(self):
        # 获取历史记录内容
        history_content = self.history_window.toPlainText()

        # 解析历史记录内容
        records = []
        for line in history_content.split('\n'):
            if not line:
                continue
            parts = line.split(' 检测到: ')
            if len(parts) != 2:
                continue
            detection_time_str, filename = parts
            try:
                # 解析时间（格式：2023-03-15 14:30:45）
                detection_time = datetime.strptime(detection_time_str, "%Y-%m-%d %H:%M:%S")

                # 从文件名解析类别和置信度（格式：person_20240312_123456_0_75.jpg）
                filename_parts = filename.split('_')
                class_name = filename_parts[0]
                conf_str = filename_parts[-1].split('.')[0]  # 获取0_75部分
                confidence = float(conf_str.replace('_', '.'))  # 转换为0.75

                records.append({
                    "time": detection_time,
                    "class": class_name,
                    "confidence": confidence
                })
            except Exception as e:
                self.log_message(f"解析记录错误: {str(e)}")
                continue

        if not records:
            QMessageBox.warning(self, "警告", "没有可用的检测记录")
            return

        # 计算统计信息
        total_detections = len(records)
        avg_confidence = sum(r["confidence"] for r in records) / total_detections

        # 目标类别分布
        class_dist = {}
        for r in records:
            class_name = r["class"]
            class_dist[class_name] = class_dist.get(class_name, 0) + 1

        # 分时段检测趋势（按小时）
        hourly_dist = {}
        for r in records:
            hour = r["time"].strftime("%H:00")
            hourly_dist[hour] = hourly_dist.get(hour, 0) + 1

        # 创建报告目录
        report_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(report_dir, exist_ok=True)
        # 创建临时目录保存图表
        temp_dir = os.path.join(report_dir, "temp_charts")
        os.makedirs(temp_dir, exist_ok=True)
        # 获取时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 生成图表
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体
            plt.rcParams['axes.unicode_minus'] = False

            # 类别分布柱状图
            plt.figure(figsize=(8, 4))
            classes = list(class_dist.keys())
            counts = list(class_dist.values())
            plt.bar(classes, counts)
            plt.title("目标类别分布")
            plt.xlabel("目标类别")
            plt.ylabel("检测次数")
            class_chart_path = os.path.join(temp_dir, f"class_dist_{timestamp}.png")
            plt.savefig(class_chart_path)
            plt.close()

            # 分时段趋势折线图
            hours = sorted(hourly_dist.keys())
            values = [hourly_dist[h] for h in hours]
            plt.figure(figsize=(8, 4))
            plt.plot(hours, values, marker='o')
            plt.title("分时段检测趋势")
            plt.xlabel("时间段")
            plt.ylabel("检测次数")
            plt.xticks(rotation=45)
            time_chart_path = os.path.join(temp_dir, f"time_trend_{timestamp}.png")
            plt.savefig(time_chart_path)
            plt.close()
        except ImportError:
            QMessageBox.warning(self, "警告", "需要安装matplotlib来生成图表")
            return

        # 获取报告类型
        report_type = self.report_type_combo.currentText()

        # 获取导出的类型和时间范围
        selected_class = self.history_class_combo.currentText()
        start_time = self.start_time_edit.dateTime().toString("yyyy-MM-dd")
        end_time = self.end_time_edit.dateTime().toString("yyyy-MM-dd")

        # 报告内容参数
        report_params = {
            "report_title": "检测统计报告",
            "total_detections": total_detections,
            "avg_confidence": avg_confidence,
            "selected_class": selected_class if selected_class else "全部类型",
            "time_range": f"{start_time} 至 {end_time}",
            "class_chart": class_chart_path,
            "time_chart": time_chart_path,
        }

        if report_type == "HTML 报告":
            # 生成HTML报告
            self.generate_html_report(report_dir, timestamp, report_params)
        else:
            # 生成PDF报告
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer

            report_file_path = os.path.join(report_dir, f"report_{timestamp}.pdf")

            doc = SimpleDocTemplate(report_file_path, pagesize=letter)
            styles = getSampleStyleSheet()

            # 设置默认字体
            styles['Normal'].fontName = 'SimSun'
            styles['Title'].fontName = 'SimSun'
            styles['Heading2'].fontName = 'SimSun'

            story = []

            # 标题
            story.append(Paragraph("检测统计报告", styles['Title']))
            story.append(Spacer(1, 12))

            # 添加导出的类型和时间范围
            story.append(Paragraph(f"导出类型: {selected_class if selected_class else '全部类型'}", styles['Normal']))
            story.append(Paragraph(f"时间范围: {start_time} 至 {end_time}", styles['Normal']))
            story.append(Spacer(1, 12))

            # 基本信息
            story.append(Paragraph(f"总检测次数: {total_detections}", styles['Normal']))
            story.append(Paragraph(f"平均置信度: {avg_confidence:.2f}", styles['Normal']))
            story.append(Spacer(1, 24))

            # 类别分布图
            story.append(Paragraph("目标类别分布", styles['Heading2']))
            img = Image(class_chart_path, width=400, height=200)
            story.append(img)
            story.append(Spacer(1, 24))

            # 分时段趋势图
            story.append(Paragraph("分时段检测趋势", styles['Heading2']))
            img = Image(time_chart_path, width=400, height=200)
            story.append(img)

            doc.build(story)
            # 显示结果
            self.log_message(f"统计报告已生成: {report_file_path}")
            QMessageBox.information(self, "完成", f"报告已生成:\n{report_file_path}")

        # # 清理临时文件
        # for f in [class_chart_path, time_chart_path]:
        #     if os.path.exists(f):
        #         os.remove(f)
    def generate_html_report(self, report_dir, timestamp, params):
        class_chart_filename = os.path.basename(params['class_chart'])
        time_chart_filename = os.path.basename(params['time_chart'])

        # 将图片路径改为相对路径（相对于 HTML 文件）
        params['class_chart'] = f"temp_charts/{class_chart_filename}"
        params['time_chart'] = f"temp_charts/{time_chart_filename}"
        report_file_path = os.path.join(report_dir, f"report_{timestamp}.html")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{params['report_title']}</title>
            <style>
                body {{ font-family: 'SimSun', sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
                h2 {{ color: #34495e; }}
                .section {{ margin-bottom: 30px; }}
                .chart {{ 
                    border: 1px solid #ddd; 
                    padding: 10px;
                    margin: 20px 0;
                    text-align: center;
                }}
                .chart img {{ max-width: 80%; height: auto; }}
                .info-table {{ 
                    width: 100%; 
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .info-table td, .info-table th {{
                    padding: 12px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                .info-table th {{
                    background-color: #f8f9fa;
                }}
            </style>
        </head>
        <body>
            <h1>{params['report_title']}</h1>

            <div class="section">
                <h2>筛选条件</h2>
                <table class="info-table">
                    <tr>
                        <th>目标类型</th>
                        <td>{params['selected_class']}</td>
                    </tr>
                    <tr>
                        <th>时间范围</th>
                        <td>{params['time_range']}</td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <h2>统计概览</h2>
                <table class="info-table">
                    <tr>
                        <th>总检测次数</th>
                        <td>{params['total_detections']}</td>
                    </tr>
                    <tr>
                        <th>平均置信度</th>
                        <td>{params['avg_confidence']:.2f}</td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <h2>目标类别分布</h2>
                <div class="chart">
                    <img src="{params['class_chart']}" alt="类别分布图">
                </div>
            </div>

            <div class="section">
                <h2>分时段检测趋势</h2>
                <div class="chart">
                    <img src="{params['time_chart']}" alt="分时段趋势图">
                </div>
            </div>
        </body>
        </html>
        """

        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.log_message(f"HTML报告已生成: {report_file_path}")
        QMessageBox.information(self, "完成", f"报告已生成:\n{report_file_path}")
    def import_config(self):
        # 打开文件对话框选择配置文件
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if not file_path:
            return
        reply = QMessageBox.question(self, "导入系统配置", "您确定要更新系统配置吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        new_config = json.load(f)
                    # 更新当前配置
                    self.config.update(new_config)
                    # 更新UI组件
                    self.conf_slider.setValue(int(self.config["conf_thres"] * 100))
                    self.conf_value_label.setText(f"{self.config['conf_thres']:.2f}")
                    self.class_combo.setCurrentText(self.config["target_classes"][0])
                    self.save_dir_edit.setText(self.config["save_dir"])
                    self.cooldown_edit.setText(str(self.config["cooldown"]))
                    self.log_message(f"配置文件{file_path}已成功导入系统")
                except Exception as e:
                    QMessageBox.warning(self, "导入失败", f"无法导入配置文件: {str(e)}")
                    self.log_message(f"导入配置文件失败: {str(e)}")
    def export_config(self):
        # 获取当前时间作为文件名的一部分
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"config_{timestamp}.json"

        # 打开文件对话框选择导出文件的位置
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "选择配置文件导出位置", default_filename,
                                                   "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=4)
                self.log_message(f"配置已导出到: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", f"无法导出配置文件: {str(e)}")
                self.log_message(f"导出配置文件失败: {str(e)}")

    def reset_to_default_config(self):
        reply = QMessageBox.question(self, "重置配置", "您确定要重置为系统默认配置吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        self.config = DEFAULT_CONFIG.copy()
        self.conf_slider.setValue(int(self.config["conf_thres"] * 100))
        self.conf_value_label.setText(f"{self.config['conf_thres']:.2f}")
        self.class_combo.setCurrentText(self.config["target_classes"][0])
        self.save_dir_edit.setText(self.config["save_dir"])
        self.cooldown_edit.setText(str(self.config["cooldown"]))
        self.log_message("系统配置已重置为默认配置")

    def open_camera(self):
        if self.video_thread and self.video_thread.isRunning():
            self.stop_camera()
            self.log_message("当前摄像头已停止")
            self.open_camera_btn.setText("开启摄像头")
        else:
            selected_camera_text = self.camera_combo.currentText()
            if "摄像头" in selected_camera_text:
                camera_id = int(selected_camera_text.split(" ")[1])  # 提取摄像头 ID
                self.start_camera_only(camera_id)
                self.open_camera_btn.setText("关闭摄像头")
            else:
                QMessageBox.warning(self, "警告", "未检测到可用摄像头！")
                return

    def start_camera_only(self, camera_id):
        """启动指定ID的摄像头，但不进行目标检测、画框和抓拍"""
        if self.video_thread and self.video_thread.isRunning():
            self.stop_camera()
            self.log_message("当前摄像头已停止")
            self.open_camera_btn.setText("启动")
        else:
            self.video_thread = VideoThread(camera_id, self.model, self.config, detect_mode=False)
            self.video_thread.update_frame.connect(self.update_video)
            self.video_thread.update_target_count.connect(self.update_target_counter)
            self.video_thread.update_fps.connect(self.update_fps_display)
            self.video_thread.log_signal.connect(self.handle_log_message)
            self.video_thread.start()

            self.status_label.setText("状态: 运行中（未抓拍）")
            self.open_camera_btn.setText("停止")
            self.update_target_counter(0)  # 初始化目标数量为零

    def on_camera_changed(self, index):
        selected_camera_text = self.camera_combo.currentText()
        """当用户选择不同的摄像头时，停止当前摄像头并启动新选择的摄像头"""
        if self.video_thread and self.video_thread.isRunning():
            # 保存当前摄像头的状态
            detect_mode = self.video_thread.detect_mode
            self.stop_camera()
            self.log_message("当前摄像头已停止")
            if "摄像头" in selected_camera_text:
                camera_id = int(selected_camera_text.split(" ")[1])  # 提取摄像头 ID
                if detect_mode:
                    self.start_camera(camera_id)
                else:
                    self.start_camera_only(camera_id)
            else:
                QMessageBox.warning(self, "警告", "未检测到可用摄像头！")
                return

    def populate_camera_list(self):
        """检测可用摄像头并填充到下拉菜单"""
        index = 0
        available_cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:  # 如果无法读取帧，说明没有更多摄像头
                break
            available_cameras.append(f"摄像头 {index}")
            cap.release()
            index += 1

        if available_cameras:
            self.camera_combo.addItems(available_cameras)
        else:
            self.camera_combo.addItem("未检测到摄像头")
            self.log_message("未检测到可用摄像头，请检查设备连接。")

    def init_model(self):
        try:
            self.model = YOLO(self.config["model"])
            if self.config["use_gpu"]:
                self.model.to('cuda')
            self.model.fuse = lambda verbose=False: self.model
            self.log_message(f"已加载模型：{self.config['model']}")
            self.log_message("系统已启动")
        except Exception as e:
            self.log_message(f"模型加载失败：{str(e)}")

    def toggle_recording(self):
        if not self.video_thread or not self.video_thread.isRunning():
            # 弹出警告框提示用户
            QMessageBox.warning(self, "警告", "请先启动摄像头再开始录制！")
            return

        if not self.is_recording:
            # 创建video目录
            video_dir = "videos"
            os.makedirs(video_dir, exist_ok=True)

            # 初始化视频写入器
            frame_size = self.video_thread.frame_size
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            self.recording_path = os.path.join(video_dir, f"recording_{timestamp}.avi")
            self.recording_writer = cv2.VideoWriter(
                self.recording_path,
                fourcc,
                20.0,
                frame_size
            )
            self.is_recording = True
            self.recording_start_time = time.time()
            self.recording_timer.start(1000)  # 每秒更新一次
            self.record_btn.setText("已录制 00:00:00")
            self.log_message(f"开始录制: {self.recording_path}")
        else:
            # 停止录制
            if self.recording_writer is not None:
                self.recording_writer.release()
                self.recording_writer = None
            self.is_recording = False
            self.recording_timer.stop()
            self.record_btn.setText("开始录制")
            self.log_message(f"录制已保存: {self.recording_path}")
            # 弹出录制完成的消息框
            QMessageBox.information(self, "录制成功", f"录像已成功导出到:\n{self.recording_path}")

    def update_recording_time(self):
        if self.recording_start_time is not None:
            elapsed_time = int(time.time() - self.recording_start_time)
            hours = elapsed_time // 3600
            minutes = (elapsed_time % 3600) // 60
            seconds = elapsed_time % 60
            time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            self.record_btn.setText(f"已录制 {time_str}")

    def process_frame_for_recording(self, qimage):
        if self.is_recording and self.recording_writer is not None:
            # 确保 QImage 的格式为 Format_RGB888
            if qimage.format() != QImage.Format.Format_RGB888:
                qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)

            # 转换QImage回OpenCV格式
            ptr = qimage.bits()

            arr = np.array(ptr, dtype=np.uint8).reshape(
                qimage.height(), qimage.width(), 3  # RGBA
            )
            rgb_frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            self.recording_writer.write(rgb_frame)

    def toggle_camera(self):
        if not self.video_thread or not self.video_thread.isRunning():
            # 弹出警告框提示用户
            QMessageBox.warning(self, "警告", "请先启动摄像头再开始检测！")
            return

        if self.video_thread.detect_mode:
            # 如果正在进行检测，则停止检测
            self.update_target_counter(0)  # 重置目标数量为零
            self.video_thread.detect_mode = False
            self.start_btn.setText("开始检测")
            self.log_message("检测已停止")
            self.update_target_counter(0)  # 重置目标数量为零
            self.status_label.setText("状态: 运行中（未抓拍）")
        else:
            # 如果未进行检测，则开始检测
            self.update_target_counter(0)  # 重置目标数量为零
            self.status_label.setText("状态: 运行中")
            self.video_thread.detect_mode = True
            self.start_btn.setText("停止检测")
            self.log_message("开始检测")


    def start_camera(self, camera_id):
        """根据用户选择启动摄像头"""
        self.start_camera_with_id(camera_id)
    def update_target_counter(self, count):
        self.target_count_label.setText(f"目标数量: {count}")
    def start_camera_with_id(self, camera_id):
        """启动指定ID的摄像头"""
        self.video_thread = VideoThread(camera_id, self.model, self.config)
        self.video_thread.update_frame.connect(self.update_video)
        self.video_thread.update_fps.connect(self.update_fps_display)
        self.video_thread.update_target_count.connect(self.update_target_counter)
        self.video_thread.log_signal.connect(self.handle_log_message)
        self.video_thread.start()

        self.status_label.setText("状态: 运行中")
        self.start_btn.setText("停止监测")
        self.update_target_counter(0)  # 初始化目标数量为零
    def handle_log_message(self, message):
        self.log_message(message)

    def update_target_counter(self, count):
        self.target_count_label.setText(f"目标数量: {count}")
    def stop_camera(self):
        if self.video_thread:
            self.video_thread.stop()
        self.video_label.clear()
        self.status_label.setText("状态: 已停止")
        self.start_btn.setText("开始检测")
        self.update_target_counter(0)  # 重置目标数量为零

    def update_video(self, image):
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        # 处理录制帧
        if self.is_recording:
            self.process_frame_for_recording(image)

    def update_target_classes(self, text):
        self.config["target_classes"] = [text]
        self.log_message(f"更新目标类别为: {self.config['target_classes']}")

    def update_fps_display(self, fps):
        self.fps_label.setText(f"FPS: {fps}")



    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        # 写入日志文件
        log_file_path = os.path.join(self.log_dir, f"log_{datetime.now().strftime('%Y%m%d')}.txt")
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(log_entry)

        # 显示在QTextEdit中
        self.log_edit.append(log_entry)
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)

    def show_history(self):
        if self.history_container.isVisible():
            if self.history_container.isMinimized():
                # 如果 history_container 处于最小化状态，则恢复到正常显示状态
                self.history_container.showNormal()
            # 如果 history_container 已经显示，则将其提升到最前端
            self.history_container.raise_()
            self.history_container.activateWindow()
        else:
            self.history_window.clear()  # 清空历史记录窗口内容
            save_dir = self.config["save_dir"]
            if os.path.exists(save_dir):
                for filename in os.listdir(save_dir):
                    if filename.endswith(".jpg"):
                        file_path = os.path.join(save_dir, filename)
                        # 获取文件的创建时间或最后修改时间
                        try:
                            file_creation_time = os.path.getctime(file_path)
                        except AttributeError:
                            file_creation_time = os.path.getmtime(file_path)

                        # 将时间格式化为字符串
                        formatted_time = datetime.fromtimestamp(file_creation_time).strftime("%Y-%m-%d %H:%M:%S")
                        # 将格式化的时间与文件名一起添加到 history_window 中
                        self.history_window.append(f"{formatted_time} 检测到: {filename}")

            # 获取主窗口的几何信息
            main_window_geometry = self.geometry()
            main_window_center = main_window_geometry.center()

            # 获取 history_container 的宽度和高度
            history_container_width = self.history_container.width()
            history_container_height = self.history_container.height()

            # 计算 history_container 的位置，使其位于主窗口的中心
            x = main_window_center.x() - history_container_width // 2
            y = main_window_center.y() - history_container_height // 2

            # 设置 history_container 的位置
            self.history_container.move(x, y)

            # 创建一个垂直布局管理器
            layout = QVBoxLayout()
            self.history_container.setLayout(layout)

            # 创建一个水平布局管理器来放置时间范围选择组件和检测类型下拉组件
            top_layout = QHBoxLayout()

            # 添加检测类型下拉组件
            top_layout.addWidget(self.history_class_name)
            top_layout.addWidget(self.history_class_combo)

            # 添加时间范围选择组件
            top_layout.addWidget(self.history_start_time_name)
            top_layout.addWidget(self.start_time_edit)
            top_layout.addWidget(self.history_end_time_name)
            top_layout.addWidget(self.end_time_edit)

            # 添加导出按钮
            top_layout.addWidget(self.report_type_name)
            top_layout.addWidget(self.report_type_combo)

            # 将水平布局添加到垂直布局中
            layout.addLayout(top_layout)

            second_layout = QHBoxLayout()
            second_layout.addWidget(self.export_btn)
            second_layout.addWidget(self.generate_report_btn)
            layout.addLayout(second_layout)

            # 将 QTextEdit 添加到布局中
            layout.addWidget(self.history_window)

            self.history_container.show()

    def export_history(self):
        # 获取选择的时间范围
        start_time = self.start_time_edit.dateTime().toString("yyyy-MM-dd")
        end_time = self.end_time_edit.dateTime().toString("yyyy-MM-dd")
        start_time = datetime.strptime(start_time, "%Y-%m-%d")
        end_time = datetime.strptime(end_time, "%Y-%m-%d")

        # 获取选择的检测类型
        selected_class = self.class_combo.currentText()

        # 创建 history 文件夹
        history_dir = os.path.join(os.getcwd(), "history")
        os.makedirs(history_dir, exist_ok=True)

        # 获取当前时间作为文件名的一部分
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file_path = os.path.join(history_dir, f"history_{timestamp}.txt")

        # 获取 history_window 中的内容
        history_content = self.history_window.toPlainText()

        # 过滤历史记录
        filtered_content = []
        for line in history_content.split('\n'):
            if not line:
                continue
            parts = line.split(' 检测到: ')
            if len(parts) != 2:
                continue
            detection_time_str, filename = parts
            detection_time_str = detection_time_str.split(' ')[0]
            detection_time = datetime.strptime(detection_time_str, "%Y-%m-%d")
            if (not selected_class or selected_class in filename) and start_time <= detection_time <= end_time:
                filtered_content.append(line)

        # 将过滤后的内容写入文件
        with open(export_file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(filtered_content))

        self.log_message(f"历史记录已导出到: {export_file_path}")

        # 弹出导出成功的消息框
        QMessageBox.information(self, "导出成功", f"历史记录已成功导出到:\n{export_file_path}")
    def select_save_dir(self):
        initial_dir = self.config["save_dir"]  # 获取当前保存路径作为初始目录
        dir_path = QFileDialog.getExistingDirectory(self, "选择监测目标保存路径", initial_dir)
        if dir_path:
            self.save_dir_edit.setText(dir_path)
            self.config["save_dir"] = dir_path

    def update_conf_thres(self, value):
        self.config["conf_thres"] = value / 100.0
        self.conf_value.setText(f"置信度阈值: {self.config['conf_thres']:.2f}")  # 显示两位小数

    def load_config(self):
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                return json.load(f)
        return DEFAULT_CONFIG

    def closeEvent(self, event):
        # 停止摄像头
        self.stop_camera()
        self.log_message("系统已关闭")

        # 保存配置
        with open('config.json', 'w') as f:
            json.dump(self.config, f)

        # 关闭 history_container 窗口
        if self.history_container.isVisible():
            self.history_container.close()

        # 接受关闭事件
        event.accept()

    def view_log_files(self):
        log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.txt')]

        if not log_files:
            self.log_message("没有日志文件。")
            return

        log_file_dialog = QFileDialog()
        log_file_dialog.setWindowTitle("选择日志文件")
        log_file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        log_file_dialog.setNameFilter("Text Files (*.txt)")
        log_file_dialog.setDirectory(self.log_dir)

        if log_file_dialog.exec():
            selected_file = log_file_dialog.selectedFiles()[0]
            try:
                # 使用系统默认文本编辑器打开文件
                os.startfile(selected_file)
                self.log_message(f"已打开日志文件: {selected_file}")
            except Exception as e:
                self.log_message(f"打开日志文件失败: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_icon = QIcon("icon/logomax.png")
    app.setWindowIcon(app_icon)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
