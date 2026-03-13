"""
라벨링 도구 GUI 버전
PyQt5 기반 그래픽 인터페이스
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
    QProgressBar, QGroupBox, QSpinBox, QCheckBox, QRadioButton,
    QSlider, QMessageBox, QSplitter, QScrollArea, QStackedWidget,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QFont, QTextCursor, QImage, QPixmap, QCursor, QPainter, QPen, QColor
import io
import os
import cv2
import numpy as np
from label_tool import SimpleLabelTool

# LogStream 클래스 제거됨 - 직접 로그 출력 방식으로 변경


class LabelingWorker(QThread):
    """백그라운드에서 라벨링 실행"""
    progress = pyqtSignal(int, int)  # (current, total)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, input_folder, output_folder,model_path, config):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_path = model_path
        self.config = config
        self.is_running = True
    
    def run(self):
        """라벨링 실행 (subprocess로 label_tool.py 호출)"""
        try:
            import subprocess
            import sys
            
            # 출력 폴더 생성 (사용자 지정 또는 날짜-시간)
            output_name = self.config.get('output_name', '').strip()
            
            # 폴더 이름이 비어있으면 기본값 사용
            if not output_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_name = f"labels_{timestamp}"
            
            output_base = Path(self.output_folder) / output_name
            
            # 폴더 생성 (이미 있으면 그대로 사용)
            if output_base.exists():
                print(f"📁 기존 폴더 사용: {output_base}")
            else:
                output_base.mkdir(parents=True, exist_ok=True)
                print(f"📁 새 폴더 생성: {output_base}")
            
            # 이미지 파일 검색
            input_path = Path(self.input_folder)
            image_files = list(input_path.glob("*.jpg")) + \
                         list(input_path.glob("*.png")) + \
                         list(input_path.glob("*.jpeg"))
            
            total = len(image_files)
            print(f"📷 총 이미지: {total}개")
            
            if total == 0:
                raise Exception(f"이미지가 없습니다: {self.input_folder}")
            

            model_path = Path(self.model_path)
            print(f"모델 경로: {model_path}")
            if not model_path.exists():
                raise Exception(f"모델이 없습니다: {self.model_path}")
            # 입력 폴더 경로를 환경변수로 전달 (복사 없이 직접 사용) ⭐
            os.environ['LABEL_TOOL_INPUT_FOLDER'] = str(input_path)
            print(f"📂 입력 폴더 설정: {input_path}")
            print(f"✅ 이미지 복사 생략: 직접 경로 사용")
            
            # 출력 폴더 확인 및 progress_log.json 처리
            import json
            
            # 환경변수로 출력 폴더 경로 전달 ⭐
            os.environ['LABEL_TOOL_OUTPUT_FOLDER'] = str(output_base)
            print(f"📁 작업 폴더 설정: {output_base}")
            
            output_base.mkdir(parents=True, exist_ok=True)
            log_file = output_base / "progress_log.json"
            
            # 기존 로그 확인
            if log_file.exists():
                print(f"⚠️ 기존 작업 로그 발견: {log_file}")
                print(f"   → 이어하기 모드로 진행합니다.")
                # 로그 읽어서 정보 확인
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        existing_log = json.load(f)
                    existing_labeled = len(existing_log.get('labeled', []))
                    existing_skipped = len(existing_log.get('skipped', []))
                    existing_total = existing_log.get('total_images', total)
                    existing_not_processed = existing_total - existing_labeled - existing_skipped
                    
                    print(f"   기존: 라벨링 {existing_labeled}개, 스킵 {existing_skipped}개, 미처리 {existing_not_processed}개")
                    
                    # 환경변수로 기존 작업 정보 전달 (GUI 업데이트용) ⭐
                    os.environ['LABEL_TOOL_RESUME_LABELED'] = str(existing_labeled)
                    os.environ['LABEL_TOOL_RESUME_SKIPPED'] = str(existing_skipped)
                    os.environ['LABEL_TOOL_RESUME_TOTAL'] = str(total)
                    
                    # total_images 업데이트 (이미지 개수가 바뀌었을 수 있음)
                    existing_log['total_images'] = total
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_log, f, indent=2, ensure_ascii=False)
                except:
                    # 로그 읽기 실패 시 새로 생성
                    print(f"   로그 읽기 실패, 초기화합니다.")
                    initial_log = {
                        'input_folder': str(input_path),
                        'output_folder': str(output_base),
                        'total_images': total,
                        'labeled': [],
                        'skipped': [],
                        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(initial_log, f, indent=2, ensure_ascii=False)
            else:
                # 신규 로그 생성
                print(f"📋 신규 작업 시작: progress_log.json 생성")
                initial_log = {
                    'input_folder': str(input_path),
                    'output_folder': str(output_base),
                    'total_images': total,
                    'labeled': [],
                    'skipped': [],
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(initial_log, f, indent=2, ensure_ascii=False)
                print(f"   전체: {total}개")
            
            print(f"\n⚠️ 현재는 인터랙티브 모드입니다")
            print(f"💡 label_tool.py가 별도 창에서 열립니다")
            print(f"📌 각 이미지를 수동으로 라벨링해주세요:")
            print(f"   1. SAM 모드에서 클릭")
            print(f"   2. SPACE로 저장")
            print(f"   3. 다음 이미지로 자동 이동\n")
            
            # 환경변수 설정 (중요!)
            os.environ['LABEL_TOOL_INPUT_FOLDER'] = str(input_path)
            os.environ['LABEL_TOOL_OUTPUT_FOLDER'] = str(output_base)
            
            os.environ['LABEL_TOOL_MODEL'] = str(model_path)
            # # label_tool.py 실행 경로 설정
            # if getattr(sys, 'frozen', False):
            #     # PyInstaller로 빌드된 경우
            #     label_tool_path = os.path.join(os.path.dirname(sys.executable), "label_tool.py")
            # else:
            #     # 개발 환경
            #     label_tool_path = "label_tool.py"

            if getattr(sys, 'frozen', False):
                # 패키징된 경우: 같은 폴더 내 label_tool_core.exe 실행
                if os.path.exists("_internal/label_tool_core.exe"):
                    cmd = ["_internal/label_tool_core.exe"]
                else:
                    label_tool_path = os.path.join(os.path.dirname(sys.executable), "label_tool_core.exe")
                    cmd = [label_tool_path]
            else:
                # 개발 환경: Python으로 실행
                label_tool_path = "label_tool.py"
                cmd = [sys.executable, label_tool_path]

            
            print(f"🔧 환경변수 설정:")
            print(f"   INPUT: {os.environ.get('LABEL_TOOL_INPUT_FOLDER')}")
            print(f"   OUTPUT: {os.environ.get('LABEL_TOOL_OUTPUT_FOLDER')}")
            print(f"   명령어: {' '.join(cmd)}")
            
            # 프로세스 시작
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 출력 실시간 표시
            output_lines = []
            for line in process.stdout:
                if not self.is_running:
                    process.terminate()
                    print("⏸️ 사용자가 중단함")
                    break
                output_lines.append(line.strip())
                print(line.strip())
            
            # 프로세스 종료 대기
            process.wait()
            
            if process.returncode == 0:
                print(f"\n✅ 라벨링 완료!")
                print(f"📁 결과: {output_base}")
                self.finished.emit()
            else:
                # 더 자세한 에러 정보
                error_msg = f"label_tool.py 실행 실패 (exit code: {process.returncode})"
                if output_lines:
                    error_msg += f"\n\n마지막 출력:\n" + "\n".join(output_lines[-10:])
                raise Exception(error_msg)
            
            # 복사 과정이 없으므로 삭제 로직도 불필요 ⭐
            print(f"✅ 라벨링 완료: 원본 폴더 안전")
            
        except Exception as e:
            error_msg = f"❌ 에러: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)
    
    def stop(self):
        """중단"""
        self.is_running = False


class LabelingWidget(QWidget):
    """PyQt5 기반 라벨링 위젯"""
    image_saved = pyqtSignal(str)  # 이미지 저장 완료 (이미지 경로)
    image_skipped = pyqtSignal(str)  # 이미지 스킵 (이미지 경로)
    
    def __del__(self):
        """소멸자 - 메모리 정리"""
        try:
            if hasattr(self, 'label_tool') and self.label_tool is not None:
                del self.label_tool
            
            # CUDA 메모리 정리
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 LabelingWidget CUDA 메모리 정리 완료")
        except Exception as e:
            print(f"⚠️ LabelingWidget 메모리 정리 중 오류: {e}")
    
    def __init__(self, checkpoint_path, config=None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.config = config or {}
        
        # SimpleLabelTool 인스턴스 (지연 로딩)
        self.label_tool = None
        
        # 원본 이미지 크기 저장
        self.original_img_size = None  # (width, height)
        # 표시 이미지 크기 저장
        self.display_img_size = None  # (width, height)
        
        # 드래그 상태
        self.is_drawing = False
        self.last_point = None
        
        # 현재 이미지 경로
        self._current_image_path = None
        
        # 업데이트 타이머
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._periodic_update)
        self.update_timer.start(33)  # 30fps
        
        self.setup_ui()
        self.setFocusPolicy(Qt.StrongFocus)  # 키보드 포커스 받기
        
        # 마우스 트래킹 활성화 (커서 업데이트용)
        self.setMouseTracking(True)
        
        # 미리보기 모드 상태
        self._preview_mode = None
    
    def setup_ui(self):
        """UI 설정"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 이미지 표시용 QLabel
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)  # 중앙 정렬로 변경
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.image_label)
        
        self.setLayout(layout)
    
    def set_image(self, img_path):
        """이미지 설정 및 표시"""
        # SimpleLabelTool 지연 로딩 (첫 이미지 로드 시)
        if self.label_tool is None:
            print("🔄 SAM 모델 로딩 중...")
            self.label_tool = SimpleLabelTool(self.checkpoint_path, self.config)
            print("✅ SAM 모델 로딩 완료")
        
        self._current_image_path = img_path
        
        # 이미지 읽기
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            return False
        
        # 원본 이미지 크기 저장
        orig_h, orig_w = cv_img.shape[:2]
        self.original_img_size = (orig_w, orig_h)
        
        # 라벨링 영역 크기 (부모 위젯 크기 기준으로 계산)
        parent_widget = self.image_label.parent()
        if parent_widget:
            parent_widget.updateGeometry()
            QApplication.processEvents()
            
            # 부모 위젯 크기에서 여백 제외
            parent_w = parent_widget.width()
            parent_h = parent_widget.height()
            
            # 여백 고려 (약간의 패딩)
            label_area_w = max(parent_w - 20, 800)
            label_area_h = max(parent_h - 20, 600)
        else:
            # 전체 화면 크기 기준
            screen = QApplication.desktop().screenGeometry()
            label_area_w = screen.width() - 450  # 사이드바 400px + 여백 50px
            label_area_h = screen.height() - 100  # 상하 여백
        
        print(f"🖼️ 라벨링 영역 크기: {label_area_w} x {label_area_h}")
        
        # 비율 유지하며 라벨링 영역에 맞추도록 스케일 계산
        scale_w = label_area_w / orig_w
        scale_h = label_area_h / orig_h
        scale = min(scale_w, scale_h)  # 비율 유지하며 라벨링 영역 안에 맞춤
        
        # 표시 크기 계산
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)
        self.display_img_size = (disp_w, disp_h)
        
        # 이미지 리사이즈
        resized_img = cv2.resize(cv_img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        
        # 검은색 배경 생성
        background = np.zeros((label_area_h, label_area_w, 3), dtype=np.uint8)
        self._background_size = (label_area_w, label_area_h)  # Background 크기 저장
        
        # 중앙 정렬을 위한 오프셋 계산
        offset_x = (label_area_w - disp_w) // 2
        offset_y = (label_area_h - disp_h) // 2
        
        print(f"📐 이미지 배치: 원본({orig_w}x{orig_h}) → 표시({disp_w}x{disp_h}) @ 오프셋({offset_x}, {offset_y})")
        print(f"📏 라벨링 영역: {label_area_w} x {label_area_h}, 여백: 좌우({offset_x}), 상하({offset_y})")
        print(f"🔍 QLabel 실제 크기: {self.image_label.width()} x {self.image_label.height()}")
        print(f"🖼️ Background 이미지 크기: {background.shape[1]} x {background.shape[0]}")
        
        # 이미지를 중앙에 배치 (비율 유지로 인해 항상 라벨링 영역 안에 맞음)
        background[offset_y:offset_y+disp_h, offset_x:offset_x+disp_w] = resized_img
        self.image_offset = (offset_x, offset_y)
        self.display_img_size = (disp_w, disp_h)  # 표시 이미지 크기 저장
        
        # SimpleLabelTool에 이미지 설정
        self.label_tool.current_img = cv_img
        self.label_tool.display_img = background.copy()
        self.label_tool.predictor.set_image(cv_img)
        
        # QLabel 크기를 Background 이미지 크기와 정확히 맞춤
        self.image_label.setFixedSize(label_area_w, label_area_h)
        print(f"🎯 QLabel 크기 설정: {label_area_w} x {label_area_h}")
        
        # 초기화
        self.label_tool.points = []
        self.label_tool.labels = []
        self.label_tool.negative_mask = None
        self.label_tool.manual_edited = False
        self.label_tool.current_mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
        self.label_tool.mode = 'sam'
        self.label_tool.is_drawing = False
        self.label_tool.split_mode = 'entire'
        self.label_tool.split_masks = {'left': None, 'right': None}
        
        # 커서 초기화
        self.update_cursor()
        
        # QPixmap으로 변환 및 표시
        self.update_display()
        
        # 라벨링 위젯이 포커스를 받도록 설정 (키보드 이벤트 처리용)
        QTimer.singleShot(100, self.setFocus)  # 약간의 지연 후 포커스 설정
        
        return True
    
    def on_splitter_moved(self, pos, index):
        """Splitter 조절 시 이미지 다시 로드"""
        if hasattr(self, '_current_image_path') and self._current_image_path:
            # 약간의 지연 후 이미지 다시 로드 (UI 업데이트 완료 후)
            QTimer.singleShot(100, lambda: self.load_image(self._current_image_path))
    
    def widget_to_image_coords(self, widget_x, widget_y):
        """위젯 좌표 → 이미지 픽셀 좌표 변환 (중앙 정렬 고려)"""
        if self.original_img_size is None or self.display_img_size is None:
            return None, None
        
        # 중앙 정렬 오프셋 계산
        label_area_w = self.image_label.width() if self.image_label.width() > 0 else 960
        label_area_h = self.image_label.height() if self.image_label.height() > 0 else 960
        disp_w, disp_h = self.display_img_size
        
        offset_x = (label_area_w - disp_w) // 2
        offset_y = (label_area_h - disp_h) // 2
        
        # 실제 이미지 영역 내 좌표로 변환
        image_area_x = widget_x - offset_x
        image_area_y = widget_y - offset_y
        
        orig_w, orig_h = self.original_img_size
        disp_w, disp_h = self.display_img_size
        
        # 이미지 영역 밖이면 None 반환
        if image_area_x < 0 or image_area_y < 0 or image_area_x >= disp_w or image_area_y >= disp_h:
            return None, None
        
        # 스케일 비율 계산
        scale_x = orig_w / disp_w
        scale_y = orig_h / disp_h
        
        # 이미지 픽셀 좌표로 변환
        image_x = int(image_area_x * scale_x)
        image_y = int(image_area_y * scale_y)
        
        # 범위 체크
        image_x = max(0, min(image_x, orig_w - 1))
        image_y = max(0, min(image_y, orig_h - 1))
        
        return image_x, image_y
    
    def _cv2_to_qpixmap(self, cv_img):
        """OpenCV 이미지 (numpy array, BGR) → QPixmap (RGB) 변환"""
        # BGR → RGB 변환
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # numpy array → QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # QImage → QPixmap
        return QPixmap.fromImage(qt_image)
    
    def update_display(self):
        """화면 업데이트"""
        if self.label_tool is None or self.label_tool.current_img is None or self.display_img_size is None:
            return
        
        # 미리보기 모드가 아닐 때만 원본 이미지로 초기화
        if not hasattr(self, '_preview_mode') or self._preview_mode is None:
            # 원본 이미지 크기로 합성 (SimpleLabelTool의 _update_display 사용)
            # 임시로 display_img를 원본 이미지로 설정
            self.label_tool.display_img = self.label_tool.current_img.copy()
            
            # OpenCV 합성 수행
            self.label_tool._update_display()
        # 미리보기 모드일 때는 이미 _generate_*_preview_gui()에서 display_img가 설정됨
        
        # 합성된 이미지를 리사이즈
        orig_h, orig_w = self.label_tool.current_img.shape[:2]
        disp_w, disp_h = self.display_img_size
        resized_display = cv2.resize(self.label_tool.display_img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        
        # 배경 이미지 생성 (라벨링 영역 크기)
        label_area_w = self.image_label.width() if self.image_label.width() > 0 else 960
        label_area_h = self.image_label.height() if self.image_label.height() > 0 else 960
        background = np.zeros((label_area_h, label_area_w, 3), dtype=np.uint8)
        
        # 중앙에 리사이즈된 이미지 배치
        offset_x = (label_area_w - disp_w) // 2
        offset_y = (label_area_h - disp_h) // 2
        background[offset_y:offset_y+disp_h, offset_x:offset_x+disp_w] = resized_display
        
        # OpenCV 이미지 → QPixmap 변환
        pixmap = self._cv2_to_qpixmap(background)
        self.image_label.setPixmap(pixmap)
    
    def _periodic_update(self):
        """주기적 업데이트 (30fps)"""
        if self.label_tool and self.label_tool.display_img is not None:
            self.update_display()
    
    def create_cross_cursor(self):
        """SAM 모드용 십자 커서 생성 (정확한 중앙 핫스팟)"""
        cursor_size = 32
        line_length = 12
        
        # 투명 배경의 픽스맵 생성
        pixmap = QPixmap(cursor_size, cursor_size)
        pixmap.fill(Qt.transparent)
        
        # 페인터로 십자 그리기
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 검은색 십자 (두께 2)
        pen = QPen(QColor(0, 0, 0), 2)
        painter.setPen(pen)
        
        center = cursor_size // 2
        
        # 세로선
        painter.drawLine(center, center - line_length // 2, center, center + line_length // 2)
        # 가로선
        painter.drawLine(center - line_length // 2, center, center + line_length // 2, center)
        
        # 중앙점 (더 정확한 클릭 위치 표시)
        painter.drawEllipse(center - 1, center - 1, 2, 2)
        
        painter.end()
        
        # 핫스팟을 정확히 중앙으로 설정
        return QCursor(pixmap, center, center)
    
    def create_brush_cursor(self, size, color):
        """그리기 모드용 원형 커서 생성"""

        # cv2.circle에서 size는 반지름, 실제 지름은 size*2
        # 커서도 실제 그려지는 지름과 맞춤
        actual_size = size  # 커서 반지름 = 브러시 반지름
        cursor_size = max(actual_size * 2 + 8, 32)  # 픽스맵 크기
        
        # 투명 배경의 픽스맵 생성
        pixmap = QPixmap(cursor_size, cursor_size)
        pixmap.fill(Qt.transparent)
        
        # 페인터로 원 그리기
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 외곽선 없음
        painter.setPen(Qt.NoPen)
        
        # 정의된 오버레이 색상 + 완전 불투명
        brush_color = QColor(color[2], color[1], color[0], 255)  # BGR → RGB, alpha=255 (완전 불투명)
        painter.setBrush(brush_color)
        
        # 중앙에 원 그리기 (반지름으로 찍히는 원과 커서 동심원정렬)
        center = cursor_size // 2
        painter.drawEllipse(center - actual_size, center - actual_size, actual_size * 2, actual_size * 2)
        
        painter.end()
        
        # 핫스팟을 중앙으로 설정
        return QCursor(pixmap, center, center)
    
    def create_eraser_cursor(self, size):
        """지우기 모드용 원형 커서 생성 (흰색 고정)"""

        # cv2.circle에서 size는 반지름, 실제 지름은 size*2
        # 커서도 실제 그려지는 지름과 맞춤
        actual_size = size  # 커서 반지름 = 브러시 반지름 
        cursor_size = max(actual_size * 2 + 8, 32)  # 픽스맵 크기
        
        # 투명 배경의 픽스맵 생성
        pixmap = QPixmap(cursor_size, cursor_size)
        pixmap.fill(Qt.transparent)
        
        # 페인터로 원 그리기
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 외곽선 없음
        painter.setPen(Qt.NoPen)
        
        # 흰색 + 완전 불투명
        eraser_color = QColor(255, 255, 255, 255)  # alpha=255 (완전 불투명)
        painter.setBrush(eraser_color)
        
        # 중앙에 원 그리기 (브러시 크기와 동일)
        center = cursor_size // 2
        painter.drawEllipse(center - actual_size, center - actual_size, actual_size * 2, actual_size * 2)
        
        painter.end()
        
        # 핫스팟을 중앙으로 설정
        return QCursor(pixmap, center, center)
    
    def update_cursor(self):
        """현재 모드에 따라 커서 업데이트"""
        if self.label_tool is None:
            self.setCursor(Qt.ArrowCursor)
            return
        
        if self.label_tool.mode == 'sam':
            # SAM 모드: 십자 커서
            cursor = self.create_cross_cursor()
            self.setCursor(cursor)
        elif self.label_tool.mode == 'draw':
            # 그리기 모드: 현재 오버레이 색상의 원형 커서 (스케일 적용)
            color_name = self.label_tool.color_names[self.label_tool.current_color_idx]
            color = self.label_tool.overlay_colors[color_name]
            
            # 현재 이미지 스케일에 맞게 브러시 크기 조정
            scaled_brush_size = self._get_scaled_brush_size()
            cursor = self.create_brush_cursor(scaled_brush_size, color)
            self.setCursor(cursor)
        elif self.label_tool.mode == 'erase':
            # 지우기 모드: 흰색 원형 커서 (스케일 적용)
            scaled_brush_size = self._get_scaled_brush_size()
            cursor = self.create_eraser_cursor(scaled_brush_size)
            self.setCursor(cursor)
        else:
            # 기본 커서
            self.setCursor(Qt.ArrowCursor)
    
    def _get_scaled_brush_size(self):
        """현재 이미지 스케일에 맞게 조정된 브러시 크기 반환"""
        if (self.original_img_size is None or 
            self.display_img_size is None or 
            self.label_tool is None):
            return self.label_tool.brush_size if self.label_tool else 5
        
        # 현재 스케일 비율 계산
        orig_w, orig_h = self.original_img_size
        disp_w, disp_h = self.display_img_size
        
        # 가로/세로 중 더 작은 스케일 사용 (이미지 비율 유지와 동일)
        scale_x = disp_w / orig_w
        scale_y = disp_h / orig_h
        scale = min(scale_x, scale_y)
        
        # 브러시 크기에 스케일 적용
        scaled_size = int(self.label_tool.brush_size * scale)
        
        # 최소 크기 보장 (너무 작으면 보이지 않음)
        return max(scaled_size, 2)
    
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트"""
        if self.label_tool is None:
            return
        
        # 위젯 좌표 (LabelingWidget 기준)
        widget_x = event.x()
        widget_y = event.y()
        
        # QLabel의 위치를 고려한 실제 좌표 계산
        label_pos = self.image_label.pos()
        actual_x = widget_x - label_pos.x()
        actual_y = widget_y - label_pos.y()
        
        # 이미지 픽셀 좌표로 변환 (실제 좌표 사용)
        image_x, image_y = self.widget_to_image_coords(actual_x, actual_y)
        if image_x is None or image_y is None:
            return  # 여백 영역 클릭 무시
        
        # SimpleLabelTool에 전달
        if self.label_tool.mode == 'sam':
            if event.button() == Qt.LeftButton:
                self.label_tool.points.append([image_x, image_y])
                self.label_tool.labels.append(1)
                self.label_tool._update_with_sam()
            elif event.button() == Qt.RightButton:
                self.label_tool.points.append([image_x, image_y])
                self.label_tool.labels.append(0)
                self.label_tool._update_with_sam()
        elif self.label_tool.mode in ['draw', 'erase']:
            self.is_drawing = True
            self.last_point = (widget_x, widget_y)
            self.label_tool._draw_at(image_x, image_y)
        
        self.update_display()
    
    def keyPressEvent(self, event):
        """라벨링 위젯 키보드 이벤트 처리"""
        if self.label_tool is None:
            super().keyPressEvent(event)
            return
        
        key = event.key()
        modifiers = event.modifiers()
        
        # PageUp 키로 텍스트 표시 토글
        if key == Qt.Key_PageUp:
            self.label_tool.show_text = not self.label_tool.show_text
            self.label_tool._update_display()
            self.update_display()
            event.accept()
            return
        
        # 라벨링 관련 키들
        elif key == Qt.Key_R:
            self.label_tool.points = []
            self.label_tool.labels = []
            self.label_tool.negative_mask = None
            self.label_tool.manual_edited = False
            self.label_tool.current_mask = np.zeros(self.label_tool.current_img.shape[:2], dtype=np.uint8)
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_U or (modifiers == Qt.ControlModifier and key == Qt.Key_Z):
            if len(self.label_tool.points) > 0:
                self.label_tool.points.pop()
                self.label_tool.labels.pop()
                if self.label_tool.mode == 'sam':
                    self.label_tool._update_with_sam()
                self.update_display()
        elif key == Qt.Key_1:
            self.label_tool.mode = 'sam'
            self.update_cursor()  # 커서 업데이트
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_2:
            self.label_tool.mode = 'draw'
            self.label_tool.manual_edited = True
            self.update_cursor()  # 커서 업데이트
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_3:
            self.label_tool.mode = 'erase'
            self.label_tool.manual_edited = True
            self.update_cursor()  # 커서 업데이트
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_V:
            self.label_tool._toggle_view_mode()
            self.update_display()
        elif key == Qt.Key_C:
            self.label_tool._change_overlay_color()
            self.update_cursor()  # 색상 변경 시 커서 업데이트
            self.update_display()
        elif key == Qt.Key_M:
            # M키: 마스크 미리보기 토글
            if self._preview_mode == 'mask':
                # 미리보기 해제 → 원본으로 복원
                self._preview_mode = None
                self.label_tool._update_display()
            else:
                # 미리보기 활성화
                self._preview_mode = 'mask'
                self.label_tool._generate_mask_preview_gui()
            self.update_cursor()  # 커서 동기화
            self.update_display()
        elif key == Qt.Key_N:
            # N키: 바이너리 마스크 미리보기 토글
            if self._preview_mode == 'binary':
                # 미리보기 해제 → 원본으로 복원
                self._preview_mode = None
                self.label_tool._update_display()
            else:
                # 미리보기 활성화
                self._preview_mode = 'binary'
                self.label_tool._generate_binary_preview_gui()
            self.update_cursor()  # 커서 동기화
            self.update_display()
        elif key == Qt.Key_S:
            # S키: 심플리파이 미리보기 토글
            if self._preview_mode == 'simplify':
                # 미리보기 해제 → 원본으로 복원
                self._preview_mode = None
                self.label_tool._update_display()
            else:
                # 미리보기 활성화
                self._preview_mode = 'simplify'
                self.label_tool._generate_simplify_preview_gui()
            self.update_cursor()  # 커서 동기화
            self.update_display()
        elif key == Qt.Key_BracketLeft:  # [
            self.label_tool.brush_size = max(1, self.label_tool.brush_size - 1)
            self.update_cursor()  # 브러시 크기 변경 시 커서 업데이트
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_BracketRight:  # ]
            self.label_tool.brush_size = min(100, self.label_tool.brush_size + 1)
            self.update_cursor()  # 브러시 크기 변경 시 커서 업데이트
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_F:
            self.label_tool.fill_holes_enabled = not self.label_tool.fill_holes_enabled
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_Minus or key == Qt.Key_Underscore:
            # S 미리보기 중이면 +/- 키 무시 (휠로만 조절)
            if self._preview_mode == 'simplify':
                return
            self.label_tool.max_hole_size = max(100, self.label_tool.max_hole_size - 200)
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_Equal or key == Qt.Key_Plus:
            # S 미리보기 중이면 +/- 키 무시 (휠로만 조절)
            if self._preview_mode == 'simplify':
                return
            self.label_tool.max_hole_size = min(10000, self.label_tool.max_hole_size + 200)
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_Q:
            if self.label_tool.split_mode == 'right' and np.any(self.label_tool.current_mask > 0):
                self.label_tool.split_masks['right'] = self.label_tool.current_mask.copy()
            self.label_tool.split_mode = 'left'
            if self.label_tool.split_masks['left'] is not None:
                self.label_tool.current_mask = self.label_tool.split_masks['left'].copy()
            else:
                self.label_tool.current_mask = np.zeros(self.label_tool.current_img.shape[:2], dtype=np.uint8)
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_W:
            if self.label_tool.split_mode == 'left' and np.any(self.label_tool.current_mask > 0):
                self.label_tool.split_masks['left'] = self.label_tool.current_mask.copy()
            self.label_tool.split_mode = 'right'
            if self.label_tool.split_masks['right'] is not None:
                self.label_tool.current_mask = self.label_tool.split_masks['right'].copy()
            else:
                self.label_tool.current_mask = np.zeros(self.label_tool.current_img.shape[:2], dtype=np.uint8)
            self.label_tool._update_display()
            self.update_display()
        elif key == Qt.Key_E:
            if self.label_tool.split_mode == 'left' and np.any(self.label_tool.current_mask > 0):
                self.label_tool.split_masks['left'] = self.label_tool.current_mask.copy()
            elif self.label_tool.split_mode == 'right' and np.any(self.label_tool.current_mask > 0):
                self.label_tool.split_masks['right'] = self.label_tool.current_mask.copy()
            self.label_tool.split_mode = 'entire'
            self.label_tool.current_mask = self.label_tool._merge_split_masks()
            self.label_tool._update_display()
            self.update_display()
        else:
            # 처리되지 않은 키는 부모로 전달
            super().keyPressEvent(event)
    
    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트 (드래그)"""
        if self.label_tool is None:
            return
        
        if self.is_drawing and self.label_tool.mode in ['draw', 'erase']:
            widget_x = event.x()
            widget_y = event.y()
            
            # QLabel의 위치를 고려한 실제 좌표 계산
            label_pos = self.image_label.pos()
            actual_x = widget_x - label_pos.x()
            actual_y = widget_y - label_pos.y()
            
            # 좌표 변환 (실제 좌표 사용)
            image_x, image_y = self.widget_to_image_coords(actual_x, actual_y)
            if image_x is None or image_y is None:
                return
            
            if self.last_point:
                # 이전 점도 QLabel 위치 고려해서 변환
                last_actual_x = self.last_point[0] - label_pos.x()
                last_actual_y = self.last_point[1] - label_pos.y()
                last_image_x, last_image_y = self.widget_to_image_coords(
                    last_actual_x, last_actual_y
                )
                if last_image_x is not None and last_image_y is not None:
                    self.label_tool._draw_line(
                        (last_image_x, last_image_y),
                        (image_x, image_y)
                    )
            else:
                self.label_tool._draw_at(image_x, image_y)
            
            self.last_point = (widget_x, widget_y)  # 원본 위젯 좌표로 저장
            self.update_display()
    
    def mouseReleaseEvent(self, event):
        """마우스 릴리즈 이벤트"""
        if self.is_drawing:
            self.is_drawing = False
            self.last_point = None
            self.update_display()
    
    def enterEvent(self, event):
        """마우스가 라벨링 영역에 들어올 때"""
        if self.label_tool is not None:
            self.update_cursor()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """마우스가 라벨링 영역을 벗어날 때"""
        self.setCursor(Qt.ArrowCursor)  # 기본 커서로 복원
        super().leaveEvent(event)
    
    def wheelEvent(self, event):
        """마우스 휠 이벤트"""
        if self.label_tool is None:
            return
        
        delta = event.angleDelta().y()
        
        # 미리보기 모드 최우선 체크
        if self._preview_mode == 'simplify':
            # Simplify 미리보기 중: 심플리파이 강도 조절
            if delta > 0:
                self.label_tool.simplify_strength = min(5, self.label_tool.simplify_strength + 1)
            else:
                self.label_tool.simplify_strength = max(0, self.label_tool.simplify_strength - 1)
            
            # 미리보기 다시 생성
            self.label_tool._generate_simplify_preview_gui()
            self.update_display()
            return
        
        # 기존 휠 기능 (미리보기가 아닐 때만)
        if self.label_tool.mode == 'sam':
            # SAM 모드: 제외 영역 확장 크기 조절
            if delta > 0:
                self.label_tool.negative_expand = min(50, self.label_tool.negative_expand + 3)
            else:
                self.label_tool.negative_expand = max(0, self.label_tool.negative_expand - 3)
            
            if len(self.label_tool.points) > 0:
                self.label_tool._update_with_sam()
        else:
            # Draw/Erase 모드: 브러시 크기 조절
            if delta > 0:
                self.label_tool.brush_size = min(100, self.label_tool.brush_size + 1)
            else:
                self.label_tool.brush_size = max(1, self.label_tool.brush_size - 1)
            
            # 브러시 크기 변경 시 커서도 업데이트
            self.update_cursor()
        
        self.label_tool._save_status()
        self.update_display()
    
    # 키 이벤트 처리는 LabelToolGUI로 이동됨
    
    def save_current_image(self, img_path, output_folder):
        """현재 이미지 저장"""
        if self.label_tool is None:
            return False
        
        # 현재 모드의 mask 저장
        if self.label_tool.split_mode == 'left' and np.any(self.label_tool.current_mask > 0):
            self.label_tool.split_masks['left'] = self.label_tool.current_mask.copy()
        elif self.label_tool.split_mode == 'right' and np.any(self.label_tool.current_mask > 0):
            self.label_tool.split_masks['right'] = self.label_tool.current_mask.copy()
        
        # 최종 mask: 좌우 합치기
        mask = self.label_tool._merge_split_masks()
        
        if not np.any(mask > 0):
            return False
        
        # 노이즈 제거
        mask = self.label_tool._remove_noise(mask, min_size=100)
        
        # 홀 채우기 적용
        if self.label_tool.fill_holes_enabled:
            mask = self.label_tool._fill_holes(mask, self.label_tool.max_hole_size)
        
        # 네거티브 마스크를 홀로 적용
        if self.label_tool.negative_mask is not None and not self.label_tool.manual_edited:
            mask[self.label_tool.negative_mask > 0] = 0
        
        # 다시 노이즈 제거
        mask = self.label_tool._remove_noise(mask, min_size=100)
        
        # Polygon 추출
        polygons = self.label_tool._mask_to_polygons(
            mask,
            min_area=500,
            fill_holes=False,
            max_hole_size=self.label_tool.max_hole_size,
            max_contours=10
        )
        
        # 저장
        from label_tool import save_labels
        save_labels(img_path, polygons, output_folder, mask)
        return True


class LabelToolGUI(QMainWindow):
    """메인 GUI 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.config_file = "config.json"
        
        # 창 최대화로 시작 (우분투 호환성 개선)
        try:
            self.showMaximized()
        except Exception as e:
            print(f"⚠️ 창 최대화 실패, 전체화면으로 대체: {e}")
            try:
                self.showFullScreen()
            except Exception as e2:
                print(f"⚠️ 전체화면도 실패, 기본 크기로 시작: {e2}")
                self.resize(1200, 800)
                self.show()
        self.status_file = "labeling_status.json"  # 상태 파일
        self.config = self.load_config()
        self.worker = None
        self.is_labeling_active = False
        self.current_work_folder = None
        self.labeling_widget = None  # LabelingWidget 인스턴스
        self.right_panel = None  # 오른쪽 패널
        self.right_panel_layout = None  # 오른쪽 패널 레이아웃
        self.image_files = []  # 이미지 파일 목록
        self.current_image_index = 0  # 현재 이미지 인덱스
        self.labeled_list = []  # 라벨링 완료된 이미지 목록
        self.skipped_list = []  # 스킵된 이미지 목록
        
        self.init_ui()
        self.setup_status_monitor()
    
    def keyPressEvent(self, event):
        """전역 키보드 이벤트 처리 - 전역 단축키만"""
        key = event.key()
        modifiers = event.modifiers()
        
        # 전역 단축키들 (텍스트 입력 중에도 작동해야 함)
        if key == Qt.Key_Escape:
            # ESC: 프로그램 종료 (항상 작동)
            QApplication.quit()
            event.accept()
            return
            
        elif key == Qt.Key_F1:
            # F1: 도움말 (항상 작동)
            if hasattr(self, 'help_toggle_btn'):
                self.help_toggle_btn.click()
            event.accept()
            return
        
        # 텍스트 입력 중이면 나머지 키는 무시
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, (QLineEdit, QTextEdit)):
            super().keyPressEvent(event)  # 기본 처리로 전달
            return
        
        # 라벨링 위젯이 없거나 label_tool이 없으면 무시
        if self.labeling_widget is None or not hasattr(self.labeling_widget, 'label_tool') or self.labeling_widget.label_tool is None:
            super().keyPressEvent(event)
            return
        
        # 라벨링 관련 전역 단축키들
        if key == Qt.Key_Space:
            # SPACE: 저장 + 다음 이미지
            if np.any(self.labeling_widget.label_tool.current_mask > 0):
                self.labeling_widget.image_saved.emit(str(self.labeling_widget._current_image_path))
            event.accept()
            return
            
        elif key == Qt.Key_AsciiTilde:  # 백틱(`)
            # 백틱: 이미지 스킵
            self.labeling_widget.image_skipped.emit(str(self.labeling_widget._current_image_path))
            event.accept()
            return
        
        # 나머지 키는 라벨링 위젯으로 전달
        else:
            # 라벨링 위젯이 포커스를 가지고 있으면 해당 위젯에서 처리
            if self.labeling_widget and self.labeling_widget.hasFocus():
                self.labeling_widget.keyPressEvent(event)
            else:
                super().keyPressEvent(event)
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("🏷️ Stall Labaling Tool For Ubuntu 1.0.4")
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # QSplitter 제거 - 간단한 고정 레이아웃 사용
        
        # 수평 레이아웃: 왼쪽(사이드바), 오른쪽(라벨링 영역)
        horizontal_layout = QHBoxLayout()
        horizontal_layout.setContentsMargins(0, 0, 0, 0)
        horizontal_layout.setSpacing(0)
        
        # 왼쪽: 사이드바 (400px 고정)
        sidebar_widget = self.create_sidebar_widget()
        horizontal_layout.addWidget(sidebar_widget)
        
        # 오른쪽: 라벨링 영역 (나머지 공간 모두 사용)
        self.right_panel = QWidget()
        # 라벨링 영역에 외곽선 추가
        self.right_panel.setStyleSheet("""
            QWidget {
                border: 1px solid #868686;
                background-color: white;
            }
        """)
        self.right_panel_layout = QVBoxLayout()
        self.right_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.right_panel.setLayout(self.right_panel_layout)
        
        # 초기에는 가이드 표시
        self.help_panel = self.create_help_panel()
        self.right_panel_layout.addWidget(self.help_panel)
        self.help_panel.setVisible(False)  # 초기에는 숨김
        
        horizontal_layout.addWidget(self.right_panel, 1)  # stretch factor 1로 나머지 공간 모두 사용
        
        main_layout.addLayout(horizontal_layout)
    
    # QSplitter 제거로 인해 더 이상 필요하지 않음
    # def on_splitter_moved(self, pos, index):
    
    def create_sidebar_widget(self):
        """사이드바 위젯 생성 (기존 메인 윈도우 구조)"""
        widget = QWidget()
        # 사이드바 너비 고정 및 외곽선 추가
        widget.setFixedWidth(400)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        widget.setLayout(layout)

        # 0. 모델 경로 그룹
        model_group = self.create_model_group()
        layout.addWidget(model_group)

        # 1. 폴더 선택 그룹
        folder_group = self.create_folder_group()
        layout.addWidget(folder_group)
        
        # 2. 설정 그룹
        self.settings_group = self.create_settings_group()
        layout.addWidget(self.settings_group)
        
        # 3. 진행 상황
        self.progress_group = self.create_progress_group()
        layout.addWidget(self.progress_group)
        
        # 4. 현재 상태 (라벨링 정보)
        self.status_group = self.create_status_group()
        layout.addWidget(self.status_group)
        
        # 5. 로그
        self.log_group = self.create_log_group()
        layout.addWidget(self.log_group, stretch=1)
        
        # 6. 버튼
        button_layout = self.create_button_layout()
        layout.addLayout(button_layout)
        
        return widget
    
    def create_model_group(self):
        """모델 경로 및 버튼 그룹"""
        group = QGroupBox("🤖 모델 및 작업")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout = QVBoxLayout()
        
        # 프로젝트 루트 기준으로 모델 경로 설정
        project_root = Path(__file__).parent.parent
        models_path = project_root / "models" / "sam2_hiera_large.pt"
        default_path = str(models_path)
        
        # models 폴더가 없으면 생성
        (project_root / "models").mkdir(parents=True, exist_ok=True)
        
        # config에서 모델 경로 가져오기 (없으면 기본값 사용)
        model_path_from_config = self.config.get('model_path', default_path)
        # 기본값이 존재하는지 확인하고, 없으면 기본값 사용
        if not Path(model_path_from_config).exists() and Path(default_path).exists():
            model_path_from_config = default_path
        
        # 모델 경로
        model_label = QLabel("모델 경로:")
        model_label.setFixedWidth(70)
        layout.addWidget(model_label)
        
        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit(model_path_from_config)
        self.model_path_edit.setMaximumWidth(250)  # 경로 입력창 최대 너비 제한
        model_browse_btn = QPushButton("찾기")
        model_browse_btn.setFixedWidth(50)  # 버튼 너비 고정
        model_browse_btn.setFocusPolicy(Qt.NoFocus)  # 키보드 포커스 차단
        model_browse_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(model_browse_btn)
        layout.addLayout(model_layout)
        
        # 버튼들 (세로로 배치)
        resume_btn = QPushButton("🔄 작업 이어하기")
        resume_btn.setToolTip("progress_log.json 파일을 선택하여\n이전 작업을 이어서 진행합니다.")
        resume_btn.setMaximumWidth(380)  # 사이드바 너비에 맞춤
        resume_btn.setFocusPolicy(Qt.NoFocus)  # 키보드 포커스 차단
        resume_btn.clicked.connect(self.resume_work)
        layout.addWidget(resume_btn)
        
        self.help_toggle_btn = QPushButton("❓ 사용 가이드")
        self.help_toggle_btn.setToolTip("사용 가이드 패널 열기/닫기")
        self.help_toggle_btn.setMaximumWidth(380)  # 사이드바 너비에 맞춤
        self.help_toggle_btn.setFocusPolicy(Qt.NoFocus)  # 키보드 포커스 차단
        self.help_toggle_btn.clicked.connect(self.toggle_help_panel)
        layout.addWidget(self.help_toggle_btn)
        
        group.setLayout(layout)
        return group
    



    # 파일 탐색 
    def browse_model_file(self):
        # 프로젝트 루트 기준으로 models 폴더 경로 설정
        project_root = Path(__file__).parent.parent
        models_path = project_root / "models"
        
        # models 폴더가 없으면 생성
        models_path.mkdir(parents=True, exist_ok=True)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "SAM 모델 파일 선택", 
            str(models_path),  # 초기 경로를 models 폴더로 설정
            "Model Files (*.pt *.pth)"
        )
        print(file_path)
        if file_path:
            self.model_path_edit.setText(file_path)


    


    # 모델
    def test_model_loading(self, model_path):
        """모델 로딩 테스트"""
        try:
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 모델 로딩 테스트 중...")
            #QApplication.processEvents()
            
            # 모델 파일 존재 확인
            if not os.path.exists(model_path):
                self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 모델 파일이 존재하지 않습니다: {model_path}")
                return False
            
            # 간단한 로딩 테스트 (실제 모델 로딩은 나중에)
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 모델 파일 확인됨: {os.path.basename(model_path)}")
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 💡 모델 로딩은 라벨링 시작 시 진행됩니다.")
            
            return True
            
        except Exception as e:
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 모델 로딩 테스트 실패: {str(e)}")
            return False





# 입출력 
    def create_folder_group(self):
        """폴더 선택 그룹"""
        group = QGroupBox("📁 폴더 설정")
        # 반응형으로 변경 - 고정 너비 제거
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout = QVBoxLayout()
        
        # 프로젝트 루트 기준으로 경로 설정
        project_root = Path(__file__).parent.parent
        datasets_path = project_root / "data_io" / "datasets"
        outputs_path = project_root / "data_io" / "outputs"
        
        # 폴더 생성 (없으면 생성)
        datasets_path.mkdir(parents=True, exist_ok=True)
        outputs_path.mkdir(parents=True, exist_ok=True)
        
        # 입력 폴더
        input_label = QLabel("입력 폴더:")
        input_label.setFixedWidth(70)
        layout.addWidget(input_label)
        
        default_input = str(datasets_path) if datasets_path.exists() else self.config.get('input_folder', 'source_image')
        self.input_folder_edit = QLineEdit(self.config.get('input_folder', default_input))
        self.input_folder_edit.setMaximumWidth(250)  # 경로 입력창 최대 너비 제한
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_folder_edit)
        self.input_browse_btn = QPushButton("찾기")
        self.input_browse_btn.setFixedWidth(50)  # 버튼 너비 고정
        self.input_browse_btn.setFocusPolicy(Qt.NoFocus)  # 키보드 포커스 차단
        self.input_browse_btn.clicked.connect(self.browse_input_folder)
        input_layout.addWidget(self.input_browse_btn)
        layout.addLayout(input_layout)
        
        # 출력 폴더
        output_label = QLabel("출력 폴더:")
        output_label.setFixedWidth(70)
        layout.addWidget(output_label)
        
        default_output = str(outputs_path) if outputs_path.exists() else self.config.get('output_folder', '.')
        self.output_folder_edit = QLineEdit(self.config.get('output_folder', default_output))
        self.output_folder_edit.setMaximumWidth(250)  # 경로 입력창 최대 너비 제한
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_folder_edit)
        self.output_browse_btn = QPushButton("찾기")
        self.output_browse_btn.setFixedWidth(50)  # 버튼 너비 고정
        self.output_browse_btn.setFocusPolicy(Qt.NoFocus)  # 키보드 포커스 차단
        self.output_browse_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(self.output_browse_btn)
        layout.addLayout(output_layout)
        
        # 입력 폴더 변경 시 출력 폴더명 자동 업데이트
        self.input_folder_edit.textChanged.connect(self._update_output_name_from_input)
        
        # 출력 폴더 이름
        name_label = QLabel("폴더 이름:")
        name_label.setFixedWidth(70)
        layout.addWidget(name_label)
        
        # 기본값: labels_날짜시간
        default_name = f"labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_name_edit = QLineEdit(self.config.get('output_name', default_name))
        self.output_name_edit.setPlaceholderText("예: labels_20250123 또는 my_labels")
        self.output_name_edit.setMaximumWidth(250)  # 입력창 최대 너비 제한
        
        output_name_layout = QHBoxLayout()
        output_name_layout.addWidget(self.output_name_edit)
        
        # 새로고침 버튼 (날짜시간 다시 생성)
        refresh_btn = QPushButton("🔄")
        refresh_btn.setToolTip("현재 날짜시간으로 다시 생성")
        refresh_btn.setFixedWidth(50)
        refresh_btn.setFocusPolicy(Qt.NoFocus)  # 키보드 포커스 차단
        refresh_btn.clicked.connect(self.refresh_output_name)
        output_name_layout.addWidget(refresh_btn)
        
        layout.addLayout(output_name_layout)
        
        # 안내 텍스트 (줄바꿈으로 조정)
        info_label = QLabel("💡 지정한 이름의 폴더가 생성됩니다\n(없으면 새로 생성, 있으면 추가 저장)")
        info_label.setStyleSheet("color: gray; font-size: 9px;")
        info_label.setWordWrap(True)  # 자동 줄바꿈
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        
        # 초기화 시 입력 폴더가 있으면 출력 폴더명 자동 설정
        if self.input_folder_edit.text():
            self._update_output_name_from_input()
        
        return group
    




    # 설정 세팅
    def create_settings_group(self):
        """설정 그룹"""
        group = QGroupBox("⚙️ 라벨링 설정")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout = QVBoxLayout()
        
        # Simplify 강도
        simplify_layout = QHBoxLayout()
        simplify_layout.addWidget(QLabel("Polygon Simplify:"))
        self.simplify_slider = QSlider(Qt.Horizontal)
        self.simplify_slider.setRange(0, 5)
        self.simplify_slider.setValue(self.config.get('simplify_strength', 0))
        self.simplify_slider.setTickPosition(QSlider.TicksBelow)
        self.simplify_slider.setTickInterval(1)
        simplify_layout.addWidget(self.simplify_slider)
        self.simplify_label = QLabel(f"{self.simplify_slider.value()}")
        self.simplify_slider.valueChanged.connect(lambda v: self.simplify_label.setText(str(v)))
        simplify_layout.addWidget(self.simplify_label)
        layout.addLayout(simplify_layout)
        
        # 홀 채우기
        hole_layout = QHBoxLayout()
        self.hole_fill_checkbox = QCheckBox("홀 채우기")
        self.hole_fill_checkbox.setChecked(self.config.get('fill_holes_enabled', True))
        hole_layout.addWidget(self.hole_fill_checkbox)
        hole_layout.addWidget(QLabel("크기:"))
        self.hole_size_spinbox = QSpinBox()
        self.hole_size_spinbox.setRange(100, 10000)
        self.hole_size_spinbox.setSingleStep(100)
        self.hole_size_spinbox.setValue(self.config.get('max_hole_size', 1000))
        self.hole_size_spinbox.setSuffix(" px")
        hole_layout.addWidget(self.hole_size_spinbox)
        hole_layout.addStretch()
        layout.addLayout(hole_layout)
        
        # 네거티브 확장
        negative_layout = QHBoxLayout()
        negative_layout.addWidget(QLabel("제외 영역 확장:"))
        self.negative_spinbox = QSpinBox()
        self.negative_spinbox.setRange(0, 50)
        self.negative_spinbox.setValue(self.config.get('negative_expand', 10))
        self.negative_spinbox.setSuffix(" px")
        negative_layout.addWidget(self.negative_spinbox)
        negative_layout.addStretch()
        layout.addLayout(negative_layout)
        
        group.setLayout(layout)
        return group
    



    # 진행도
    def create_progress_group(self):
        """진행 상황 그룹"""
        group = QGroupBox("📊 진행 상황")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout = QVBoxLayout()
        
        # 통합 진행 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v/%m)")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                background-color: #f0f0f0;
                height: 25px;
                text-align: center;
                font-weight: bold;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 상태 라벨 (메인)
        self.status_label = QLabel("대기 중...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # 상세 통계 라벨 (색상 표시 추가)
        self.stats_label = QLabel("🟢 라벨링: 0/0 (0%) | 🟠 스킵: 0/0 (0%) | ⚪ 미처리: 0/0 (0%)")
        self.stats_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.stats_label)
        
        group.setLayout(layout)
        return group
    


    # 컨텍스트
    def create_status_group(self):
        """현재 상태 그룹 (라벨링 중 실시간 정보)"""
        group = QGroupBox("🎯 현재 상태")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #f0f8f0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #4CAF50;
            }
        """)
        layout = QVBoxLayout()
        
        # 상태 정보 레이블들
        info_layout = QVBoxLayout()
        
        # 모드 정보
        self.mode_label = QLabel("📌 모드: 대기 중...")
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        info_layout.addWidget(self.mode_label)
        
        # Split 모드 정보
        self.split_label = QLabel("🔀 영역: 전체 (E)")
        self.split_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #555;")
        info_layout.addWidget(self.split_label)
        
        # 브러시 크기 (그리기/지우기)
        self.brush_label = QLabel("🖌️ 브러시 크기: - px")
        self.brush_label.setStyleSheet("font-size: 12px; color: #555;")
        info_layout.addWidget(self.brush_label)
        
        # SAM Expand 크기
        self.expand_label = QLabel("🎯 제외 영역 확장: 10 px")
        self.expand_label.setStyleSheet("font-size: 12px; color: #555;")
        info_layout.addWidget(self.expand_label)
        
        # Simplify 강도 (실시간)
        self.simplify_status_label = QLabel("📐 Simplify 강도: 0")
        self.simplify_status_label.setStyleSheet("font-size: 12px; color: #555;")
        info_layout.addWidget(self.simplify_status_label)
        
        layout.addLayout(info_layout)
        
        # 안내 텍스트
        hint_label = QLabel("💡 라벨링 중 실시간 업데이트")
        hint_label.setStyleSheet("color: gray; font-size: 9px; font-style: italic;")
        layout.addWidget(hint_label)
        
        group.setLayout(layout)
        return group
    



    # GUI 로그
    def create_log_group(self):
        """로그 그룹"""
        group = QGroupBox("📝 로그")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout = QVBoxLayout()
        
        # 로그 텍스트
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)
        
        # 로그 지우기 버튼
        clear_btn = QPushButton("로그 지우기")
        clear_btn.setFocusPolicy(Qt.NoFocus)  # 키보드 포커스 차단
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)
        
        group.setLayout(layout)
        return group
    



    # 라벨링 시작 중단 종료 버튼
    def create_button_layout(self):
        """버튼 레이아웃"""
        layout = QHBoxLayout()
        
        # 시작 버튼
        self.start_btn = QPushButton("🚀 라벨링 시작")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; font-weight: bold;")
        self.start_btn.setToolTip("라벨링 작업 시작")
        self.start_btn.setFocusPolicy(Qt.NoFocus)  # 키보드 포커스 차단
        self.start_btn.clicked.connect(self.start_labeling)
        layout.addWidget(self.start_btn, stretch=2)
        
        # 중단 버튼 (라벨링 작업만 중단)
        self.stop_btn = QPushButton("⏹️ 라벨링 중단")
        self.stop_btn.setStyleSheet("background-color: #ff9800; color: white; font-size: 14px; padding: 10px; font-weight: bold;")
        self.stop_btn.setToolTip("현재 라벨링 작업 중단 (프로그램은 계속 실행)")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setFocusPolicy(Qt.NoFocus)  # 키보드 포커스 차단
        self.stop_btn.clicked.connect(self.stop_labeling)
        layout.addWidget(self.stop_btn, stretch=2)
        
        # 종료 버튼
        self.quit_btn = QPushButton("🚪 프로그램 종료")
        self.quit_btn.setStyleSheet("background-color: #757575; color: white; font-size: 14px; padding: 10px; font-weight: bold;")
        self.quit_btn.setToolTip("프로그램 완전 종료")
        self.quit_btn.setFocusPolicy(Qt.NoFocus)  # 키보드 포커스 차단
        self.quit_btn.clicked.connect(self.close_application)
        layout.addWidget(self.quit_btn, stretch=2)
        
        return layout
    


    

    # 사용 가이드 
    def create_help_panel(self):
        """사용 가이드 패널 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # 제목
        title = QLabel("<h2>🏷️ 사용 가이드</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 가이드 내용 (스크롤 가능)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        help_text = QLabel("""
            <h3>📋 기본 사용법</h3>
            <ol>
            <li><b>입력 폴더 선택</b>: 라벨링할 이미지가 있는 폴더</li>
            <li><b>출력 폴더 선택</b>: 결과를 저장할 상위 폴더</li>
            <li><b>폴더 이름 설정</b>: 라벨링 결과를 저장할 폴더 이름
            <ul>
            <li>기본값: labels_날짜시간</li>
            <li>원하는 이름으로 수정 가능</li>
            <li>🔄 버튼: 현재 날짜시간으로 새로고침</li>
            <li>이미 폴더가 있으면 그 안에 추가 저장됨</li>
            </ul>
            </li>
            <li><b>설정 조정</b> (선택사항): Simplify 강도, 홀 채우기 등</li>
            <li><b>Start Labeling 클릭</b>: 라벨링 시작</li>
            </ol>

            <h3>⌨️ 라벨링 키보드 단축키</h3>
            <b>모드 전환:</b>
            <ul>
            <li><b>1</b> - SAM 모드 (자동 인식, 추천)</li>
            <li><b>2</b> - 그리기 모드 (수동 추가)</li>
            <li><b>3</b> - 지우개 모드 (수동 제거)</li>
            </ul>

            <b>Split Mode (좌우 분리):</b>
            <ul>
            <li><b>Q</b> - 왼쪽 영역만 라벨링</li>
            <li><b>W</b> - 오른쪽 영역만 라벨링</li>
            <li><b>E</b> - 전체 영역 라벨링 (기본값)</li>
            </ul>

            <b>기능:</b>
            <ul>
            <li><b>M</b> - Mask 미리보기 (컬러)</li>
            <li><b>N</b> - Binary Mask 미리보기 (흑백)</li>
            <li><b>S</b> - Polygon Simplify 조절 (+/- 키로 조절)</li>
            <li><b>V</b> - 배경 시각화 변경 (일반/엣지/대비)</li>
            <li><b>C</b> - 오버레이 색상 변경</li>
            <li><b>F</b> - 홀 채우기 ON/OFF</li>
            <li><b>U 또는 Ctrl+Z</b> - Undo</li>
            <li><b>PageUp</b> - 텍스트 표시 ON/OFF</li>
            </ul>

            <b>진행:</b>
            <ul>
            <li><b>SPACE</b> - 저장하고 다음 이미지</li>
            <li><b>` (백틱)</b> - 현재 이미지 건너뛰기 (탭 위의 키)</li>
            <li><b>R</b> - 현재 이미지 리셋</li>
            <li><b>ESC</b> - 라벨링 종료</li>
            </ul>

            <h3>🖱️ 마우스 조작</h3>
            <ul>
            <li><b>좌클릭</b> - 포함할 영역 선택</li>
            <li><b>우클릭</b> - 제외할 영역 선택</li>
            <li><b>마우스 휠</b> - 브러시 크기 / Expand 크기 조절</li>
            <li><b>드래그</b> - 그리기/지우기 (모드 2, 3)</li>
            </ul>

            <h3>⚙️ 설정 항목</h3>
            <ul>
            <li><b>Polygon Simplify (0~5)</b>: 외곽선 직선화 강도
            <ul>
            <li>0 = 원본 그대로 (기본값)</li>
            <li>1-2 = 약함</li>
            <li>3-4 = 보통</li>
            <li>5 = 매우 강함</li>
            <li>💡 S 키로 실시간 조절 가능</li>
            </ul>
            </li>
            <li><b>홀 채우기</b>: 영역 내부 작은 구멍 자동 채우기</li>
            <li><b>제외 영역 확장</b>: 우클릭 시 제외 영역 크기</li>
            </ul>

            <h3>🔄 작업 이어하기</h3>
            <ul>
            <li><b>progress_log.json</b> 파일을 선택하여 이전 작업 이어하기</li>
            <li>입력/출력 폴더 경로 자동 설정</li>
            <li>이미 라벨링한 이미지는 자동 스킵</li>
            </ul>

            <h3>📊 진행 상황</h3>
            <ul>
            <li><b>처리</b>: 현재까지 확인한 이미지</li>
            <li><b>라벨링</b>: 실제로 저장한 이미지</li>
            <li><b>스킵</b>: 건너뛴 이미지 (K 키)</li>
            </ul>

            <h3>📁 결과 폴더 구조</h3>
            <pre>
            출력폴더/
            └── 폴더이름/
                ├── json/           # Simple JSON
                ├── yolo/labels/    # YOLO Segmentation
                ├── segman/masks/   # SegMAN
                └── visualizations/ # 시각화 이미지
            </pre>

            <h3>💡 팁</h3>
            <ul>
            <li>좌우 스톨이 겹치면 <b>Q/W 키</b>로 분리 라벨링</li>
            <li>울퉁불퉁한 경계는 <b>S 키</b>로 Simplify 조절</li>
            <li>격자 부분이 채워지면 <b>F 키</b>로 홀 채우기 OFF</li>
            <li>잘못 클릭하면 <b>U 키</b>로 Undo</li>
            </ul>

            <h3>❓ 문제 해결</h3>
            <ul>
            <li><b>SAM 로딩 느림</b>: 첫 실행 시 정상 (5~10초)</li>
            <li><b>메모리 부족</b>: 16GB RAM 권장</li>
            <li><b>이미지 사라짐</b>: 입력 폴더와 source_image 같으면 복사 안 함</li>
            </ul>

            <h3>🎯 버튼 설명</h3>
            <ul>
            <li><b>🚀 라벨링 시작</b>: 라벨링 작업 시작</li>
            <li><b>⏹️ 라벨링 중단</b>: 현재 라벨링 작업 중단 (프로그램은 계속 실행)</li>
            <li><b>🚪 프로그램 종료</b>: 프로그램 완전 종료</li>
            <li><b>🔄 작업 이어하기</b>: progress_log.json을 선택하여 이전 작업 이어하기</li>
            <li><b>❓ 사용 가이드</b>: 이 가이드 패널 열기/닫기</li>
            </ul>
        """)
        help_text.setTextFormat(Qt.RichText)
        help_text.setWordWrap(True)
        help_text.setAlignment(Qt.AlignTop)
        help_text.setOpenExternalLinks(True)
        
        scroll.setWidget(help_text)
        layout.addWidget(scroll)
        
        return widget
    




    # 가이드 표시 <-> 라벨링 영역
    def toggle_help_panel(self):
        """사용 가이드 패널 토글"""
        # 오른쪽 패널에서 가이드와 라벨링 영역 토글
        current_widget = self.right_panel_layout.itemAt(0).widget() if self.right_panel_layout.count() > 0 else None
        
        if current_widget == self.help_panel:
            # 가이드 → 라벨링 영역
            self.right_panel_layout.removeWidget(self.help_panel)
            self.help_panel.setParent(None)
            if self.labeling_widget:
                self.right_panel_layout.addWidget(self.labeling_widget)
            self.help_toggle_btn.setText("❓ 사용 가이드")
        else:
            # 라벨링 영역 → 가이드
            if self.labeling_widget:
                self.right_panel_layout.removeWidget(self.labeling_widget)
                self.labeling_widget.setParent(None)
            self.right_panel_layout.addWidget(self.help_panel)
            self.help_panel.setVisible(True)
            self.help_toggle_btn.setText("❌ 가이드 닫기")
    





    # 프로그램 종료 
    def close_application(self):
        """프로그램 종료"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, '확인',
                "라벨링이 진행 중입니다. 프로그램을 종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait()
                self.close()
        else:
            self.close()
    


    # 오버레이 상태값 
    def setup_status_monitor(self):
        """상태 모니터링 타이머 설정"""
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_display)
        self.status_timer.start(500)  # 0.5초마다 업데이트
    




    def update_status_display(self):
        """상태 파일을 읽어서 현재 상태 표시"""
        # 라벨링이 실행 중이지 않으면 업데이트하지 않음 ⭐
        if not self.is_labeling_active:
            return
        
        try:
            status_path = Path(self.status_file)
            if not status_path.exists():
                return  # 상태 파일이 없으면 업데이트하지 않음
            
            with open(status_path, 'r', encoding='utf-8') as f:
                status = json.load(f)
            
            if status:
                
                # 모드 표시
                mode = status.get('mode', 'SAM')
                mode_text_map = {
                    'SAM': '📌 모드: SAM (자동 인식)',
                    'DRAW': '📌 모드: 그리기 (수동 추가)',
                    'ERASE': '📌 모드: 지우개 (수동 제거)'
                }
                self.mode_label.setText(mode_text_map.get(mode, f'📌 모드: {mode}'))
                
                # Split 모드 표시
                split_mode = status.get('split_mode', 'entire')
                split_text_map = {
                    'left': '🔀 영역: 왼쪽 (Q)',
                    'right': '🔀 영역: 오른쪽 (W)',
                    'entire': '🔀 영역: 전체 (E)'
                }
                self.split_label.setText(split_text_map.get(split_mode, '🔀 영역: 전체 (E)'))
                
                # 브러시 크기 (항상 표시)
                brush_size = status.get('brush_size', 0)
                self.brush_label.setText(f"🖌️ 브러시 크기: {brush_size} px")
                
                # Expand 크기
                expand_size = status.get('negative_expand', 10)
                self.expand_label.setText(f"🎯 제외 영역 확장: {expand_size} px")
                
                # Simplify 강도
                simplify = status.get('simplify_strength', 0)
                self.simplify_status_label.setText(f"📐 Simplify 강도: {simplify}")
                
                # 진행상황 업데이트 (작업 폴더의 로그 읽기) ⭐
                if self.current_work_folder is None:
                    return  # 작업 폴더가 설정되지 않았으면 업데이트하지 않음
                
                log_file = self.current_work_folder / "progress_log.json"
                
                total_images = None
                labeled_count = 0
                skipped_count = 0
                current_idx = status.get('current_index')
                
                # progress_log.json에서 전체 정보 읽기 (우선순위)
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            log_data = json.load(f)
                        total_images = log_data.get('total_images')  # ⭐ 여기서 읽기
                        labeled_count = len(log_data.get('labeled', []))
                        skipped_count = len(log_data.get('skipped', []))
                    except Exception as e:
                        import traceback
                        print(f"⚠️ progress_log.json 읽기 실패: {e}")
                        traceback.print_exc()
                
                # progress_log.json에 없으면 status에서 읽기 (fallback)
                if total_images is None:
                    total_images = status.get('total_images')
                
                if total_images is not None and total_images > 0:
                    
                    # 진행 개수 = 라벨링 + 스킵
                    processed_count = labeled_count + skipped_count
                    not_processed_count = total_images - processed_count
                    
                    # 진행률 계산
                    progress_percent = (processed_count / total_images) * 100 if total_images > 0 else 0
                    
                    # 통합 프로그레스 바 업데이트
                    self.progress_bar.setMaximum(total_images)
                    self.progress_bar.setValue(processed_count)
                    
                    # 상태 라벨 업데이트
                    if current_idx is not None:
                        self.status_label.setText(f"진행 중... {progress_percent:.1f}%")
                    else:
                        self.status_label.setText(f"처리 완료: {processed_count}/{total_images}")
                    
                    # 통계 라벨 업데이트 (상세 정보)
                    self.stats_label.setText(
                        f"🟢 라벨링: {labeled_count}개 | "
                        f"🟠 스킵: {skipped_count}개 | "
                        f"⚪ 미처리: {not_processed_count}개"
                    )
                
        except Exception:
            pass  # 파일이 없거나 읽기 실패 시 무시
    


    # GUI 로그 출력

    def append_log(self, text):
        """로그 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {text.strip()}"
        self.log_text.append(formatted_message)
        # 스크롤을 맨 아래로
        self.log_text.moveCursor(QTextCursor.End)
    



    # 폴더 선택 다이얼로그
    def browse_input_folder(self):
        """입력 폴더 선택"""
        # 프로젝트 루트 기준으로 datasets 폴더 경로 설정
        project_root = Path(__file__).parent.parent  # src의 부모 = 프로젝트 루트
        datasets_path = project_root / "data_io" / "datasets"
        
        # datasets 폴더가 없으면 생성
        datasets_path.mkdir(parents=True, exist_ok=True)
        
        folder = QFileDialog.getExistingDirectory(
            self, 
            "입력 폴더 선택",
            str(datasets_path)  # 초기 경로 설정
        )
        if folder:
            self.input_folder_edit.setText(folder)
            # 입력 폴더 변경 시 출력 폴더명 자동 업데이트
            self._update_output_name_from_input()
    
    def browse_output_folder(self):
        """출력 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(self, "출력 폴더 선택")
        if folder:
            self.output_folder_edit.setText(folder)
    



    # 출력폴더 생성 자동화
    def _update_output_name_from_input(self):
        """입력 폴더명을 기반으로 출력 폴더명 자동 업데이트"""
        input_folder = self.input_folder_edit.text()
        if not input_folder:
            return
        
        # 입력 폴더명 추출
        input_path = Path(input_folder)
        input_folder_name = input_path.name
        
        # 출력 폴더명 생성: 입력 폴더명 + "_outputs"
        output_name = f"{input_folder_name}_outputs"
        self.output_name_edit.setText(output_name)
        
        # 출력 폴더 경로 자동 설정
        project_root = Path(__file__).parent.parent
        outputs_path = project_root / "data_io" / "outputs"
        outputs_path.mkdir(parents=True, exist_ok=True)
        self.output_folder_edit.setText(str(outputs_path))
    
    def refresh_output_name(self):
        """출력 폴더 이름을 현재 날짜시간으로 새로고침"""
        new_name = f"labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_name_edit.setText(new_name)
    



    # 라벨링 시작 
    def start_labeling(self):
        """라벨링 시작"""
        # 설정 수집
        config = {
            'model_path': self.model_path_edit.text(),
            'input_folder': self.input_folder_edit.text(),
            'output_folder': self.output_folder_edit.text(),
            'output_name': self.output_name_edit.text().strip(),  # 사용자 지정 폴더 이름
            'simplify_strength': self.simplify_slider.value(),
            'fill_holes_enabled': self.hole_fill_checkbox.isChecked(),
            'max_hole_size': self.hole_size_spinbox.value(),
            'negative_expand': self.negative_spinbox.value(),
        }
        
        # 유효성 검사
        input_path = Path(config['input_folder'])
        if not input_path.exists():
            QMessageBox.warning(self, "경고", f"입력 폴더가 존재하지 않습니다:\n{config['input_folder']}")
            return
        
        # 입력 폴더에 이미지 확인 ⭐
        image_files = list(input_path.glob("*.jpg")) + \
                     list(input_path.glob("*.png")) + \
                     list(input_path.glob("*.jpeg")) + \
                     list(input_path.glob("*.JPG")) + \
                     list(input_path.glob("*.PNG"))
        
        if len(image_files) == 0:
            QMessageBox.warning(
                self,
                "경고",
                f"입력 폴더에 이미지가 없습니다:\n{config['input_folder']}\n\n"
                f"jpg, png, jpeg 형식의 이미지 파일을 추가해주세요."
            )
            return
        
        # 출력 폴더 이름 유효성 검사
        output_name = config['output_name']
        if not output_name:
            QMessageBox.warning(self, "경고", "출력 폴더 이름을 입력하세요!")
            return
        
        # 파일명으로 사용할 수 없는 문자 체크
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        if any(char in output_name for char in invalid_chars):
            QMessageBox.warning(self, "경고", f"출력 폴더 이름에 사용할 수 없는 문자가 있습니다:\n{', '.join(invalid_chars)}")
            return
        
        # 출력 폴더 상위 경로 존재 확인 ⭐
        output_base_parent = Path(config['output_folder'])
        if not output_base_parent.exists():
            QMessageBox.warning(
                self,
                "경고",
                f"출력 폴더의 상위 경로가 존재하지 않습니다:\n{output_base_parent}\n\n"
                f"경로를 생성하거나 올바른 경로를 선택해주세요."
            )
            return
        
        # 현재 작업 폴더 경로 설정 ⭐
        output_name = config['output_name']
        if not output_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"labels_{timestamp}"
        self.current_work_folder = Path(config['output_folder']) / output_name
        print(f"📁 작업 폴더: {self.current_work_folder}")
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("라벨링 중...")
        self.is_labeling_active = True  # ⭐ 라벨링 시작
        
        # 프로그레스 바 초기화 (기존 작업 정보가 있으면 반영) ⭐
        import os
        resume_labeled = int(os.environ.get('LABEL_TOOL_RESUME_LABELED', '0'))
        resume_skipped = int(os.environ.get('LABEL_TOOL_RESUME_SKIPPED', '0'))
        resume_total = int(os.environ.get('LABEL_TOOL_RESUME_TOTAL', '0'))
        
        if resume_total > 0 and (resume_labeled > 0 or resume_skipped > 0):
            # 이어하기 모드
            resume_processed = resume_labeled + resume_skipped
            resume_not_processed = resume_total - resume_processed
            resume_percent = (resume_processed / resume_total) * 100
            
            self.progress_bar.setMaximum(resume_total)
            self.progress_bar.setValue(resume_processed)
            self.stats_label.setText(
                f"🟢 라벨링: {resume_labeled}개 | 🟠 스킵: {resume_skipped}개 | ⚪ 미처리: {resume_not_processed}개"
            )
            print(f"📊 이어하기: {resume_percent:.1f}% ({resume_processed}/{resume_total})")
            
            # 환경변수 클리어
            os.environ.pop('LABEL_TOOL_RESUME_LABELED', None)
            os.environ.pop('LABEL_TOOL_RESUME_SKIPPED', None)
            os.environ.pop('LABEL_TOOL_RESUME_TOTAL', None)
        else:
            # 신규 작업
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(image_files))
            self.stats_label.setText(f"🟢 라벨링: 0개 | 🟠 스킵: 0개 | ⚪ 미처리: {len(image_files)}개")
        
        # 이미지 파일 목록 저장
        self.image_files = sorted(image_files)
        self.current_image_index = 0
        self.labeled_list = []
        self.skipped_list = []
        
        # progress_log.json 초기화 또는 로드
        from label_tool import load_progress_log, save_progress_log
        log_data = load_progress_log(self.current_work_folder)
        if 'labeled' in log_data:
            self.labeled_list = log_data['labeled']
        if 'skipped' in log_data:
            self.skipped_list = log_data['skipped']
        
        # 이미 처리된 이미지 제외
        processed = set(self.labeled_list + self.skipped_list)
        remaining_images = [img for img in self.image_files if str(img) not in processed]
        
        if len(remaining_images) == 0:
            QMessageBox.information(self, "알림", "모든 이미지가 이미 처리되었습니다.")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            return
        
        self.image_files = remaining_images
        self.current_image_index = 0
        
        # progress_log.json에 전체 개수 저장
        # save_progress_log는 키워드 인자를 받지 않으므로 직접 파일에 저장
        log_file = self.current_work_folder / "progress_log.json"
        log_data = {
            'labeled': self.labeled_list,
            'skipped': self.skipped_list,
            'input_folder': str(input_path),
            'output_folder': str(self.current_work_folder),
            'total_images': len(image_files),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.current_work_folder.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print("="*60)
        print("🚀 라벨링 시작")
        print(f"📁 작업 폴더: {self.current_work_folder}")
        print(f"📊 총 이미지: {len(image_files)}개 (미처리: {len(remaining_images)}개)")
        print("="*60)
        
        # 라벨링 모드로 전환
        self.setup_labeling_widget(config)
        
        # 첫 이미지 로드
        self.load_next_image()
    
    def setup_labeling_widget(self, config):
        """라벨링 위젯 설정"""
        # 오른쪽 패널의 기존 위젯 제거 (가이드)
        for i in reversed(range(self.right_panel_layout.count())):
            item = self.right_panel_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        
        # LabelingWidget 생성 및 추가
        if self.labeling_widget is None:
            self.labeling_widget = LabelingWidget(config['model_path'], config)
            self.labeling_widget.image_saved.connect(self.on_image_saved)
            self.labeling_widget.image_skipped.connect(self.on_image_skipped)
        
        # 라벨링 영역에 추가
        self.right_panel_layout.addWidget(self.labeling_widget)
        self.is_labeling_active = True
    
    def cleanup_labeling_widget(self):
        """라벨링 위젯 정리"""
        # LabelingWidget 명시적 정리
        if self.labeling_widget:
            try:
                # 타이머 정지
                if hasattr(self.labeling_widget, 'update_timer'):
                    self.labeling_widget.update_timer.stop()
                
                # SimpleLabelTool 정리
                if hasattr(self.labeling_widget, 'label_tool') and self.labeling_widget.label_tool:
                    del self.labeling_widget.label_tool
                    self.labeling_widget.label_tool = None
                
                # CUDA 메모리 정리
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("🧹 라벨링 위젯 정리 시 CUDA 메모리 정리")
                    
            except Exception as e:
                print(f"⚠️ 라벨링 위젯 정리 중 오류: {e}")
        
        # 오른쪽 패널의 기존 위젯 제거 (라벨링 영역)
        for i in reversed(range(self.right_panel_layout.count())):
            item = self.right_panel_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        
        # 가이드 패널 다시 추가
        self.right_panel_layout.addWidget(self.help_panel)
        self.help_panel.setVisible(False)  # 초기에는 숨김
        
        self.is_labeling_active = False
    
    def load_next_image(self):
        """다음 이미지 로드"""
        if self.current_image_index >= len(self.image_files):
            # 모든 이미지 처리 완료
            self.finish_labeling()
            return
        
        current_image = self.image_files[self.current_image_index]
        
        if self.labeling_widget:
            success = self.labeling_widget.set_image(current_image)
            if success:
                self.append_log(f"📌 이미지 로드: {Path(current_image).name} ({self.current_image_index + 1}/{len(self.image_files)})")
                self.status_label.setText(f"라벨링 중... ({self.current_image_index + 1}/{len(self.image_files)})")
            else:
                self.append_log(f"⚠️ 이미지 로드 실패: {Path(current_image).name}")
                # 다음 이미지로 이동
                self.current_image_index += 1
                self.load_next_image()
    
    def on_image_saved(self, img_path):
        """이미지 저장 완료 처리"""
        img_path_str = str(img_path)
        
        # 저장
        if self.labeling_widget and self.current_work_folder:
            success = self.labeling_widget.save_current_image(img_path, self.current_work_folder)
            if success:
                # 라벨링 목록에 추가
                if img_path_str not in self.labeled_list:
                    self.labeled_list.append(img_path_str)
                
                # progress_log.json 업데이트
                from label_tool import save_progress_log
                save_progress_log(
                    self.current_work_folder,
                    self.labeled_list,
                    self.skipped_list
                )
                
                self.append_log(f"✅ 저장 완료: {Path(img_path).name}")
                
                # 프로그레스 바 업데이트
                total = self.progress_bar.maximum()
                processed = len(self.labeled_list) + len(self.skipped_list)
                self.progress_bar.setValue(processed)
                
                # 통계 업데이트
                not_processed = total - processed
                self.stats_label.setText(
                    f"🟢 라벨링: {len(self.labeled_list)}개 | "
                    f"🟠 스킵: {len(self.skipped_list)}개 | "
                    f"⚪ 미처리: {not_processed}개"
                )
            else:
                self.append_log(f"⚠️ 저장 실패: {Path(img_path).name} (마스크가 없습니다)")
        
        # 다음 이미지로 이동
        self.current_image_index += 1
        self.load_next_image()
    
    def on_image_skipped(self, img_path):
        """이미지 스킵 처리"""
        img_path_str = str(img_path)
        
        # 스킵 목록에 추가
        if img_path_str not in self.skipped_list:
            self.skipped_list.append(img_path_str)
        
        # progress_log.json 업데이트
        from label_tool import save_progress_log
        save_progress_log(
            self.current_work_folder,
            self.labeled_list,
            self.skipped_list
        )
        
        self.append_log(f">> 스킵: {Path(img_path).name}")
        
        # 프로그레스 바 업데이트
        total = self.progress_bar.maximum()
        processed = len(self.labeled_list) + len(self.skipped_list)
        self.progress_bar.setValue(processed)
        
        # 통계 업데이트
        not_processed = total - processed
        self.stats_label.setText(
            f"🟢 라벨링: {len(self.labeled_list)}개 | "
            f"🟠 스킵: {len(self.skipped_list)}개 | "
            f"⚪ 미처리: {not_processed}개"
        )
        
        # 다음 이미지로 이동
        self.current_image_index += 1
        self.load_next_image()
    
    def finish_labeling(self):
        """라벨링 완료 처리"""
        total = self.progress_bar.maximum()
        labeled_count = len(self.labeled_list)
        skipped_count = len(self.skipped_list)
        processed_count = labeled_count + skipped_count
        
        self.append_log(f"✅ 라벨링 완료! (라벨링: {labeled_count}개, 스킵: {skipped_count}개)")
        
        # 완료 메시지
        message = f"""라벨링이 완료되었습니다!

🟢 라벨링: {labeled_count}개
🟠 스킵: {skipped_count}개
⚪ 미처리: {total - processed_count}개

📁 결과 폴더: {self.current_work_folder}"""
        
        QMessageBox.information(self, "완료", message)
        
        # 일반 모드로 복귀
        self.cleanup_labeling_widget()
        
        # UI 상태 복원
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("완료!")
        
        # 프로그레스 바 최종 업데이트
        self.progress_bar.setValue(processed_count)
        self.stats_label.setText(
            f"🟢 라벨링: {labeled_count}개 | "
            f"🟠 스킵: {skipped_count}개 | "
            f"⚪ 미처리: {total - processed_count}개"
        )
    





    # 라벨링 중단
    def stop_labeling(self):
        """라벨링 중단"""
        if self.is_labeling_active:
            # 진행 상황 저장
            from label_tool import save_progress_log
            if self.current_work_folder:
                save_progress_log(
                    self.current_work_folder,
                    self.labeled_list,
                    self.skipped_list
                )
            
            self.append_log("⏸️ 라벨링 중단됨")
            
            # 일반 모드로 복귀
            self.cleanup_labeling_widget()
            
            # UI 상태 복원
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("중단됨")
    
    def update_progress(self, current, total):
        """진행 상황 업데이트 (Worker에서 호출, 현재는 사용 안 함)"""
        # 상태 파일 기반 업데이트로 대체됨
        pass
    



    # 라벨링 작업 완료
    def labeling_finished(self):
        """라벨링 완료"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("완료!")
        self.is_labeling_active = False  # ⭐ 라벨링 종료
        
        # 최종 통계 계산 (progress_log.json에서 읽기)
        total = 0
        labeled_count = 0
        skipped_count = 0
        not_processed_count = 0
        
        try:
            # progress_log.json에서 읽기 (작업 폴더) ⭐
            if self.current_work_folder is None:
                QMessageBox.information(self, "완료", "라벨링이 완료되었습니다!")
                return
            
            log_file = self.current_work_folder / "progress_log.json"
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                total = log_data.get('total_images', 0)
                labeled_count = len(log_data.get('labeled', []))
                skipped_count = len(log_data.get('skipped', []))
            
            # 미처리 개수
            not_processed_count = total - labeled_count - skipped_count
            
            # 진행 개수
            processed_count = labeled_count + skipped_count
            
            # 프로그레스 바 최종 업데이트
            if total > 0:
                self.progress_bar.setMaximum(total)
                self.progress_bar.setValue(processed_count)
                
                # 통계 업데이트
                self.stats_label.setText(
                    f"🟢 라벨링: {labeled_count}개 | "
                    f"🟠 스킵: {skipped_count}개 | "
                    f"⚪ 미처리: {not_processed_count}개"
                )
                
                # 완료 메시지 (통계 포함)
                message = f"""
                라벨링이 완료되었습니다!

                📊 최종 통계:
                • 총 이미지: {total}개
                • 처리: {processed_count}개 ({processed_count/total*100:.1f}%)
                
                🟢 라벨링 완료: {labeled_count}개
                🟠 스킵: {skipped_count}개
                ⚪ 미처리: {not_processed_count}개
                            
                """
            else:
                message = "라벨링이 완료되었습니다!"
            
            QMessageBox.information(self, "✅ 라벨링 완료", message)
            
        except Exception as e:
            QMessageBox.information(self, "완료", "라벨링이 완료되었습니다!")
    
    def labeling_error(self, error_msg):
        """라벨링 에러"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("에러 발생")
        self.is_labeling_active = False  # ⭐ 라벨링 종료
        
        QMessageBox.critical(self, "에러", error_msg)
    
    def load_config(self):
        """설정 로드"""
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # 기본 설정
        return {
            'input_folder': 'source_image',
            'output_folder': '.',
            'simplify_strength': 0,  # 원본 그대로 (S 키로 조절 가능)
            'fill_holes_enabled': True,
            'max_hole_size': 1000,
            'negative_expand': 10,  # 제외 영역 확장 기본값 10px
        }
    
    def resume_work(self):
        """작업 이어하기 - progress_log.json 파일 선택"""
        from PyQt5.QtWidgets import QFileDialog
        
        # progress_log.json 파일 선택
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "작업 로그 파일 선택",
            "",
            "Progress Log (progress_log.json);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # 로그 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            input_folder = log_data.get('input_folder')
            output_folder = log_data.get('output_folder')
            labeled_count = len(log_data.get('labeled', []))
            skipped_count = len(log_data.get('skipped', []))
            
            if not input_folder or not output_folder:
                QMessageBox.warning(self, "에러", "로그 파일에 폴더 정보가 없습니다.\n이전 버전의 로그 파일일 수 있습니다.")
                return
            
            # 입력 폴더 존재 확인 ⭐
            input_path = Path(input_folder)
            if not input_path.exists():
                QMessageBox.critical(
                    self,
                    "❌ 작업 이어하기 불가",
                    f"입력 폴더가 존재하지 않습니다:\n{input_folder}\n\n"
                    f"폴더를 찾을 수 없어 작업을 이어갈 수 없습니다."
                )
                return
            
            # 입력 폴더에 이미지 확인 ⭐
            image_files = list(input_path.glob("*.jpg")) + \
                         list(input_path.glob("*.png")) + \
                         list(input_path.glob("*.jpeg")) + \
                         list(input_path.glob("*.JPG")) + \
                         list(input_path.glob("*.PNG"))
            
            if len(image_files) == 0:
                QMessageBox.critical(
                    self,
                    "❌ 작업 이어하기 불가",
                    f"입력 폴더에 이미지가 없습니다:\n{input_folder}\n\n"
                    f"작업을 이어갈 수 없습니다."
                )
                return
            
            # 출력 폴더 확인 ⭐
            output_path = Path(output_folder)
            if not output_path.exists():
                QMessageBox.warning(
                    self,
                    "⚠️ 출력 폴더 없음",
                    f"출력 폴더가 존재하지 않습니다:\n{output_folder}\n\n"
                    f"새로 생성하여 작업합니다."
                )
            
            # 확인 메시지
            total_images = log_data.get('total_images', len(image_files))
            not_processed = total_images - labeled_count - skipped_count
            
            msg = f"📋 작업 로그 정보\n\n"
            msg += f"입력 폴더: {input_folder}\n"
            msg += f"  └─ 이미지: {len(image_files)}개\n\n"
            msg += f"출력 폴더: {output_folder}\n\n"
            msg += f"진행 상황:\n"
            msg += f"  🟢 라벨링 완료: {labeled_count}개\n"
            msg += f"  🟠 스킵: {skipped_count}개\n"
            msg += f"  ⚪ 미처리: {not_processed}개\n\n"
            msg += "이 작업을 이어서 진행하시겠습니까?"
            
            reply = QMessageBox.question(
                self,
                "작업 이어하기",
                msg,
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # 폴더 경로 설정
            self.input_folder_edit.setText(input_folder)
            
            # 출력 폴더는 부모 폴더만 설정
            output_parent = str(Path(output_folder).parent)
            output_name = Path(output_folder).name
            self.output_folder_edit.setText(output_parent)
            self.output_name_edit.setText(output_name)
            
            # 작업 폴더 설정 ⭐
            self.current_work_folder = Path(output_folder)
            
            # GUI 진행상황 즉시 업데이트 ⭐
            processed_count = labeled_count + skipped_count
            progress_percent = (processed_count / total_images) * 100 if total_images > 0 else 0
            
            self.progress_bar.setMaximum(total_images)
            self.progress_bar.setValue(processed_count)
            self.stats_label.setText(
                f"🟢 라벨링: {labeled_count}개 | 🟠 스킵: {skipped_count}개 | ⚪ 미처리: {not_processed}개"
            )
            self.status_label.setText(f"이어하기 준비 완료: {progress_percent:.1f}%")
            
            print(f"🔄 작업 이어하기 설정 완료")
            print(f"  입력: {input_folder}")
            print(f"  출력: {output_folder}")
            print(f"  이전 작업: 라벨링 {labeled_count}개, 스킵 {skipped_count}개")
            print(f"  📊 진행률: {progress_percent:.1f}% ({processed_count}/{total_images})")
            
            QMessageBox.information(
                self,
                "설정 완료",
                f"작업 이어하기 설정이 완료되었습니다.\n\n"
                f"📊 현재 진행률: {progress_percent:.1f}%\n"
                f"  🟢 라벨링: {labeled_count}개\n"
                f"  🟠 스킵: {skipped_count}개\n"
                f"  ⚪ 미처리: {not_processed}개\n\n"
                f"'🚀 라벨링 시작' 버튼을 눌러 이어서 작업하세요."
            )
            
        except Exception as e:
            QMessageBox.warning(self, "에러", f"로그 파일 읽기 실패:\n{str(e)}")
    
    def show_help(self):
        """사용 가이드 표시"""
        help_text = """
<h2>🏷️ Stall Labeling Tool 사용 가이드</h2>

    <h3>📋 기본 사용법</h3>
    <ol>
    <li><b>입력 폴더 선택</b>: 라벨링할 이미지가 있는 폴더</li>
    <li><b>출력 폴더 선택</b>: 결과를 저장할 상위 폴더</li>
    <li><b>폴더 이름 설정</b>: 라벨링 결과를 저장할 폴더 이름
      <ul>
      <li>기본값: labels_날짜시간 (예: labels_20250123_133052)</li>
      <li>원하는 이름으로 수정 가능 (예: my_stall_labels)</li>
      <li>🔄 버튼: 현재 날짜시간으로 새로고침</li>
      <li>이미 폴더가 있으면 그 안에 추가 저장됨</li>
      </ul>
    </li>
    <li><b>설정 조정</b> (선택사항): Simplify 강도, 홀 채우기 등</li>
    <li><b>Start Labeling 클릭</b>: 라벨링 시작</li>
    </ol>

<h3>⌨️ 라벨링 키보드 단축키</h3>
<b>모드 전환:</b>
<ul>
<li><b>1</b> - SAM 모드 (자동 인식, 추천)</li>
<li><b>2</b> - 그리기 모드 (수동 추가)</li>
<li><b>3</b> - 지우개 모드 (수동 제거)</li>
</ul>

<b>Split Mode (좌우 분리):</b>
<ul>
<li><b>Q</b> - 왼쪽 영역만 라벨링</li>
<li><b>W</b> - 오른쪽 영역만 라벨링</li>
<li><b>E</b> - 전체 영역 라벨링 (기본값)</li>
</ul>

<b>기능:</b>
<ul>
<li><b>M</b> - Mask 미리보기 (현재 색상으로)</li>
<li><b>N</b> - Binary Mask 미리보기 (흰색/검정색, 저장될 형태)</li>
<li><b>S</b> - Polygon Simplify 조절 (+/- 키로 조절)</li>
<li><b>V</b> - 배경 시각화 변경 (일반/엣지/대비)</li>
<li><b>C</b> - 오버레이 색상 변경</li>
<li><b>F</b> - 홀 채우기 ON/OFF</li>
<li><b>U 또는 Ctrl+Z</b> - 마지막 클릭 취소 (Undo)</li>
<li><b>TAB</b> - 텍스트 표시 ON/OFF</li>
</ul>

<b>진행:</b>
<ul>
<li><b>SPACE</b> - 저장하고 다음 이미지</li>
<li><b>` (백틱)</b> - 현재 이미지 건너뛰기 (탭 위의 키)</li>
<li><b>R</b> - 현재 이미지 리셋</li>
<li><b>ESC</b> - 라벨링 종료</li>
</ul>

<h3>🖱️ 마우스 조작</h3>
<ul>
<li><b>좌클릭</b> - 포함할 영역 선택 (SAM 모드)</li>
<li><b>우클릭</b> - 제외할 영역 선택 (SAM 모드)</li>
<li><b>마우스 휠</b> - 브러시 크기 / Expand 크기 조절</li>
<li><b>드래그</b> - 그리기/지우기 (모드 2, 3)</li>
</ul>

<h3>⚙️ 설정 항목</h3>
<ul>
<li><b>Polygon Simplify (0~5)</b>: 외곽선 직선화 강도
  <ul>
  <li>0 = 원본 그대로 (기본값)</li>
  <li>1-2 = 약함</li>
  <li>3-4 = 보통</li>
  <li>5 = 매우 강함</li>
  </ul>
</li>
<li><b>홀 채우기</b>: 영역 내부 작은 구멍 자동 채우기
  <ul>
  <li>크기: 이 픽셀 이하의 구멍만 채움</li>
  </ul>
</li>
<li><b>제외 영역 확장</b>: 우클릭 시 제외 영역 크기 (픽셀)</li>
</ul>

    <h3>🔄 작업 이어하기</h3>
    <p><b>"🔄 작업 이어하기"</b> 버튼을 누르면:</p>
    <ul>
    <li><b>progress_log.json</b> 파일을 선택합니다</li>
    <li>해당 파일에서 입력/출력 폴더 경로를 자동으로 읽어옵니다</li>
    <li>이전 작업 현황 (라벨링 완료, 스킵)을 확인할 수 있습니다</li>
    <li>라벨링 시작 시 이미 처리한 이미지는 자동으로 건너뛰고, 미처리 이미지만 작업합니다</li>
    <li>💡 작업 중단 후 이어하기에 유용합니다</li>
    </ul>

<h3>📊 진행 상황 보기</h3>
<ul>
<li><b>처리</b>: 현재까지 확인한 이미지 개수</li>
<li><b>라벨링</b>: 실제로 저장한 이미지 개수</li>
<li><b>스킵</b>: 건너뛴 이미지 개수 (K 키)</li>
</ul>

    <h3>📁 결과 폴더 구조</h3>
    <pre>
    출력폴더/
    └── 폴더이름/  (예: labels_20250123_133052 또는 my_labels)
        ├── json/           # Simple JSON
        ├── yolo/labels/    # YOLO Segmentation
        ├── segman/masks/   # SegMAN (학습용)
        └── visualizations/ # 시각화 이미지
    </pre>
    <p>💡 같은 폴더 이름이면 기존 파일에 추가 저장됩니다</p>

<h3>💡 팁</h3>
<ul>
<li>좌우 스톨이 겹치면 <b>Q/W 키</b>로 분리해서 라벨링</li>
<li>울퉁불퉁한 경계는 <b>S 키</b>로 Simplify 조절</li>
<li>격자 부분이 채워지면 <b>F 키</b>로 홀 채우기 OFF</li>
<li>잘못 클릭하면 <b>U 키</b>로 Undo</li>
</ul>

<h3>❓ 문제 해결</h3>
<ul>
<li><b>SAM 로딩 느림</b>: 첫 실행 시 정상 (5~10초)</li>
<li><b>메모리 부족</b>: 16GB RAM 권장</li>
<li><b>이미지 사라짐</b>: 입력 폴더와 source_image 같으면 복사 안 함</li>
</ul>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("사용 가이드")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.setStyleSheet("QLabel{min-width: 600px; min-height: 500px;}")
        msg.exec_()
    
    def closeEvent(self, event):
        """종료 시"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, '확인',
                "라벨링이 진행 중입니다. 종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 다크 모드 스타일 (선택사항)
    # app.setStyle('Fusion')
    
    window = LabelToolGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

