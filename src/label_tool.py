"""
간단한 SAM Polygon 라벨링 도구

기능:
- source_image 폴더의 이미지를 하나씩 불러오기
- SAM으로 polygon 추출
- 작은 영역 자동 제거
- 영역 내부 홀 채우기
- 1~10개 정도의 깔끔한 polygon 생성
- 학습 형식으로 자동 저장 (YOLO, SegMAN 등)

사용법:
python label_tool.py
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import json
import glob
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys

# SAM 2 import
# try:
#     from sam2.build_sam import build_sam2
#     from sam2.sam2_image_predictor import SAM2ImagePredictor
# except ImportError:
#     print("❌ SAM 2 설치 필요")
#     print("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
#     exit(1)


class SimpleLabelTool:
    """간단한 SAM 라벨링 도구 + 수동 편집"""
    
    def __del__(self):
        """소멸자 - CUDA 메모리 정리"""
        try:
            if hasattr(self, 'predictor') and self.predictor is not None:
                # SAM predictor 정리
                del self.predictor
            
            # CUDA 메모리 강제 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 CUDA 메모리 정리 완료")
        except Exception as e:
            print(f"⚠️ 메모리 정리 중 오류: {e}")
    
    def _put_korean_text(self, img, text, position, font_size=20, color=(255, 255, 255), thickness=2):
        """한글 텍스트를 이미지에 추가 (PIL 사용)"""
        try:
            # OpenCV BGR -> PIL RGB
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 폰트 로드 (여러 시도)
            font = None
            font_paths = [
                "malgun.ttf",
                "C:/Windows/Fonts/malgun.ttf",
                "C:/Windows/Fonts/gulim.ttf",
                "C:/Windows/Fonts/batang.ttf",
                "C:/Windows/Fonts/arial.ttf"
            ]
            
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
            
            # 폰트를 못 찾으면 기본 폰트 (한글 지원 안됨)
            if font is None:
                # 한글 없이 영어/숫자만 표시
                import re
                text = re.sub(r'[가-힣ㄱ-ㅎㅏ-ㅣ]', '', text)  # 한글 제거
                cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                           font_size/30, color, thickness)
                return img
            
            # 텍스트 그리기 (검은 테두리)
            x, y = position
            for dx in [-thickness, 0, thickness]:
                for dy in [-thickness, 0, thickness]:
                    if dx != 0 or dy != 0:
                        draw.text((x+dx, y+dy), text, font=font, fill=(0, 0, 0))
            
            # 텍스트 그리기 (메인 색상)
            draw.text((x, y), text, font=font, fill=color[::-1])  # BGR -> RGB
            
            # PIL RGB -> OpenCV BGR
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            # 완전 실패시 OpenCV 기본 폰트
            cv2.putText(img, str(text), position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_size/30, color, thickness)
            return img
    
    def __init__(self, checkpoint_path,config=None):
        print("🔄 SAM 2 로딩 중...")
        
        # 모델 파일명에서 config 자동 감지
        model_name = os.path.basename(checkpoint_path).lower()
        
        if 'sam2.1' in model_name:
            if 'tiny' in model_name or '_t' in model_name:
                config_file = 'configs/sam2.1/sam2.1_hiera_t.yaml'
            elif 'small' in model_name or '_s' in model_name:
                config_file = 'configs/sam2.1/sam2.1_hiera_s.yaml'
            elif 'base' in model_name or '_b' in model_name:
                config_file = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
            else:  # large 또는 기타
                config_file = 'configs/sam2.1/sam2.1_hiera_l.yaml'
        else:  # SAM 2.0
            if 'tiny' in model_name or '_t' in model_name:
                config_file = 'configs/sam2/sam2_hiera_t.yaml'
            elif 'small' in model_name or '_s' in model_name:
                config_file = 'configs/sam2/sam2_hiera_s.yaml'
            elif 'base' in model_name or '_b' in model_name:
                config_file = 'configs/sam2/sam2_hiera_b+.yaml'
            else:  # large 또는 기타
                config_file = 'configs/sam2/sam2_hiera_l.yaml'

        # SAM 2 초기화
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔧 디바이스 설정: {device}")
            print(f"🔧 모델 경로: {checkpoint_path}")
            print(f"🔧 설정 파일: {config_file}")
            
            sam2 = build_sam2(config_file, checkpoint_path, device=device)
            self.predictor = SAM2ImagePredictor(sam2)
            
            print(f"✅ 모델 준비 완료 (Device: {device})\n")
            
        except Exception as e:
            print(f"❌ SAM 모델 로딩 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            # CUDA 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
        # 클릭 상태
        self.points = []
        self.labels = []
        self.negative_mask = None  # 제외할 영역 (우클릭한 부분)
        self.current_img = None
        self.display_img = None
        self.current_mask = None
        
        # 수동 편집 상태
        self.mode = 'sam'  # 'sam', 'draw', 'erase'
        self.brush_size = 10
        self.is_drawing = False
        self.last_point = None
        self.manual_edited = False  # 그리기/지우개로 수동 편집했는지
        
        # 시각화 모드
        self.view_mode = 'normal'  # 'normal', 'edge', 'contrast'
        
        # 설정값 적용 (GUI에서 전달받음)
        if config is None:
            config = {}
        
        # 홀 채우기 설정
        self.max_hole_size = config.get('max_hole_size', 1000)
        self.fill_holes_enabled = config.get('fill_holes_enabled', True)
        
        # 네거티브 마스크 확장 설정
        self.negative_expand = config.get('negative_expand', 10)
        
        # Polygon Simplify 강도
        self.simplify_strength = config.get('simplify_strength', 0)
        
        # Split Mode (분할 모드) ⭐ NEW
        self.split_mode = 'entire'  # 'entire', 'left', 'right' (기본값: entire)
        self.split_masks = {'left': None, 'right': None}  # 좌/우 mask 저장
        
        # 텍스트 표시 ON/OFF ⭐ NEW
        self.show_text = True  # TAB 키로 토글
        
        # 오버레이 색상 (BGR)
        self.overlay_colors = {
            'green': [0, 255, 0],
            'red': [0, 0, 255],
            'blue': [255, 0, 0],
            'yellow': [0, 255, 255],
            'cyan': [255, 255, 0],
            'magenta': [255, 0, 255],
            'white': [255, 255, 255]
        }
        self.color_names = list(self.overlay_colors.keys())
        self.current_color_idx = 0  # green
        
        # 상태 파일 경로
        self.status_file = Path("labeling_status.json")
    
    def _save_status(self, current_idx=None, total=None):
        """현재 상태를 파일에 저장 (GUI와 동기화)"""
        try:
            mode_map = {
                'sam': 'SAM',
                'draw': 'DRAW',
                'erase': 'ERASE'
            }
            
            status = {
                'mode': mode_map.get(self.mode, 'SAM'),
                'split_mode': self.split_mode,
                'brush_size': self.brush_size,
                'negative_expand': self.negative_expand,
                'simplify_strength': self.simplify_strength
            }
            
            # 진행상황 추가
            if current_idx is not None:
                status['current_index'] = current_idx
            if total is not None:
                status['total_images'] = total
            
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass  # 파일 저장 실패 시 무시
    
    def _download_checkpoint(self):
        """체크포인트 다운로드"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "sam2_hiera_large.pt"
        
        if not checkpoint_path.exists():
            print("📥 모델 다운로드 중... (처음만, 약 900MB)")
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
            urllib.request.urlretrieve(url, checkpoint_path)
        
        return str(checkpoint_path)
    
    def label_image(self, img_path):
        """이미지 라벨링"""
        self.current_img = cv2.imread(str(img_path))
        if self.current_img is None:
            return None
        
        self.display_img = self.current_img.copy()
        self.points = []
        self.labels = []
        self.negative_mask = None
        self.manual_edited = False
        self.current_mask = np.zeros(self.current_img.shape[:2], dtype=np.uint8)
        self.mode = 'sam'
        self.is_drawing = False
        
        # Split Mode 초기화 ⭐ BUG FIX: 다음 이미지 시 잔상 제거
        self.split_mode = 'entire'
        self.split_masks = {'left': None, 'right': None}
        
        # SAM 설정
        self.predictor.set_image(self.current_img)
        
        # 윈도우 생성
        cv2.namedWindow("Label Tool")
        cv2.setMouseCallback("Label Tool", self._mouse_callback)
        
        print(f"📌 라벨링: {Path(img_path).name}")
        print("\n🎨 모드:")
        print("  - 1: SAM 모드 (클릭) - 자동 인식")
        print("  - 2: 그리기 모드 (드래그) - 수동 추가")
        print("  - 3: 지우개 모드 (드래그) - 수동 제거")
        print("\n👁️ 시각화:")
        print("  - V: 배경 변경 (일반/엣지/대비)")
        print(f"  - C: 색상 변경 (현재: {self.color_names[self.current_color_idx]})")
        print("  - M: Mask 미리보기 (저장 전 확인)")
        print("\n🖱️ SAM 모드:")
        print("  - 좌클릭: 포함할 영역")
        print("  - 우클릭: 제외할 영역")
        print("  - 마우스 휠: 제외 영역 확장 크기 조절 ⭐")
        print(f"  - 현재 확장: {self.negative_expand}픽셀")
        print("  - U 또는 Ctrl+Z: 마지막 클릭 취소 (Undo)")
        print("\n🖌️ 그리기/지우개 모드:")
        print("  - 드래그: 그리기/지우기")
        print("  - [/]: 브러시 크기 조절")
        print("  - 마우스 휠: 브러시 크기 조절")
        print("\n🔧 홀 채우기:")
        print("  - F: 홀 채우기 ON/OFF 토글 ⭐")
        print(f"  - 상태: {'✅ ON' if self.fill_holes_enabled else '❌ OFF'}")
        print("  - -/+: 홀 채우기 크기 조절")
        print(f"  - 현재: {self.max_hole_size} 픽셀")
        print("\n💡 팁:")
        print("  - 아래쪽 격자 부분이 채워지면 안 될 때:")
        print("    1) F 키로 홀 채우기 OFF")
        print("    2) 또는 지우개(3)로 해당 부분 제거")
        print("\n⌨️ 기타:")
        print("  - SPACE: 완료")
        print("  - R: 리셋")
        print("  - K: 건너뛰기")
        print("  - ESC: 종료\n")
        
        while True:
            cv2.imshow("Label Tool", self.display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE - 완료
                if np.any(self.current_mask > 0):
                    break
            elif key == ord('r') or key == ord('R'):  # R - 리셋
                self.points = []
                self.labels = []
                self.negative_mask = None
                self.manual_edited = False
                self.current_mask = np.zeros(self.current_img.shape[:2], dtype=np.uint8)
                self._update_display()
            elif key == ord('`'):  # ` - 건너뛰기 (탭 위의 키)
                cv2.destroyAllWindows()
                return None
            elif key == ord('s') or key == ord('S'):  # S - Simplify 강도 조절 ⭐ NEW
                self._adjust_simplify()
            elif key == 27:  # ESC - 종료 ⭐ 변경 (Q에서 ESC로)
                cv2.destroyAllWindows()
                return 'QUIT'
            elif key == ord('u') or key == ord('U') or key == 26:  # U 또는 Ctrl+Z - Undo
                if len(self.points) > 0:
                    removed_point = self.points.pop()
                    removed_label = self.labels.pop()
                    label_type = "포함(초록)" if removed_label == 1 else "제외(빨강)"
                    print(f"  ↶ Undo: 마지막 클릭 취소 ({label_type})")
                    if self.mode == 'sam':
                        self._update_with_sam()
                else:
                    print(f"  ⚠️ 취소할 클릭이 없습니다")
            elif key == ord('1'):  # SAM 모드
                self.mode = 'sam'
                print(f"  🔵 SAM 모드 (클릭)")
                self._update_display()  # 화면 업데이트
            elif key == ord('2'):  # 그리기 모드
                self.mode = 'draw'
                self.manual_edited = True  # 수동 편집 플래그
                print(f"  🟢 그리기 모드 (브러시: {self.brush_size})")
                self._update_display()  # 화면 업데이트
            elif key == ord('3'):  # 지우개 모드
                self.mode = 'erase'
                self.manual_edited = True  # 수동 편집 플래그
                print(f"  🔴 지우개 모드 (브러시: {self.brush_size})")
                self._update_display()  # 화면 업데이트
            elif key == ord('v') or key == ord('V'):  # 배경 시각화 변경
                self._toggle_view_mode()
            elif key == ord('c') or key == ord('C'):  # 색상 변경
                self._change_overlay_color()
            elif key == ord('m') or key == ord('M'):  # Mask 미리보기
                self._preview_mask()
            elif key == ord('n') or key == ord('N'):  # Mask Binary 미리보기 (흰색/검정색) ⭐
                self._preview_mask_binary()
            elif key == ord('['):  # 브러시 크기 감소
                self.brush_size = max(1, self.brush_size - 1)
                print(f"  브러시 크기: {self.brush_size}")
            elif key == ord(']'):  # 브러시 크기 증가
                self.brush_size = min(100, self.brush_size + 1)
                print(f"  브러시 크기: {self.brush_size}")
            elif key == ord('f') or key == ord('F'):  # 홀 채우기 토글
                self.fill_holes_enabled = not self.fill_holes_enabled
                status = '✅ ON' if self.fill_holes_enabled else '❌ OFF'
                print(f"  🔧 홀 채우기: {status}")
                if not self.fill_holes_enabled:
                    print(f"     → 영역 내부 구멍이 그대로 유지됩니다")
                else:
                    print(f"     → 영역 내부 {self.max_hole_size}픽셀 이하 구멍을 채웁니다")
            elif key == ord('-') or key == ord('_'):  # 홀 크기 감소
                self.max_hole_size = max(100, self.max_hole_size - 200)
                print(f"  홀 채우기 크기: {self.max_hole_size} (작은 홀만 채움)")
            elif key == ord('=') or key == ord('+'):  # 홀 크기 증가
                self.max_hole_size = min(10000, self.max_hole_size + 200)
                print(f"  홀 채우기 크기: {self.max_hole_size} (더 큰 홀도 채움)")
            elif key == ord('q') or key == ord('Q'):  # Split Mode - 왼쪽 ⭐ NEW
                # Q 영역으로 전환 전에 현재 mask 저장
                if self.split_mode == 'right' and np.any(self.current_mask > 0):
                    self.split_masks['right'] = self.current_mask.copy()
                    print(f"  💾 오른쪽 영역 저장됨")
                
                self.split_mode = 'left'
                print(f"  ⬅️ Q: 왼쪽만 라벨링")
                
                # 이전에 저장한 왼쪽 mask 불러오기
                if self.split_masks['left'] is not None:
                    self.current_mask = self.split_masks['left'].copy()
                    print(f"  [Undo] 이전 왼쪽 영역 복원")
                else:
                    self.current_mask = np.zeros(self.current_img.shape[:2], dtype=np.uint8)
                
                self._update_display()
            elif key == ord('w') or key == ord('W'):  # Split Mode - 오른쪽 ⭐ NEW
                # W 영역으로 전환 전에 현재 mask 저장
                if self.split_mode == 'left' and np.any(self.current_mask > 0):
                    self.split_masks['left'] = self.current_mask.copy()
                    print(f"  💾 왼쪽 영역 저장됨")
                
                self.split_mode = 'right'
                print(f"  ➡️ W: 오른쪽만 라벨링")
                
                # 이전에 저장한 오른쪽 mask 불러오기
                if self.split_masks['right'] is not None:
                    self.current_mask = self.split_masks['right'].copy()
                    print(f"  [Undo] 이전 오른쪽 영역 복원")
                else:
                    self.current_mask = np.zeros(self.current_img.shape[:2], dtype=np.uint8)
                
                self._update_display()
            elif key == ord('e') or key == ord('E'):  # Split Mode - 전체 ⭐ NEW
                # 현재 mask 저장
                if self.split_mode == 'left' and np.any(self.current_mask > 0):
                    self.split_masks['left'] = self.current_mask.copy()
                    print(f"  💾 왼쪽 영역 저장됨")
                elif self.split_mode == 'right' and np.any(self.current_mask > 0):
                    self.split_masks['right'] = self.current_mask.copy()
                    print(f"  💾 오른쪽 영역 저장됨")
                
                self.split_mode = 'entire'
                print(f"  🔓 E: 전체 영역 (Split OFF)")
                
                # 좌우 mask 합치기
                self.current_mask = self._merge_split_masks()
                self._update_display()
            elif key == 9:  # TAB - 텍스트 표시 ON/OFF ⭐ NEW
                self.show_text = not self.show_text
                status = 'ON' if self.show_text else 'OFF'
                print(f"  📝 텍스트 표시: {status}")
                self._update_display()
        
        cv2.destroyAllWindows()
        
        # 현재 모드의 mask 저장 (저장 전에)
        if self.split_mode == 'left' and np.any(self.current_mask > 0):
            self.split_masks['left'] = self.current_mask.copy()
        elif self.split_mode == 'right' and np.any(self.current_mask > 0):
            self.split_masks['right'] = self.current_mask.copy()
        
        # 최종 mask: 좌우 합치기 ⭐
        mask = self._merge_split_masks()
        
        if not np.any(mask > 0):
            return None
        
        # 1. 노이즈 제거 (작은 파편) ⭐
        mask = self._remove_noise(mask, min_size=100)
        
        # 2. 홀 채우기 적용
        if self.fill_holes_enabled:
            mask = self._fill_holes(mask, self.max_hole_size)
        
        # 4. 네거티브 마스크를 홀로 적용 (SAM만 사용하고 수동 편집 안 한 경우)
        if self.negative_mask is not None and not self.manual_edited:
            mask[self.negative_mask > 0] = 0
        
        # 5. 다시 노이즈 제거 (홀 적용 후 생긴 작은 파편) ⭐
        mask = self._remove_noise(mask, min_size=100)
        
        # Polygon 추출 (홀 처리된 mask로부터)
        polygons = self._mask_to_polygons(
            mask,
            min_area=500,                    # 최소 영역 크기
            fill_holes=False,                # 이미 홀 채우기 완료
            max_hole_size=self.max_hole_size,
            max_contours=10                  # 최대 contour 수
        )
        
        # polygons와 최종 mask를 함께 반환 ⭐
        return {'polygons': polygons, 'mask': mask}
    
    def _mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백"""
        if self.mode == 'sam':
            # SAM 모드 - 클릭
            if event == cv2.EVENT_LBUTTONDOWN:  # 좌클릭
                self.points.append([x, y])
                self.labels.append(1)  # foreground
                self._update_with_sam()
                
            elif event == cv2.EVENT_RBUTTONDOWN:  # 우클릭
                self.points.append([x, y])
                self.labels.append(0)  # background
                self._update_with_sam()
        
        elif self.mode in ['draw', 'erase']:
            # 그리기/지우기 모드 - 드래그
            if event == cv2.EVENT_LBUTTONDOWN:
                self.is_drawing = True
                self.last_point = (x, y)
                self._draw_at(x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.is_drawing:
                    if self.last_point:
                        self._draw_line(self.last_point, (x, y))
                    self.last_point = (x, y)
                else:
                    # 브러시 커서 표시
                    self._show_brush_cursor(x, y)
                    
            elif event == cv2.EVENT_LBUTTONUP:
                self.is_drawing = False
                self.last_point = None
                self._update_display()
        
        # 마우스 휠
        if event == cv2.EVENT_MOUSEWHEEL:
            if self.mode == 'sam':
                # SAM 모드: 네거티브 마스크 확장 크기 조절
                if flags > 0:  # 위로
                    self.negative_expand = min(50, self.negative_expand + 3)
                else:  # 아래로
                    self.negative_expand = max(0, self.negative_expand - 3)
                print(f"  제외 영역 확장: {self.negative_expand}픽셀")
                self._save_status()  # GUI 동기화
                # SAM 재업데이트
                if len(self.points) > 0:
                    self._update_with_sam()
            else:
                # 그리기/지우개 모드: 브러시 크기 조절
                if flags > 0:  # 위로
                    self.brush_size = min(100, self.brush_size + 1)
                else:  # 아래로
                    self.brush_size = max(1, self.brush_size - 1)
                print(f"  브러시 크기: {self.brush_size}")
                self._save_status()  # GUI 동기화
    
    def _draw_at(self, x, y):
        """특정 위치에 그리기/지우기"""
        if self.mode == 'draw':
            cv2.circle(self.current_mask, (x, y), self.brush_size, 255, -1)
        elif self.mode == 'erase':
            cv2.circle(self.current_mask, (x, y), self.brush_size, 0, -1)
        
        self._update_display()
    
    def _draw_line(self, pt1, pt2):
        """두 점 사이에 선 그리기"""
        if self.mode == 'draw':
            cv2.line(self.current_mask, pt1, pt2, 255, self.brush_size * 2)
        elif self.mode == 'erase':
            cv2.line(self.current_mask, pt1, pt2, 0, self.brush_size * 2)
    
    def _show_brush_cursor(self, x, y):
        """브러시 커서 표시"""
        temp_img = self.display_img.copy()
        color = (0, 255, 0) if self.mode == 'draw' else (0, 0, 255)
        cv2.circle(temp_img, (x, y), self.brush_size, color, 2)
        cv2.imshow("Label Tool", temp_img)
    
    def _update_with_sam(self):
        """SAM으로 mask 업데이트"""
        if len(self.points) == 0:
            return
        
        mask = self._generate_mask()
        if mask is not None:
            # 네거티브 마스크 먼저 적용 (우클릭 영역 제외)
            self._update_negative_mask()
            
            if self.negative_mask is not None:
                # 우클릭한 영역을 mask에서 제거
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(self.negative_mask))
            
            # Connected Components로 분리된 영역만 유지
            mask = self._filter_connected_components(mask)
            
            # Split Mode 적용 ⭐ NEW
            if self.split_mode is not None:
                mask = self._apply_split_mode(mask)
            
            self.current_mask = mask
            self._update_display()
    
    def _apply_split_mode(self, mask):
        """Split Mode에 따라 mask를 절반으로 제한 ⭐ NEW"""
        if self.split_mode == 'entire':
            return mask
        
        h, w = mask.shape
        mid_x = w // 2
        
        if self.split_mode == 'left':
            # 오른쪽 절반을 0으로
            mask[:, mid_x:] = 0
        elif self.split_mode == 'right':
            # 왼쪽 절반을 0으로
            mask[:, :mid_x] = 0
        
        return mask
    
    def _merge_split_masks(self):
        """좌우 split mask를 합치기 ⭐ NEW"""
        # 현재 mask 포함
        if self.split_mode == 'entire':
            merged = self.current_mask.copy()
        elif self.split_mode == 'left':
            merged = self.current_mask.copy()
        elif self.split_mode == 'right':
            merged = self.current_mask.copy()
        else:
            merged = np.zeros(self.current_img.shape[:2], dtype=np.uint8)
        
        # 저장된 좌우 mask 합치기
        if self.split_masks['left'] is not None:
            merged = cv2.bitwise_or(merged, self.split_masks['left'])
        
        if self.split_masks['right'] is not None:
            merged = cv2.bitwise_or(merged, self.split_masks['right'])
        
        return merged
    
    def _filter_connected_components(self, mask):
        """좌클릭이 포함된 connected component만 유지"""
        # 좌클릭(foreground) 포인트만 추출
        positive_points = [p for p, l in zip(self.points, self.labels) if l == 1]
        
        if len(positive_points) == 0:
            return mask
        
        # Connected components 찾기
        num_labels, labels = cv2.connectedComponents(mask)
        
        # 결과 mask
        filtered_mask = np.zeros_like(mask)
        
        # 각 좌클릭 포인트가 포함된 component 찾기
        selected_labels = set()
        for point in positive_points:
            x, y = point
            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                label = labels[y, x]
                if label > 0:  # 0은 배경
                    selected_labels.add(label)
        
        # 선택된 component만 유지
        for label in selected_labels:
            filtered_mask[labels == label] = 255
        
        return filtered_mask
    
    def _update_negative_mask(self):
        """우클릭(제외)한 영역을 별도 마스크로 저장 (확장 적용) - Split Mode 고려 ⭐"""
        # 우클릭(label=0)한 포인트만 추출
        negative_points = [p for p, l in zip(self.points, self.labels) if l == 0]
        
        if len(negative_points) == 0:
            self.negative_mask = None
            return
        
        # 각 네거티브 포인트 주변을 마스크로 저장
        h, w = self.current_mask.shape
        negative_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 각 네거티브 포인트에 대해 큰 원으로 직접 표시 (더 확실함) ⭐
        base_radius = 30  # 기본 반경
        for point in negative_points:
            # SAM으로 영역 찾기 시도
            try:
                input_points = np.array([point])
                input_labels = np.array([1])
                
                masks, _, _ = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=False
                )
                
                temp_mask = (masks[0] * 255).astype(np.uint8)
                negative_mask = cv2.bitwise_or(negative_mask, temp_mask)
            except:
                pass
            
            # 큰 원을 추가로 그려서 확실히 제거 ⭐
            total_radius = base_radius + self.negative_expand
            cv2.circle(negative_mask, tuple(point), total_radius, 255, -1)
        
        # 추가 확장 적용
        if self.negative_expand > 0:
            kernel_size = max(3, int(self.negative_expand * 0.5))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            negative_mask = cv2.dilate(negative_mask, kernel, iterations=1)
        
        # Split Mode 적용 ⭐ BUG FIX: 네거티브 마스크도 split 영역에만 적용
        negative_mask = self._apply_split_mode(negative_mask)
        
        self.negative_mask = negative_mask
    
    def _update_display(self):
        """디스플레이 업데이트"""
        # 시각화 모드에 따라 배경 선택
        if self.view_mode == 'edge':
            base_img = self._get_edge_image()
        elif self.view_mode == 'contrast':
            base_img = self._get_contrast_image()
        else:
            base_img = self.current_img.copy()
        
        # Mask 오버레이 (현재 색상 사용)
        current_color = self.overlay_colors[self.color_names[self.current_color_idx]]
        overlay = base_img.copy()
        overlay[self.current_mask > 0] = current_color
        self.display_img = cv2.addWeighted(base_img, 0.7, overlay, 0.3, 0)
        
        # Split Mode: 비활성 영역 반투명 표시 ⭐ NEW
        if self.split_mode in ['left', 'right']:
            h, w = self.display_img.shape[:2]
            mid_x = w // 2
            dark_overlay = self.display_img.copy()
            
            if self.split_mode == 'left':
                # 오른쪽 어둡게
                dark_overlay[:, mid_x:] = dark_overlay[:, mid_x:] // 2
                # 중앙 선 표시
                cv2.line(self.display_img, (mid_x, 0), (mid_x, h), (255, 255, 0), 2)
            elif self.split_mode == 'right':
                # 왼쪽 어둡게
                dark_overlay[:, :mid_x] = dark_overlay[:, :mid_x] // 2
                # 중앙 선 표시
                cv2.line(self.display_img, (mid_x, 0), (mid_x, h), (255, 255, 0), 2)
            
            self.display_img = dark_overlay
        
        # SAM 모드: 포인트 표시
        if self.mode == 'sam':
            for point, label in zip(self.points, self.labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(self.display_img, tuple(point), 5, color, -1)
            
            # 네거티브 마스크 영역 반투명 표시
            if self.negative_mask is not None:
                neg_overlay = self.display_img.copy()
                neg_overlay[self.negative_mask > 0] = [0, 0, 255]  # 빨강
                self.display_img = cv2.addWeighted(self.display_img, 0.85, neg_overlay, 0.15, 0)
        
        # 텍스트 표시 (TAB 키로 ON/OFF) ⭐ NEW
        if self.show_text:
            # 모드 표시 (영어로 변경 - 이모지는 유지)
            mode_text = {
                'sam': f'SAM (Expand:{self.negative_expand})',
                'draw': f'Draw ({self.brush_size})',
                'erase': f'Erase ({self.brush_size})'
            }
            cv2.putText(self.display_img, mode_text[self.mode], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Split Mode 표시
            split_text = {
                'entire': 'ENTIRE (E)',
                'left': 'LEFT (Q)',
                'right': 'RIGHT (W)'
            }
            cv2.putText(self.display_img, split_text[self.split_mode], (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 상태 저장 (GUI 동기화)
        self._save_status()
    
    def _generate_mask_preview(self):
        """마스크 미리보기 생성 (토글용)"""
        if self.split_mode == 'left' and np.any(self.current_mask > 0):
            self.split_masks['left'] = self.current_mask.copy()
        elif self.split_mode == 'right' and np.any(self.current_mask > 0):
            self.split_masks['right'] = self.current_mask.copy()
        
        merged_mask = self._merge_split_masks()
        if not np.any(merged_mask > 0):
            print("  ⚠️ 선택된 영역이 없습니다")
            return
        
        # 최종 mask 생성
        preview_mask = merged_mask.copy()
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        if self.fill_holes_enabled:
            preview_mask = self._fill_holes(preview_mask, self.max_hole_size)
        
        if self.negative_mask is not None and not self.manual_edited:
            preview_mask[self.negative_mask > 0] = 0
        
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # 시각화
        preview_img = self.current_img.copy()
        overlay = np.zeros_like(preview_img)
        overlay[preview_mask > 0] = [0, 255, 0]  # 초록색
        self.display_img = cv2.addWeighted(preview_img, 0.7, overlay, 0.3, 0)
    
    def _generate_binary_preview(self):
        """바이너리 마스크 미리보기 생성 (토글용)"""
        if self.split_mode == 'left' and np.any(self.current_mask > 0):
            self.split_masks['left'] = self.current_mask.copy()
        elif self.split_mode == 'right' and np.any(self.current_mask > 0):
            self.split_masks['right'] = self.current_mask.copy()
        
        merged_mask = self._merge_split_masks()
        if not np.any(merged_mask > 0):
            print("  ⚠️ 선택된 영역이 없습니다")
            return
        
        # 최종 mask 생성
        preview_mask = merged_mask.copy()
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        if self.fill_holes_enabled:
            preview_mask = self._fill_holes(preview_mask, self.max_hole_size)
        
        if self.negative_mask is not None and not self.manual_edited:
            preview_mask[self.negative_mask > 0] = 0
        
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # 바이너리 이미지로 변환 (흑백)
        binary_img = np.zeros_like(self.current_img)
        binary_img[preview_mask > 0] = [255, 255, 255]  # 흰색
        self.display_img = binary_img
    
    def _generate_simplify_preview(self):
        """심플리파이 미리보기 생성 (토글용)"""
        if self.split_mode == 'left' and np.any(self.current_mask > 0):
            self.split_masks['left'] = self.current_mask.copy()
        elif self.split_mode == 'right' and np.any(self.current_mask > 0):
            self.split_masks['right'] = self.current_mask.copy()
        
        merged_mask = self._merge_split_masks()
        if not np.any(merged_mask > 0):
            print("  ⚠️ 선택된 영역이 없습니다")
            return
        
        # 최종 mask 생성
        preview_mask = merged_mask.copy()
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        if self.fill_holes_enabled:
            preview_mask = self._fill_holes(preview_mask, self.max_hole_size)
        
        if self.negative_mask is not None and not self.manual_edited:
            preview_mask[self.negative_mask > 0] = 0
        
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # Polygon 추출 및 시각화
        contours, hierarchy = cv2.findContours(preview_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        preview_img = self.current_img.copy()
        colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 0, 255]]
        
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] == -1 and cv2.contourArea(contours[idx]) >= 100:  # 외부 윤곽
                    epsilon_factors = [0.001, 0.002, 0.004, 0.006, 0.008, 0.010]
                    epsilon_factor = epsilon_factors[min(5, max(0, self.simplify_strength))]
                    epsilon = epsilon_factor * cv2.arcLength(contours[idx], True)
                    approx = cv2.approxPolyDP(contours[idx], epsilon, True)
                    
                    color = colors[idx % len(colors)]
                    cv2.drawContours(preview_img, [approx], -1, color, 3)
                    
                    # 포인트 개수 표시
                    M = cv2.moments(approx)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.putText(preview_img, f"{len(approx)}pts", (cx-30, cy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        self.display_img = preview_img
    
    def _generate_mask_preview_gui(self):
        """GUI용 Mask 미리보기 생성 (토글용) - cv2.imshow 없이"""
        # 현재 모드의 mask 임시 저장
        if self.split_mode == 'left' and np.any(self.current_mask > 0):
            self.split_masks['left'] = self.current_mask.copy()
        elif self.split_mode == 'right' and np.any(self.current_mask > 0):
            self.split_masks['right'] = self.current_mask.copy()
        
        # 전체 영역(좌우 합친 것) 생성
        merged_mask = self._merge_split_masks()
        
        if not np.any(merged_mask > 0):
            print("  ⚠️ 선택된 영역이 없습니다")
            return
        
        # 최종 mask 생성
        preview_mask = merged_mask.copy()
        
        # 1. 노이즈 제거
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # 2. 홀 채우기
        if self.fill_holes_enabled:
            preview_mask = self._fill_holes(preview_mask, self.max_hole_size)
        
        # 3. 네거티브 마스크 적용
        if self.negative_mask is not None and not self.manual_edited:
            preview_mask[self.negative_mask > 0] = 0
        
        # 4. 다시 노이즈 제거
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # RETR_CCOMP로 계층 구조 분석
        contours, hierarchy = cv2.findContours(preview_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 외부 contour만 선택
        external_contours = []
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] == -1:  # parent == -1: 외부 윤곽
                    if cv2.contourArea(contours[idx]) >= 100:
                        external_contours.append(contours[idx])
        else:
            external_contours = [c for c in contours if cv2.contourArea(c) >= 100]
        
        # 홀 개수 세기
        hole_count = 0
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] != -1:  # parent != -1: 홀
                    hole_count += 1
                    

        # 시각화
        preview_img = self.current_img.copy()
        
        # 현재 선택된 색상 사용
        color = self.overlay_colors[self.color_names[self.current_color_idx]]
        
        # 외부 contour 그리기
        for idx, contour in enumerate(external_contours):
            overlay = preview_img.copy()
            cv2.drawContours(overlay, [contour], -1, color, -1)
            preview_img = cv2.addWeighted(preview_img, 0.7, overlay, 0.3, 0)
            
            # 번호 표시 (TAB 설정 반영)
            if self.show_text:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(preview_img, f"#{idx+1}", (cx-20, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        # 내부 홀 표시
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] != -1:  # parent != -1: 홀
                    if cv2.contourArea(contours[idx]) >= 100:
                        cv2.drawContours(preview_img, contours, idx, (0, 0, 255), 2)
        
        # 정보 텍스트 (TAB 설정 반영)
        if self.show_text:
            fill_status = "ON" if self.fill_holes_enabled else "OFF"
            info_text = f"Mask Preview | Contours: {len(external_contours)} | Fill: {fill_status}"
            if self.fill_holes_enabled:
                info_text += f" ({self.max_hole_size})"
            if hole_count > 0:
                info_text += f" | Holes: {hole_count}"
            info_text += f" | Simplify: {self.simplify_strength}"
            cv2.putText(preview_img, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return_text = "Press M to return"
            cv2.putText(preview_img, return_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # display_img 설정 (GUI에서 표시용)
        self.display_img = preview_img
        
        print(f"\n  👀 Mask Preview")
        print(f"     - Contours: {len(external_contours)}")
        if hole_count > 0:
            print(f"     - Holes: {hole_count}")
        fill_status = '✅ ON' if self.fill_holes_enabled else '❌ OFF'
        print(f"     - Fill: {fill_status}", end='')
        if self.fill_holes_enabled:
            print(f" ({self.max_hole_size} px)")
        else:
            print(f" (Keep holes)")
        print(f"     - Simplify: {self.simplify_strength}")
    
    def _generate_binary_preview_gui(self):
        """GUI용 Binary Mask 미리보기 생성 (토글용) - cv2.imshow 없이"""
        # 현재 모드의 mask 임시 저장
        if self.split_mode == 'left' and np.any(self.current_mask > 0):
            self.split_masks['left'] = self.current_mask.copy()
        elif self.split_mode == 'right' and np.any(self.current_mask > 0):
            self.split_masks['right'] = self.current_mask.copy()
        
        # 전체 영역(좌우 합친 것) 생성
        merged_mask = self._merge_split_masks()
        
        if not np.any(merged_mask > 0):
            print("  ⚠️ 선택된 영역이 없습니다")
            return
        
        # 최종 mask 생성 (동일한 로직)
        preview_mask = merged_mask.copy()
        
        # 1. 노이즈 제거
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # 2. 홀 채우기
        if self.fill_holes_enabled:
            preview_mask = self._fill_holes(preview_mask, self.max_hole_size)
        
        # 3. 네거티브 마스크 적용
        if self.negative_mask is not None and not self.manual_edited:
            preview_mask[self.negative_mask > 0] = 0
        
        # 4. 다시 노이즈 제거
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # Binary 마스크 생성 (흰색/검정색)
        binary_mask = np.zeros_like(preview_mask)
        binary_mask[preview_mask > 0] = 255
        
        # 3채널로 변환 (시각화용)
        binary_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        
        # Contour 정보 계산
        contours, hierarchy = cv2.findContours(preview_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 외부 contour 개수
        external_count = 0
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] == -1:  # parent == -1: 외부 윤곽
                    if cv2.contourArea(contours[idx]) >= 100:
                        external_count += 1
        else:
            external_count = len([c for c in contours if cv2.contourArea(c) >= 100])
        
        # 홀 개수
        hole_count = 0
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] != -1:  # parent != -1: 홀
                    hole_count += 1
        
        # 정보 텍스트 (TAB 설정 반영)
        if self.show_text:
            info_lines = [
                f"Binary Mask Preview (White/Black)",
                f"Contours: {external_count} | Holes: {hole_count}",
                f"Fill: {'ON' if self.fill_holes_enabled else 'OFF'}",
                f"Press N to return"
            ]
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(binary_img, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 35
        
        # display_img 설정 (GUI에서 표시용)
        self.display_img = binary_img
        
        print(f"\n  🖼️ Binary Mask Preview (저장될 형태)")
        print(f"     - Contours: {external_count}")
        if hole_count > 0:
            print(f"     - Holes: {hole_count}")
        fill_status = '✅ ON' if self.fill_holes_enabled else '❌ OFF'
        print(f"     - Fill: {fill_status}")
    
    def _generate_simplify_preview_gui(self):
        """GUI용 Simplify 미리보기 생성 (토글용) - cv2.imshow 없이"""
        # 현재 모드의 mask 임시 저장
        if self.split_mode == 'left' and np.any(self.current_mask > 0):
            self.split_masks['left'] = self.current_mask.copy()
        elif self.split_mode == 'right' and np.any(self.current_mask > 0):
            self.split_masks['right'] = self.current_mask.copy()
        
        # 전체 영역(좌우 합친 것) 생성
        merged_mask = self._merge_split_masks()
        
        if not np.any(merged_mask > 0):
            print("  ⚠️ 선택된 영역이 없습니다")
            return
        
        # 최종 mask 생성
        preview_mask = merged_mask.copy()
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        if self.fill_holes_enabled:
            preview_mask = self._fill_holes(preview_mask, self.max_hole_size)
        
        if self.negative_mask is not None and not self.manual_edited:
            preview_mask[self.negative_mask > 0] = 0
        
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # Polygon 추출 및 시각화
        contours, hierarchy = cv2.findContours(preview_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        preview_img = self.current_img.copy()
        colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 0, 255]]
        
        total_points = 0
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] == -1 and cv2.contourArea(contours[idx]) >= 100:  # 외부 윤곽
                    epsilon_factors = [0.001, 0.002, 0.004, 0.006, 0.008, 0.010]
                    epsilon_factor = epsilon_factors[min(5, max(0, self.simplify_strength))]
                    epsilon = epsilon_factor * cv2.arcLength(contours[idx], True)
                    approx = cv2.approxPolyDP(contours[idx], epsilon, True)
                    
                    color = colors[idx % len(colors)]
                    cv2.drawContours(preview_img, [approx], -1, color, 3)
                    
                    # 포인트 개수 표시
                    M = cv2.moments(approx)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.putText(preview_img, f"{len(approx)}pts", (cx-30, cy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    total_points += len(approx)
        
        # 정보 텍스트
        if self.show_text:
            info_lines = [
                f"Simplify Preview | Strength: {self.simplify_strength}",
                f"Total Points: {total_points}",
                f"Press S to return | +/- to adjust"
            ]
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(preview_img, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 35
        
        # display_img 설정 (GUI에서 표시용)
        self.display_img = preview_img
        
        print(f"\n  🔧 Simplify Preview")
        print(f"     - Strength: {self.simplify_strength}")
        print(f"     - Total Points: {total_points}")
    
    def _toggle_view_mode(self):
        """배경 시각화 모드 변경"""
        modes = ['normal', 'edge', 'contrast']
        idx = modes.index(self.view_mode)
        self.view_mode = modes[(idx + 1) % len(modes)]
        
        mode_names = {
            'normal': '일반',
            'edge': '엣지',
            'contrast': '대비'
        }
        print(f"  👁️ 배경: {mode_names[self.view_mode]}")
        self._update_display()
    
    def _change_overlay_color(self):
        """오버레이 색상 변경"""
        self.current_color_idx = (self.current_color_idx + 1) % len(self.color_names)
        color_name = self.color_names[self.current_color_idx]
        
        color_emoji = {
            'green': '🟢',
            'red': '🔴',
            'blue': '🔵',
            'yellow': '🟡',
            'cyan': '🔵',
            'magenta': '🟣',
            'white': '⚪'
        }
        
        print(f"  🎨 색상: {color_emoji.get(color_name, '⚪')} {color_name}")
        self._update_display()
    
    def _adjust_simplify(self):
        """Simplify 강도 인터랙티브 조절 (마우스 휠) ⭐⭐⭐"""
        # 현재 모드의 mask 임시 저장
        if self.split_mode == 'left' and np.any(self.current_mask > 0):
            self.split_masks['left'] = self.current_mask.copy()
        elif self.split_mode == 'right' and np.any(self.current_mask > 0):
            self.split_masks['right'] = self.current_mask.copy()
        
        # 전체 영역(좌우 합친 것) 생성
        merged_mask = self._merge_split_masks()
        
        if not np.any(merged_mask > 0):
            print("  ⚠️ 선택된 영역이 없습니다")
            return
        
        # 미리보기는 라벨링 영역에서 표시 (새 창 없음)
        
        # 마우스 휠 콜백용 변수
        preview_state = {
            'simplify_strength': self.simplify_strength,
            'update': True
        }
        
        # 키보드 입력으로 조절 (+ / - 키)
        
        print(f"\n  🔧 Simplify Adjustment (Interactive)")
        print(f"     - +/- Keys: Adjust strength (Current: {preview_state['simplify_strength']})")
        print(f"     - ESC: Save & Exit")
        print(f"     - Other Key: Cancel\n")
        
        while True:
            if preview_state['update']:
                # 최종 mask 생성
                preview_mask = merged_mask.copy()
                
                # 1. 노이즈 제거
                preview_mask = self._remove_noise(preview_mask, min_size=100)
                
                # 2. 홀 채우기
                if self.fill_holes_enabled:
                    preview_mask = self._fill_holes(preview_mask, self.max_hole_size)
                
                # 3. 네거티브 마스크 적용
                if self.negative_mask is not None and not self.manual_edited:
                    preview_mask[self.negative_mask > 0] = 0
                
                # 4. 다시 노이즈 제거
                preview_mask = self._remove_noise(preview_mask, min_size=100)
                
                # 5. Polygon 추출 (Simplify 강도 적용) ⭐⭐⭐
                # 임시로 simplify_strength 변경
                original_strength = self.simplify_strength
                self.simplify_strength = preview_state['simplify_strength']
                
                # RETR_CCOMP로 계층 구조 분석
                contours, hierarchy = cv2.findContours(preview_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                
                # 외부 contour만 선택하고 simplify 적용
                external_contours = []
                simplified_contours = []
                if hierarchy is not None:
                    for idx, h in enumerate(hierarchy[0]):
                        if h[3] == -1:  # parent == -1: 외부 윤곽
                            if cv2.contourArea(contours[idx]) >= 100:
                                external_contours.append(contours[idx])
                                
                                # Polygon simplification ⭐
                                epsilon_factors = [0.001, 0.002, 0.004, 0.006, 0.008, 0.010]
                                epsilon_factor = epsilon_factors[min(5, max(0, self.simplify_strength))]
                                epsilon = epsilon_factor * cv2.arcLength(contours[idx], True)
                                approx = cv2.approxPolyDP(contours[idx], epsilon, True)
                                simplified_contours.append(approx)
                else:
                    external_contours = [c for c in contours if cv2.contourArea(c) >= 100]
                    for contour in external_contours:
                        epsilon_factors = [0.001, 0.002, 0.004, 0.006, 0.008, 0.010]
                        epsilon_factor = epsilon_factors[min(5, max(0, self.simplify_strength))]
                        epsilon = epsilon_factor * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        simplified_contours.append(approx)
                
                # 원래 강도로 복원
                self.simplify_strength = original_strength
                
                # 홀 개수 세기
                hole_count = 0
                if hierarchy is not None:
                    for idx, h in enumerate(hierarchy[0]):
                        if h[3] != -1:  # parent != -1: 홀
                            hole_count += 1
                
                # 시각화
                preview_img = self.current_img.copy()
                
                # Contour별 다른 색상
                colors = [
                    [0, 255, 0],    # 초록
                    [0, 0, 255],    # 빨강
                    [255, 0, 0],    # 파랑
                    [0, 255, 255],  # 노랑
                    [255, 0, 255],  # 마젠타
                ]
                
                # Simplified contour 그리기 (외곽선만)
                for idx, contour in enumerate(simplified_contours):
                    color = colors[idx % len(colors)]
                    cv2.drawContours(preview_img, [contour], -1, color, 3)
                    
                    # 번호 표시
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.putText(preview_img, f"#{idx+1}", (cx-20, cy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                        # Point 개수 표시
                        point_count = len(contour)
                        cv2.putText(preview_img, f"{point_count}pts", (cx-30, cy+30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 내부 홀 표시
                if hierarchy is not None:
                    for idx, h in enumerate(hierarchy[0]):
                        if h[3] != -1:  # parent != -1: 홀
                            if cv2.contourArea(contours[idx]) >= 100:
                                cv2.drawContours(preview_img, contours, idx, (0, 0, 255), 2)
                
                # 정보 텍스트
                total_points = sum(len(c) for c in simplified_contours)
                info_lines = [
                    f"Simplify: {preview_state['simplify_strength']} (Wheel to adjust)",
                    f"Contours: {len(simplified_contours)} | Total Points: {total_points}",
                    f"Holes: {hole_count}",
                    f"ESC: Save / Other: Cancel"
                ]
                
                y_offset = 30
                for line in info_lines:
                    cv2.putText(preview_img, line, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(preview_img, line, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    y_offset += 35
                
                # 라벨링 영역에 미리보기 표시 (새 창 없음)
                self.display_img = preview_img.copy()
                preview_state['update'] = False
                
                # PyQt5 환경에서는 즉시 종료 (키 입력은 GUI에서 처리)
                break
            
            # 이 부분은 더 이상 사용되지 않음 (PyQt5에서 키 입력 처리)
            key = cv2.waitKey(50) & 0xFF
            
            if key == 27:  # ESC - 강도 저장하고 종료
                self.simplify_strength = preview_state['simplify_strength']
                print(f"  ✅ Saved: Simplify={self.simplify_strength}")
                break
            # +/- 키 기능 제거 (PyQt GUI에서는 마우스 휠로 처리)
            # elif key == ord('+') or key == ord('='):  # + 키 - 강도 증가
            #     preview_state['simplify_strength'] = min(5, preview_state['simplify_strength'] + 1)
            #     preview_state['update'] = True
            #     print(f"  🔧 Simplify: {preview_state['simplify_strength']} (0=Original, 5=Max)")
            # elif key == ord('-') or key == ord('_'):  # - 키 - 강도 감소
            #     preview_state['simplify_strength'] = max(0, preview_state['simplify_strength'] - 1)
            #     preview_state['update'] = True
            #     print(f"  🔧 Simplify: {preview_state['simplify_strength']} (0=Original, 5=Max)")
            elif key != 255:  # 다른 키 - 강도 되돌리고 종료
                print(f"  ❌ Cancelled (Keep: Simplify={self.simplify_strength})")
                break
        
        # 원래 화면으로 복귀
        self._update_display()
    
    def _preview_mask(self):
        """Mask 미리보기 - 전체 영역 (단순 미리보기) ⭐"""
        # 현재 모드의 mask 임시 저장
        if self.split_mode == 'left' and np.any(self.current_mask > 0):
            self.split_masks['left'] = self.current_mask.copy()
        elif self.split_mode == 'right' and np.any(self.current_mask > 0):
            self.split_masks['right'] = self.current_mask.copy()
        
        # 전체 영역(좌우 합친 것) 생성 ⭐
        merged_mask = self._merge_split_masks()
        
        if not np.any(merged_mask > 0):
            print("  ⚠️ 선택된 영역이 없습니다")
            return
        
        # 최종 mask 생성
        preview_mask = merged_mask.copy()
        
        # 1. 노이즈 제거
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # 2. 홀 채우기
        if self.fill_holes_enabled:
            preview_mask = self._fill_holes(preview_mask, self.max_hole_size)
        
        # 3. 네거티브 마스크 적용
        if self.negative_mask is not None and not self.manual_edited:
            preview_mask[self.negative_mask > 0] = 0
        
        # 4. 다시 노이즈 제거
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # RETR_CCOMP로 계층 구조 분석 ⭐
        contours, hierarchy = cv2.findContours(preview_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 외부 contour만 선택
        external_contours = []
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] == -1:  # parent == -1: 외부 윤곽
                    if cv2.contourArea(contours[idx]) >= 100:
                        external_contours.append(contours[idx])
        else:
            external_contours = [c for c in contours if cv2.contourArea(c) >= 100]
        
        # 홀 개수 세기
        hole_count = 0
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] != -1:  # parent != -1: 홀
                    hole_count += 1
        
        # 시각화
        preview_img = self.current_img.copy()
        
        # 현재 선택된 색상 사용 ⭐
        color = self.overlay_colors[self.color_names[self.current_color_idx]]
        
        # 외부 contour 그리기
        for idx, contour in enumerate(external_contours):
            overlay = preview_img.copy()
            cv2.drawContours(overlay, [contour], -1, color, -1)
            preview_img = cv2.addWeighted(preview_img, 0.7, overlay, 0.3, 0)
            
            # 번호 표시 (TAB 설정 반영) ⭐
            if self.show_text:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(preview_img, f"#{idx+1}", (cx-20, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        # 내부 홀 표시
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] != -1:  # parent != -1: 홀
                    if cv2.contourArea(contours[idx]) >= 100:
                        cv2.drawContours(preview_img, contours, idx, (0, 0, 255), 2)
        
        # 정보 텍스트 (TAB 설정 반영) ⭐
        if self.show_text:
            fill_status = "ON" if self.fill_holes_enabled else "OFF"
            info_text = f"Contours: {len(external_contours)} | Fill: {fill_status}"
            if self.fill_holes_enabled:
                info_text += f" ({self.max_hole_size})"
            if hole_count > 0:
                info_text += f" | Holes: {hole_count}"
            info_text += f" | Simplify: {self.simplify_strength}"
            cv2.putText(preview_img, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return_text = "Press any key to return"
            cv2.putText(preview_img, return_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 미리보기 표시
        cv2.imshow("Label Tool", preview_img)
        print(f"\n  👀 Mask Preview")
        print(f"     - Contours: {len(external_contours)}")
        if hole_count > 0:
            print(f"     - Holes: {hole_count}")
        fill_status = '✅ ON' if self.fill_holes_enabled else '❌ OFF'
        print(f"     - Fill: {fill_status}", end='')
        if self.fill_holes_enabled:
            print(f" ({self.max_hole_size} px)")
        else:
            print(f" (Keep holes)")
        print(f"     - Simplify: {self.simplify_strength} (S key to adjust)\n")
        
        cv2.waitKey(0)
        
        # 원래 화면으로 복귀
        self._update_display()
        cv2.imshow("Label Tool", self.display_img)
    
    def _preview_mask_binary(self):
        """Mask Binary 미리보기 - 저장될 흰색/검정색 마스크 확인 ⭐"""
        # 현재 모드의 mask 임시 저장
        if self.split_mode == 'left' and np.any(self.current_mask > 0):
            self.split_masks['left'] = self.current_mask.copy()
        elif self.split_mode == 'right' and np.any(self.current_mask > 0):
            self.split_masks['right'] = self.current_mask.copy()
        
        # 전체 영역(좌우 합친 것) 생성
        merged_mask = self._merge_split_masks()
        
        if not np.any(merged_mask > 0):
            print("  ⚠️ 선택된 영역이 없습니다")
            return
        
        # 최종 mask 생성 (동일한 로직)
        preview_mask = merged_mask.copy()
        
        # 1. 노이즈 제거
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # 2. 홀 채우기
        if self.fill_holes_enabled:
            preview_mask = self._fill_holes(preview_mask, self.max_hole_size)
        
        # 3. 네거티브 마스크 적용
        if self.negative_mask is not None and not self.manual_edited:
            preview_mask[self.negative_mask > 0] = 0
        
        # 4. 다시 노이즈 제거
        preview_mask = self._remove_noise(preview_mask, min_size=100)
        
        # Binary 마스크 생성 (흰색/검정색) ⭐
        binary_mask = np.zeros_like(preview_mask)
        binary_mask[preview_mask > 0] = 255
        
        # 3채널로 변환 (시각화용)
        binary_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        
        # Contour 정보 계산
        contours, hierarchy = cv2.findContours(preview_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 외부 contour 개수
        external_count = 0
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] == -1:  # parent == -1: 외부 윤곽
                    if cv2.contourArea(contours[idx]) >= 100:
                        external_count += 1
        else:
            external_count = len([c for c in contours if cv2.contourArea(c) >= 100])
        
        # 홀 개수
        hole_count = 0
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                if h[3] != -1:  # parent != -1: 홀
                    hole_count += 1
        
        # 정보 텍스트 (TAB 설정 반영) ⭐
        if self.show_text:
            info_lines = [
                f"Binary Mask Preview (White/Black)",
                f"Contours: {external_count} | Holes: {hole_count}",
                f"Fill: {'ON' if self.fill_holes_enabled else 'OFF'}",
                f"Press any key to return"
            ]
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(binary_img, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 35
        
        # 미리보기 표시
        cv2.imshow("Label Tool", binary_img)
        print(f"\n  🖼️ Binary Mask Preview (저장될 형태)")
        print(f"     - Contours: {external_count}")
        if hole_count > 0:
            print(f"     - Holes: {hole_count}")
        fill_status = '✅ ON' if self.fill_holes_enabled else '❌ OFF'
        print(f"     - Fill: {fill_status}\n")
        
        cv2.waitKey(0)
        
        # 원래 화면으로 복귀
        self._update_display()
        cv2.imshow("Label Tool", self.display_img)
    
    def _get_edge_image(self):
        """엣지 강조 이미지"""
        gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 원본과 엣지 합성
        result = cv2.addWeighted(self.current_img, 0.7, edges_colored, 0.3, 0)
        return result
    
    def _get_contrast_image(self):
        """대비 강조 이미지"""
        lab = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE로 명암 강조
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return result
    
    def _generate_mask(self):
        """현재 포인트로 mask 생성"""
        if len(self.points) == 0:
            return None
        
        try:
            input_points = np.array(self.points)
            input_labels = np.array(self.labels)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )
            
            return (masks[0] * 255).astype(np.uint8)
        except:
            return None
    
    def _mask_to_polygons(self, mask, min_area=500, fill_holes=True, max_hole_size=1000, max_contours=10):
        """
        Mask를 polygon으로 변환 (홀 기반 제외 방식) ⭐
        
        Args:
            min_area: 최소 영역 크기 (작은 영역 제거)
            fill_holes: 영역 내부 홀 채우기
            max_hole_size: 이 크기보다 작은 홀만 채움 (큰 구멍 유지)
            max_contours: 최대 contour 수
        """
        # 작은 홀만 채우기 (큰 구멍은 유지)
        if fill_holes:
            mask = self._fill_holes(mask, max_hole_size)
        
        # 네거티브 마스크를 홀로 적용 ⭐ (SAM만 사용하고 수동 편집 안 한 경우)
        if self.negative_mask is not None and not self.manual_edited:
            # 기존: bitwise_and로 제거 → 작은 연결 부분이 남음
            # 개선: 네거티브 영역을 mask에서 0으로 만들기 (홀로 만들기)
            mask[self.negative_mask > 0] = 0
        
        # RETR_CCOMP로 계층 구조 분석 ⭐
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 외부 contour만 선택 (내부 홀 제외) ⭐
        external_contours = []
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                # h = [next, prev, child, parent]
                # parent == -1: 외부 윤곽
                if h[3] == -1:
                    external_contours.append(contours[idx])
        else:
            external_contours = contours
        
        # 외부 contour만 면적으로 정렬 (큰 것부터)
        contours = sorted(external_contours, key=cv2.contourArea, reverse=True)
        
        polygons = []
        
        for contour in contours[:max_contours]:
            area = cv2.contourArea(contour)
            
            # 작은 영역 제거
            if area < min_area:
                continue
            
            # Polygon 단순화 (직선/곡선으로 부드럽게) ⭐ S 키로 조절 가능
            # simplify_strength: 0~5
            epsilon_factors = [0.001, 0.002, 0.004, 0.006, 0.008, 0.010]  # 강도별 epsilon
            epsilon_factor = epsilon_factors[min(5, max(0, self.simplify_strength))]
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.reshape(-1, 2).tolist()
            
            # 중심점
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = polygon[0]
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            polygons.append({
                'polygon': polygon,
                'area': float(area),
                'center': [cx, cy],
                'bbox': [x, y, w, h],
                'num_points': len(polygon)
            })
        
        return polygons
    
    def _remove_noise(self, mask, min_size=100):
        """
        작은 노이즈(파편) 제거 ⭐
        
        Args:
            min_size: 이 크기보다 작은 영역 제거 (픽셀)
        """
        # Connected components 분석
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 새로운 mask 생성
        cleaned = np.zeros_like(mask)
        
        # 각 component를 확인 (0은 배경이므로 1부터 시작)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            
            # 충분히 큰 영역만 유지
            if area >= min_size:
                cleaned[labels == label_id] = 255
        
        return cleaned
    
    def _fill_holes(self, mask, max_hole_size=1000):
        """
        각 영역 내부의 작은 홀만 채우기 (영역 사이 공간은 유지)
        
        Args:
            max_hole_size: 이 크기보다 작은 홀만 채움 (픽셀)
        """
        # 외부 contour 찾기
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None:
            return mask
        
        filled = np.zeros_like(mask)
        
        # 각 contour별로 처리
        for idx, h in enumerate(hierarchy[0]):
            # h = [next, prev, child, parent]
            # parent == -1: 최상위 contour (외부 윤곽)
            # parent != -1: 내부 홀
            
            if h[3] == -1:  # 외부 윤곽
                # 외부 윤곽 그리기
                cv2.drawContours(filled, contours, idx, 255, -1)
                
                # 이 외부 윤곽의 자식(홀) 찾기
                child_idx = h[2]
                while child_idx != -1:
                    hole_area = cv2.contourArea(contours[child_idx])
                    
                    # 큰 홀은 유지 (지우기)
                    if hole_area >= max_hole_size:
                        cv2.drawContours(filled, contours, child_idx, 0, -1)
                    # 작은 홀은 채움 (아무것도 안 함, 이미 외부 윤곽으로 채워짐)
                    
                    # 다음 형제 홀로 이동
                    child_idx = hierarchy[0][child_idx][0]
        
        return filled


def save_labels(img_path, polygons, output_folder, mask=None):
    """
    라벨 저장 (다양한 형식)
    
    Args:
        img_path: 원본 이미지 경로
        polygons: polygon 리스트
        output_folder: 출력 폴더
        mask: 최종 처리된 mask (홀 포함) ⭐ NEW
    """
    output_path = Path(output_folder)
    img_name = Path(img_path).stem
    
    # 1. JSON 형식 (Simple)
    json_path = output_path / 'json'
    json_path.mkdir(exist_ok=True)
    
    json_data = {
        'filename': Path(img_path).name,
        'image_path': str(img_path),
        'polygons': polygons
    }
    
    with open(json_path / f"{img_name}.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # 2. YOLO Segmentation 형식
    yolo_path = output_path / 'yolo' / 'labels'
    yolo_path.mkdir(parents=True, exist_ok=True)
    
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    yolo_lines = []
    for polygon_data in polygons:
        polygon = polygon_data['polygon']
        
        # 정규화 (0~1)
        normalized = []
        for x, y in polygon:
            normalized.extend([x/w, y/h])
        
        # YOLO format: class_id x1 y1 x2 y2 ...
        line = "0 " + " ".join([f"{coord:.6f}" for coord in normalized])
        yolo_lines.append(line)
    
    with open(yolo_path / f"{img_name}.txt", 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    # 3. SegMAN 형식 (Segmentation Mask)
    segman_path = output_path / 'segman' / 'masks'
    segman_path.mkdir(parents=True, exist_ok=True)
    
    # mask가 전달되면 사용 (홀 포함), 없으면 fillPoly로 생성 ⭐
    if mask is None:
        mask = np.zeros((h, w), dtype=np.uint8)
        for polygon_data in polygons:
            polygon = polygon_data['polygon']
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)  # class_id = 1
    else:
        # 전달받은 mask를 class_id=1로 변환 (0=배경, 1=전경)
        mask = (mask > 0).astype(np.uint8)
    
    cv2.imwrite(str(segman_path / f"{img_name}.png"), mask)
    
    # 시각적 확인용 mask (픽셀값 255로 스케일링)
    mask_vis = (mask > 0).astype(np.uint8) * 255
    mask_vis_path = output_path / 'segman' / 'masks_visualized'
    mask_vis_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_vis_path / f"{img_name}.png"), mask_vis)
    
    # 4. 시각화 (mask 직접 사용) ⭐
    vis_path = output_path / 'visualizations'
    vis_path.mkdir(exist_ok=True)
    
    vis_img = img.copy()
    overlay = img.copy()
    
    # mask를 직접 사용 (polygon 대신) ⭐
    if mask is not None:
        # RETR_CCOMP로 계층 구조 분석
        mask_255 = (mask * 255).astype(np.uint8)
        contours_all, hierarchy = cv2.findContours(mask_255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is not None:
            # 외부 contour (전경) - 초록색
            for idx, h in enumerate(hierarchy[0]):
                if h[3] == -1:  # parent == -1: 외부 윤곽
                    # 채우기 (랜덤 색상)
                    color = tuple(np.random.randint(100, 255, 3).tolist())
                    cv2.drawContours(overlay, contours_all, idx, color, -1)
                    # 윤곽선 (초록색)
                    cv2.drawContours(vis_img, contours_all, idx, (0, 255, 0), 2)
            
            # 내부 홀 (제외영역) - 빨간색 ⭐
            hole_count = 0
            for idx, h in enumerate(hierarchy[0]):
                if h[3] != -1:  # parent != -1: 내부 홀
                    if cv2.contourArea(contours_all[idx]) >= 100:
                        # 홀을 반투명 빨간색으로 표시
                        cv2.drawContours(overlay, contours_all, idx, (0, 0, 200), -1)
                        # 홀 윤곽선 (빨간색)
                        cv2.drawContours(vis_img, contours_all, idx, (0, 0, 255), 2)
                        hole_count += 1
    else:
        # mask가 없으면 polygon 사용 (하위 호환성)
        for idx, polygon_data in enumerate(polygons):
            polygon = polygon_data['polygon']
            pts = np.array(polygon, dtype=np.int32)
            color = tuple(np.random.randint(50, 255, 3).tolist())
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
    
    # Polygon 정보 표시 (중심점, 번호)
    for idx, polygon_data in enumerate(polygons):
        cx, cy = polygon_data['center']
        cv2.circle(vis_img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(vis_img, f"#{idx+1}", (cx-10, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    result = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
    
    # 범례 추가
    cv2.putText(result, "Green: Foreground", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(result, "Red: Excluded (Holes)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imwrite(str(vis_path / f"{img_name}.jpg"), result)


def load_progress_log(output_folder):
    """작업 로그 로드"""
    log_file = output_folder / "progress_log.json"
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {'labeled': [], 'skipped': []}

def save_progress_log(output_folder, labeled, skipped):
    """작업 로그 저장 (폴더 경로 및 전체 개수 유지)"""
    log_file = output_folder / "progress_log.json"
    try:
        # 기존 로그의 폴더 경로 및 전체 개수 읽기
        input_folder = None
        output_folder_path = None
        total_images = None
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
                input_folder = existing_log.get('input_folder')
                output_folder_path = existing_log.get('output_folder')
                total_images = existing_log.get('total_images')
            except Exception as e:
                print(f"⚠️ 기존 로그 읽기 실패: {e}")
        
        # 중복 제거 ⭐
        labeled_unique = list(dict.fromkeys(labeled))  # 순서 유지하며 중복 제거
        skipped_unique = list(dict.fromkeys(skipped))  # 순서 유지하며 중복 제거
        
        log_data = {
            'labeled': labeled_unique,
            'skipped': skipped_unique,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 폴더 경로 및 전체 개수 추가 (있으면)
        if input_folder:
            log_data['input_folder'] = input_folder
        if output_folder_path:
            log_data['output_folder'] = output_folder_path
        if total_images:
            log_data['total_images'] = total_images
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        # 디버깅: 저장 확인
        if len(labeled) != len(labeled_unique) or len(skipped) != len(skipped_unique):
            print(f"⚠️ 중복 제거: 라벨링 {len(labeled)}→{len(labeled_unique)}개, 스킵 {len(skipped)}→{len(skipped_unique)}개")
        print(f"📝 progress_log.json 업데이트: 라벨링 {len(labeled_unique)}개, 스킵 {len(skipped_unique)}개")
    except Exception as e:
        print(f"❌ progress_log.json 저장 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("="*60)
    print("🏷️ SAM Polygon 라벨링 도구")
    print("="*60)
    print()
    
    # 설정 로드 (config.json에서)
    config = {}
    config_file = Path("config.json")
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"⚙️ 설정 로드: {config_file}")
            print(f"  - Simplify: {config.get('simplify_strength', 0)}")
            print(f"  - 홀 채우기: {config.get('fill_holes_enabled', True)} ({config.get('max_hole_size', 1000)}px)")
            print(f"  - 제외 영역 확장: {config.get('negative_expand', 10)}px")
            print()
        except:
            print("⚠️ config.json 로드 실패, 기본값 사용\n")
    

    model = os.environ.get('LABEL_TOOL_MODEL', 'models/sam2_hiera_large.pt')
    model_path = Path(model)
    # 입력 폴더 확인 (환경변수에서 읽기) ⭐
    input_folder_str = os.environ.get('LABEL_TOOL_INPUT_FOLDER', 'source_image')
    source_folder = Path(input_folder_str)
    
    print(f"🔍 환경변수 확인: LABEL_TOOL_INPUT_FOLDER = {input_folder_str}")
    print(f"🔍 폴더 존재 여부: {source_folder.exists()}")
    print(f"📂 입력 폴더: {source_folder.resolve()}")
    print(f"모델 경로: {model_path.exists()}")

    if not model_path.exists():
        print(f"❌ 모델이 없습니다: {source_folder}")
        print(f"   모델 경로를 확인해주세요.")
        return
    
    if not source_folder.exists():
        print(f"❌ 입력 폴더가 없습니다: {source_folder}")
        print(f"   폴더를 생성하고 이미지를 넣어주세요.")
        return
    
    # 이미지 파일 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.JPEG']:
        image_files.extend(glob.glob(str(source_folder / ext)))
    
    if not image_files:
        print(f"❌ 입력 폴더에 이미지가 없습니다: {source_folder}")
        return
    
    # 출력 폴더 (환경변수에서 읽기, 없으면 기본값)
    output_folder_str = os.environ.get('LABEL_TOOL_OUTPUT_FOLDER', 'labels')
    output_folder = Path(output_folder_str)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"📁 출력 폴더: {output_folder.resolve()}")
    
    # 작업 로그 로드
    progress_log = load_progress_log(output_folder)
    processed_files = set(progress_log['labeled'] + progress_log['skipped'])
    
    # 미처리 이미지만 필터링
    remaining_files = [f for f in image_files if Path(f).name not in processed_files]
    total_images = len(image_files)  # 전체 이미지 개수 저장 ⭐
    
    if len(processed_files) > 0:
        print(f"📋 이전 작업 로그 발견")
        print(f"  - 전체: {total_images}개")
        print(f"  - 라벨링 완료: {len(progress_log['labeled'])}개")
        print(f"  - 스킵: {len(progress_log['skipped'])}개")
        print(f"  - 미처리: {len(remaining_files)}개")
        
        # 디버깅: 처리된 파일 목록 확인 ⭐
        if len(remaining_files) < 5:
            print(f"  - 미처리 파일: {[Path(f).name for f in remaining_files]}")
        
        print()
        
        if len(remaining_files) == 0:
            print("✅ 모든 이미지 처리 완료!")
            print(f"   라벨링: {len(progress_log['labeled'])}개")
            print(f"   스킵: {len(progress_log['skipped'])}개")
            print(f"   합계: {len(processed_files)}개")
            return
        
        print(f"🔄 이어서 작업합니다: {len(remaining_files)}개\n")
        image_files = remaining_files
    else:
        print(f"📸 {total_images}개 이미지 발견\n")
    
    # 라벨링 도구 초기화 (설정 전달)
    tool = SimpleLabelTool(model_path,config)
    
    # 이미지 하나씩 처리
    labeled_count = 0
    skipped_count = 0
    labeled_list = list(progress_log['labeled'])  # 복사본
    skipped_list = list(progress_log['skipped'])  # 복사본
    
    for idx, img_path in enumerate(image_files, 1):
        # 루프 시작 시 이미 완료되었는지 체크 ⭐
        current_processed = len(labeled_list) + len(skipped_list)
        if current_processed >= total_images:
            print(f"\n✅ 모든 이미지 처리 완료! ({current_processed}/{total_images})")
            break
        
        img_name = Path(img_path).name
        print(f"\n[{idx}/{len(image_files)}] {img_name}")
        
        # 진행상황 저장 (GUI 동기화) - 전체 이미지 개수와 현재 처리된 개수 전달 ⭐
        tool._save_status(current_idx=current_processed, total=total_images)
        
        # 라벨링
        result = tool.label_image(img_path)
        
        if result == 'QUIT':
            print("\n⚠️ 중단됨")
            # 작업 로그 저장 (현재까지만)
            save_progress_log(output_folder, labeled_list, skipped_list)
            # GUI 상태 업데이트 ⭐
            current_processed = len(labeled_list) + len(skipped_list)
            tool._save_status(current_idx=current_processed, total=total_images)
            # 남은 이미지는 미처리로 남김
            skipped_count = 0  # 중단 시에는 스킵 카운트 초기화
            break
        
        if result is None:
            print("  >> 건너뛰기")
            skipped_count += 1
            # 중복 방지 ⭐
            if img_name not in skipped_list:
                skipped_list.append(img_name)
            # 작업 로그 저장
            save_progress_log(output_folder, labeled_list, skipped_list)
            # GUI 상태 업데이트 ⭐
            current_processed = len(labeled_list) + len(skipped_list)
            tool._save_status(current_idx=current_processed, total=total_images)
            
            # 마지막 이미지 체크 (즉시 종료용) ⭐
            if current_processed >= total_images:
                print(f"\n🎉 모든 이미지 처리 완료! (마지막 이미지 스킵) - {current_processed}/{total_images}")
                break
            continue
        
        # 저장 (polygons와 mask 전달) ⭐
        polygons = result['polygons']
        mask = result['mask']
        save_labels(img_path, polygons, output_folder, mask=mask)
        
        print(f"  ✅ 완료: {len(polygons)}개 polygon")
        for p in polygons:
            print(f"    - 면적: {p['area']:.0f}, 점: {p['num_points']}개")
        
        labeled_count += 1
        # 중복 방지 ⭐
        if img_name not in labeled_list:
            labeled_list.append(img_name)
        # 작업 로그 저장
        save_progress_log(output_folder, labeled_list, skipped_list)
        # GUI 상태 업데이트 ⭐
        current_processed = len(labeled_list) + len(skipped_list)
        tool._save_status(current_idx=current_processed, total=total_images)
        
        # 마지막 이미지 체크 (즉시 종료용) ⭐
        if current_processed >= total_images:
            print(f"\n🎉 모든 이미지 처리 완료! (마지막 이미지 라벨링) - {current_processed}/{total_images}")
            break
    
    print("\n" + "="*60)
    print(f"✅ 라벨링 완료!")
    print(f"   이번 세션: {len(image_files)}개 처리 (라벨링: {labeled_count}개 | 스킵: {skipped_count}개)")
    print(f"   전체 통계:")
    print(f"     - 라벨링 완료: {len(labeled_list)}개")
    print(f"     - 스킵: {len(skipped_list)}개")
    print(f"     - 합계: {len(labeled_list) + len(skipped_list)}개 / {total_images}개")
    print("="*60)
    print(f"\n📁 결과:")
    print(f"  - JSON: {output_folder}/json/")
    print(f"  - YOLO: {output_folder}/yolo/labels/")
    print(f"  - SegMAN: {output_folder}/segman/masks/ (학습용)")
    print(f"  - SegMAN 시각화: {output_folder}/segman/masks_visualized/ (확인용)")
    print(f"  - 시각화: labels/visualizations/")
    print()


if __name__ == "__main__":
    try:
        print("🚀 label_tool.py 시작")
        print(f"🔍 Python 경로: {sys.executable}")
        print(f"🔍 작업 디렉토리: {os.getcwd()}")
        print(f"🔍 환경변수 INPUT: {os.environ.get('LABEL_TOOL_INPUT_FOLDER', 'NOT SET')}")
        print(f"🔍 환경변수 OUTPUT: {os.environ.get('LABEL_TOOL_OUTPUT_FOLDER', 'NOT SET')}")
        
        main()
        
    except Exception as e:
        print(f"❌ label_tool.py 에러: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

