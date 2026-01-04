import cv2
import numpy as np
import pyautogui
import time
import os
from PIL import Image, ImageDraw, ImageFont

# ======================== 參數優化（遠距離 + 15FPS + 快速點擊 + 邊緣增強） ========================
CAMERA_INDEX = 0
WINDOW_NAME = "Projection Game (遠距離優化 + 15FPS + 快速點擊 + 邊緣增強)"
SCREEN_W, SCREEN_H = pyautogui.size()

# 1. 黑色物體參數
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 80])

# 2. 物體檢測參數（移除細長閾值）
MIN_AREA = 30
MAX_AREA = 5000

# 3. 跟蹤參數（降低点击延迟帧数）
MAX_DISAPPEARED = 8
AUTO_RESET_FRAMES = 15
CLICK_DELAY_FRAMES = 0  # 从1改为0，检测到物体立即点击
MATCH_DISTANCE = 80
# 新增：短按持续时长参数（核心修改）
SHORT_PRESS_DURATION = 0.2  # 短按保持0.2秒后松开

# 4. 帧率控制参数
TARGET_FPS = 15
FRAME_DELAY = 1.0 / TARGET_FPS
CAMERA_BUFFER_SIZE = 1

# 5. 中文字體路徑
COMMON_CHINESE_FONTS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc"
]
FONT_SIZE = 22

# 6. 邊緣增強參數（新增）
EDGE_CANNY_THRESH1 = 50    # Canny邊緣檢測低閾值
EDGE_CANNY_THRESH2 = 150   # Canny邊緣檢測高閾值
EDGE_DILATE_KERNEL = (3, 3)# 邊緣膨脹核大小
EDGE_ERODE_KERNEL = (2, 2) # 邊緣腐蝕核大小
# =========================================================

# 物體跟蹤類（不變）
class ObjectTracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}  # {ID: (最下端點, 首次出現幀號, 連續檢測幀數)}
        self.disappeared = {}
        self.processed = set()

    def register(self, bottommost, current_frame):
        self.objects[self.next_id] = (bottommost, current_frame, 1)
        self.disappeared[self.next_id] = 0
        return self.next_id - 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.processed:
            self.processed.remove(object_id)

    def update(self, new_bottommost_list, current_frame):
        if len(new_bottommost_list) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > MAX_DISAPPEARED:
                    self.deregister(object_id)
            return self.objects

        matched = [-1] * len(new_bottommost_list)

        for object_id in list(self.objects.keys()):
            old_bottommost, _, _ = self.objects[object_id]
            min_dist = float('inf')
            min_idx = -1

            for i, new_bottommost in enumerate(new_bottommost_list):
                if matched[i] == -1:
                    dist = np.hypot(
                        old_bottommost[0] - new_bottommost[0],
                        old_bottommost[1] - new_bottommost[1]
                    )
                    if dist < min_dist and dist < MATCH_DISTANCE:
                        min_dist = dist
                        min_idx = i

            if min_idx != -1:
                matched[min_idx] = object_id
                _, first_frame, frame_count = self.objects[object_id]
                self.objects[object_id] = (new_bottommost_list[min_idx], first_frame, frame_count + 1)
                self.disappeared[object_id] = 0

        for i in range(len(new_bottommost_list)):
            if matched[i] == -1:
                self.register(new_bottommost_list[i], current_frame)

        for object_id in list(self.objects.keys()):
            if object_id not in matched:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > MAX_DISAPPEARED:
                    self.deregister(object_id)

        return self.objects

    def reset(self):
        self.next_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.processed.clear()
        print("🔄 自動重置跟蹤（連續無物體）")


class ProjectionGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        # 設置攝像機分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # 减少摄像头缓冲区
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        if not self.cap.isOpened():
            print("❌ 無法開啟攝影機，請檢查連線。")
            exit(1)

        # 驗證實際分辨率
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"📹 攝像機分辨率：{actual_w:.0f}x{actual_h:.0f}")

        # 加載中文字體
        self.font = None
        for font_path in COMMON_CHINESE_FONTS:
            if os.path.exists(font_path):
                try:
                    self.font = ImageFont.truetype(font_path, FONT_SIZE)
                    print(f"✅ 成功加載中文字體：{os.path.basename(font_path)}")
                    break
                except Exception as e:
                    continue
        if self.font is None:
            self.font = ImageFont.load_default()
            print("⚠️ 未找到中文字體，可能顯示異常")

        # 核心參數（移除細長閾值相關變量）
        self.background = None
        self.calib_points = []
        self.screen_points = [(0,0), (SCREEN_W,0), (SCREEN_W,SCREEN_H), (0,SCREEN_H)]
        self.H_matrix = None
        self.is_calibrated = False
        
        self.sensitivity = 3
        self.click_enabled = False  # 点击功能开关（快捷键改为L）
        self.lower_black = LOWER_BLACK
        self.upper_black = UPPER_BLACK
        self.tracker = ObjectTracker()
        self.frame_counter = 0
        self.no_object_counter = 0

        # 保留亮度、對比度参数
        self.brightness = 0
        self.contrast = 1.0

        # 帧率控制变量
        self.last_frame_time = time.time()
        self.current_fps = 0
        self.fps_update_interval = 1.0
        self.fps_frame_count = 0
        self.fps_last_update = time.time()

        # 快速点击优化：大幅降低背景重置延迟
        self.background_reset_delay = 0.05  # 从0.2秒改为0.05秒
        self.reset_background_after_click = True

        # ========== 新增：点击冷却时间控制 ==========
        self.click_cooldown = 1.0  # 点击后必须等待1秒才能再次点击
        self.last_click_timestamp = 0.0  # 记录最后一次点击的时间戳

    # 格式轉換與中文繪製（不變）
    def cv2_to_pil(self, cv_img):
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    def pil_to_cv2(self, pil_img):
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def draw_chinese_text(self, cv_img, text, pos, color=(0, 255, 0)):
        try:
            pil_img = self.cv2_to_pil(cv_img)
            draw = ImageDraw.Draw(pil_img)
            pil_color = (color[2], color[1], color[0])
            draw.text(pos, text, font=self.font, fill=pil_color)
            return self.pil_to_cv2(pil_img)
        except Exception as e:
            cv2.putText(cv_img, text[:4] + "...", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return cv_img

    # 校準功能（不變，僅修改提示文字）
    def calibrate(self):
        self.calib_points = []
        self.H_matrix = None
        self.is_calibrated = False
        print("\n📌 校準說明：請按順序點擊4個角點（遠距離時建議包含更大區域）")
        print("   1. 左上 → 2. 右上 → 3. 右下 → 4. 左下")
        print("   按 ESC 取消校準")

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            # 帧率控制
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if elapsed < FRAME_DELAY:
                time.sleep(FRAME_DELAY - elapsed)
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            frame = self.adjust_brightness_contrast(frame)
            frame = self.draw_calib_guide(frame)
            cv2.imshow(WINDOW_NAME, frame)

            self.last_frame_time = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("校準取消")
                return False
            if len(self.calib_points) == 4:
                src = np.array(self.calib_points, dtype=np.float32)
                dst = np.array(self.screen_points, dtype=np.float32)
                self.H_matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                
                if self.H_matrix is not None:
                    self.is_calibrated = True
                    ret, self.background = self.cap.read()
                    self.background = self.adjust_brightness_contrast(self.background)
                    self.background = cv2.GaussianBlur(self.background, (5, 5), 0)
                    print("✅ 校準完成！已捕獲背景（遠距離優化 + 邊緣增強）")
                    print(f"提示1：檢測「黑色」的物體，最小面積{MIN_AREA*self.sensitivity}")
                    print(f"提示2：按 'q' 降低敏感度（檢測更小物體），按 'w' 提高敏感度")
                    return True
                else:
                    print("❌ 校準失敗：無法計算坐標映射矩陣")
                    return False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.calib_points) < 4:
            self.calib_points.append((x, y))
            print(f"校準點 {len(self.calib_points)}/4: ({x}, {y})")

    def draw_calib_guide(self, frame):
        for i, (x, y) in enumerate(self.calib_points):
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
            cv2.circle(frame, (x, y), 6, colors[i], -1)
            frame = self.draw_chinese_text(frame, f"{i+1}", (x+10, y-10), colors[i])
        
        if len(self.calib_points) >= 2:
            pts = np.array(self.calib_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=(len(self.calib_points)==4), 
                         color=(255, 255, 0), thickness=2)
        
        if len(self.calib_points) < 4:
            steps = ["左上", "右上", "右下", "左下"]
            frame = self.draw_chinese_text(frame, f"請點擊{steps[len(self.calib_points)]}", 
                                          (30, 30), (0, 255, 255))
        return frame

    # 坐標映射（不變）
    def cam_to_screen(self, cam_x, cam_y):
        if self.H_matrix is None:
            return None
        point = np.array([[[cam_x, cam_y]]], dtype=np.float32)
        screen_point = cv2.perspectiveTransform(point, self.H_matrix)
        x = int(screen_point[0][0][0])
        y = int(screen_point[0][0][1])
        x = np.clip(x, 0, SCREEN_W-1)
        y = np.clip(y, 0, SCREEN_H-1)
        return (x, y)

    # 亮度和對比度調整（不變）
    def adjust_brightness_contrast(self, frame):
        adjusted = cv2.addWeighted(frame, self.contrast, np.zeros_like(frame), 0, self.brightness)
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    # 黑色掩码提取（新增邊緣增強邏輯，不變）
    def get_black_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        
        # ========== 邊緣增強核心修改 ==========
        # 1. 提取灰度圖用於邊緣檢測
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. 高斯模糊降噪（避免雜訊干擾邊緣檢測）
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # 3. Canny邊緣檢測
        edges = cv2.Canny(gray_blur, EDGE_CANNY_THRESH1, EDGE_CANNY_THRESH2)
        # 4. 膨脹邊緣（讓邊緣更明顯）
        edge_dilate_kernel = np.ones(EDGE_DILATE_KERNEL, np.uint8)
        edges_dilated = cv2.dilate(edges, edge_dilate_kernel, iterations=1)
        # 5. 將邊緣與原掩码融合（增強掩码的邊緣）
        mask_with_edges = cv2.bitwise_or(mask, edges_dilated)
        # 6. 優化形態學操作（先腐蝕去雜點，再閉運算填充空洞）
        edge_erode_kernel = np.ones(EDGE_ERODE_KERNEL, np.uint8)
        mask_with_edges = cv2.erode(mask_with_edges, edge_erode_kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        mask_with_edges = cv2.morphologyEx(mask_with_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_with_edges = cv2.morphologyEx(mask_with_edges, cv2.MORPH_DILATE, kernel, iterations=2)
        # =====================================
        
        return mask_with_edges

    # 物體有效性判斷（移除长宽比判断）
    def is_valid_object(self, contour):
        area = cv2.contourArea(contour)
        adjusted_min_area = MIN_AREA * self.sensitivity
        adjusted_max_area = MAX_AREA * self.sensitivity
        # 只保留面积判断，移除长宽比判断
        return adjusted_min_area < area < adjusted_max_area

    # 自動重置檢查（不變）
    def check_auto_reset(self, tracked_objects, new_bottommost_list):
        if len(tracked_objects) == 0 and len(new_bottommost_list) == 0:
            self.no_object_counter += 1
            if self.no_object_counter >= AUTO_RESET_FRAMES:
                self.tracker.reset()
                self.no_object_counter = 0
        else:
            self.no_object_counter = 0

    # 重置背景方法（优化：减少读取帧数，不變）
    def reset_background(self):
        """快速重置背景"""
        if self.is_calibrated:
            # 减少读取帧数，加快重置
            for _ in range(1):  # 从3帧改为1帧
                ret, temp_frame = self.cap.read()
                if not ret or temp_frame is None:
                    print("⚠️ 重置背景失敗：無法讀取攝像機畫面")
                    return False
            
            self.background = self.adjust_brightness_contrast(temp_frame)
            self.background = cv2.GaussianBlur(self.background, (5, 5), 0)
            print("🔄 背景已快速重置（邊緣增強模式）")
            return True
        return False

    # ========== 主循環（快速點擊 + 邊緣增強優化 + 点击冷却 + 下方目标优先） ==========
    def run(self):
        if not self.calibrate():
            return

        # 核心优化：设置pyautogui无延迟
        pyautogui.PAUSE = 0.0  # 从0.1改为0，取消点击延迟
        pyautogui.MINIMUM_DURATION = 0.0  # 最小点击时长设为0
        pyautogui.MINIMUM_SLEEP = 0.0     # 点击间隔设为0

        print("\n--- 控制說明（遠距離優化 + 15FPS + 快速點擊 + 邊緣增強 + 1秒点击冷却 + 下方目标优先） ---")
        print("l: 開啟/關閉點擊功能 (當前: 關閉)")
        print("q/w: 調整敏感度 (1-20) → 1=最小面積30，20=最小面積600（遠距離建議1-5）")
        print("z/x: 調整黑色檢測閾值（z更嚴格/x更寬鬆，遠距離建議按x）")
        print("e/r: 調整亮度（e增加/r降低，範圍-100至100）")
        print("f/g: 調整對比度（f增加/g降低，範圍0.1至3.0）")
        print("p: 重新校準 | b: 手動重置背景 | ESC: 退出")
        print("備註1：點擊延遲已降至0，檢測到物體立即點擊")
        print("備註2：背景重置延遲從0.2秒降至0.05秒，響應更快")
        print("備註3：已啟用Canny邊緣檢測增強，物體邊緣檢測更精准")
        print("備註4：已移除細長閾值限制，任意形狀黑色物體均可被檢測")
        print("備註5：已添加1秒点击冷却，点击后必须等待1秒才能再次点击")
        print("備註6：同時檢測多目標時，優先選擇最下方（y坐標最大）的目標")
        print("備註7：短按保持0.2秒后松开，适配游戏点击识别")

        while True:
            # 帧率控制
            current_time = time.time()
            elapsed_since_last_frame = current_time - self.last_frame_time
            if elapsed_since_last_frame < FRAME_DELAY:
                time.sleep(FRAME_DELAY - elapsed_since_last_frame)
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            # 更新FPS统计
            self.update_fps()

            # 圖像預處理
            frame = self.adjust_brightness_contrast(frame)
            self.frame_counter += 1
            frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)

            # 1. 提取黑色掩码（已包含邊緣增強）
            black_mask = self.get_black_mask(frame_blur)
            
            # 2. 運動檢測
            gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (25, 25), 0)
            background_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
            background_gray = cv2.GaussianBlur(background_gray, (25, 25), 0)
            diff_frame = cv2.absdiff(background_gray, gray)
            thresh_diff = cv2.threshold(diff_frame, 20, 255, cv2.THRESH_BINARY)[1]
            thresh_diff = cv2.dilate(thresh_diff, None, iterations=3)

            # 3. 聯合掩码
            combined_mask = cv2.bitwise_and(thresh_diff, thresh_diff, mask=black_mask)

            # 4. 校準區域過濾
            if self.is_calibrated:
                mask = np.zeros_like(gray)
                pts = np.array(self.calib_points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
                combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=mask)

            # 5. 輪廓提取（優化輪廓逼近精度 + 下方目标优先）
            contours, _ = cv2.findContours(combined_mask.copy(),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 第一步：收集所有有效物体的bottommost点
            all_valid_bottommost = []
            for contour in contours:
                if self.is_valid_object(contour):
                    # 優化輪廓逼近（減少輪廓點數，提升精度）
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)
                    contour_points = contour.reshape(-1, 2)
                    max_y_index = np.argmax(contour_points[:, 1])
                    bottommost = (int(contour_points[max_y_index][0]), int(contour_points[max_y_index][1]))
                    all_valid_bottommost.append((bottommost, contour))

            # 第二步：筛选出最下方的目标（y坐标最大）
            new_bottommost_list = []
            selected_contour = None
            if len(all_valid_bottommost) > 0:
                # 按y坐标降序排序，取第一个（最下方）
                all_valid_bottommost.sort(key=lambda x: x[0][1], reverse=True)
                selected_bottommost, selected_contour = all_valid_bottommost[0]
                new_bottommost_list = [selected_bottommost]  # 只保留最下方的目标

                # 绘制所有有效目标（区分选中/未选中）
                for i, (bottommost, contour) in enumerate(all_valid_bottommost):
                    x, y, w, h = cv2.boundingRect(contour)
                    if i == 0:
                        # 选中的最下方目标：橙色框+红色点+优先标记
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 3)
                        cv2.circle(frame, bottommost, 8, (0, 0, 255), -1)
                        frame = self.draw_chinese_text(frame, f"優先目標(面積:{int(cv2.contourArea(contour))})", (x, y-10), (255, 165, 0))
                    else:
                        # 未选中的目标：灰色框+蓝色点+忽略标记
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                        cv2.circle(frame, bottommost, 6, (255, 0, 0), -1)
                        frame = self.draw_chinese_text(frame, f"忽略目標(面積:{int(cv2.contourArea(contour))})", (x, y-10), (128, 128, 128))
                    cv2.drawContours(frame, [contour], -1, (0, 255, 255), 1)
                    cv2.circle(frame, bottommost, 4, (0, 255, 255), -1)
                    frame = self.draw_chinese_text(frame, f"最下端點", (bottommost[0]+10, bottommost[1]+20), (0, 0, 255) if i==0 else (255, 0, 0))

            # 6. 更新跟蹤（仅处理最下方的目标）
            tracked_objects = self.tracker.update(new_bottommost_list, self.frame_counter)

            # 7. 檢查自動重置
            self.check_auto_reset(tracked_objects, new_bottommost_list)

            # 8. 快速點擊處理（添加1秒冷却判断 + 仅处理优先目标 + 短按保持0.2秒）
            click_performed = False
            # 获取当前时间，判断是否在冷却期内
            now = time.time()
            if now - self.last_click_timestamp >= self.click_cooldown:
                # 不在冷却期，可执行点击（仅处理选中的优先目标）
                for obj_id, (bottommost, first_frame, frame_count) in tracked_objects.items():
                    if (obj_id not in self.tracker.processed and 
                        frame_count >= CLICK_DELAY_FRAMES and 
                        self.click_enabled):
                        
                        screen_pos = self.cam_to_screen(bottommost[0], bottommost[1])
                        if screen_pos:
                            # 核心修改：短按保持0.2秒后松开（替代原瞬时点击）
                            pyautogui.mouseDown(x=screen_pos[0], y=screen_pos[1])  # 按下鼠标
                            time.sleep(SHORT_PRESS_DURATION)                       # 保持0.2秒
                            pyautogui.mouseUp(x=screen_pos[0], y=screen_pos[1])    # 松开鼠标
                            
                            self.tracker.processed.add(obj_id)
                            self.last_click_timestamp = now  # 更新最后点击时间戳
                            print(f"⚡ 短按(保持0.2秒): ID={obj_id} | 屏幕({screen_pos[0]},{screen_pos[1]}) | 檢測{frame_count}幀 | 冷却倒计时: {self.click_cooldown}秒")
                            click_performed = True
                            break  # 冷却期内只触发一次点击
            else:
                # 仍在冷却期，打印提示（可选）
                remaining = self.click_cooldown - (now - self.last_click_timestamp)
                if self.click_enabled and len(tracked_objects) > 0:
                    print(f"⏳ 点击冷却中，剩余 {remaining:.1f} 秒")

            # 快速重置背景
            if click_performed and self.reset_background_after_click:
                time.sleep(self.background_reset_delay)
                self.reset_background()
                self.tracker.reset()

            # 9. 繪製跟蹤狀態（仅显示优先目标）
            for obj_id, (bottommost, first_frame, frame_count) in tracked_objects.items():
                if obj_id in self.tracker.processed:
                    color = (0, 255, 0)
                    text = f"ID:{obj_id}（已點擊）"
                else:
                    # 判断是否在冷却期，显示不同提示
                    if now - self.last_click_timestamp < self.click_cooldown:
                        color = (255, 0, 0)
                        remaining = self.click_cooldown - (now - self.last_click_timestamp)
                        text = f"ID:{obj_id}（冷却中 {remaining:.1f}s）"
                    else:
                        color = (0, 255, 255)
                        text = f"ID:{obj_id}（可點擊）"
                
                cv2.circle(frame, bottommost, 6, color, -1)
                frame = self.draw_chinese_text(frame, text, (bottommost[0]+10, bottommost[1]-10), color)

            # 10. 繪製狀態信息（添加冷却时间+下方优先提示）
            frame = self.draw_status(frame)
            # 11. 顯示窗口
            cv2.imshow("黑色掩码（邊緣增強）", black_mask)
            cv2.imshow("聯合掩码", combined_mask)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.imshow("差異圖像（運動檢測）", diff_frame)
            
            # 顯示邊緣檢測結果
            gray_for_edge = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
            gray_for_edge = cv2.GaussianBlur(gray_for_edge, (3, 3), 0)
            edges_show = cv2.Canny(gray_for_edge, EDGE_CANNY_THRESH1, EDGE_CANNY_THRESH2)
            cv2.imshow("邊緣檢測結果", edges_show)

            # 更新上一帧时间
            self.last_frame_time = time.time()

            # 12. 按鍵處理（不變）
            key = cv2.waitKey(1) & 0xFF
            if self.handle_key(key):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("程式結束")

    # FPS统计更新（不變）
    def update_fps(self):
        current_time = time.time()
        self.fps_frame_count += 1
        if current_time - self.fps_last_update >= self.fps_update_interval:
            self.current_fps = self.fps_frame_count / (current_time - self.fps_last_update)
            self.fps_frame_count = 0
            self.fps_last_update = current_time

    # 狀態顯示（添加冷却时间+下方优先信息）
    def draw_status(self, frame):
        if self.is_calibrated and len(self.calib_points) == 4:
            pts = np.array(self.calib_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
            frame = self.draw_chinese_text(frame, "檢測區域（遠距離+邊緣增強）", (self.calib_points[0][0]+10, self.calib_points[0][1]-10), (255, 0, 255))
        
        current_min_area = MIN_AREA * self.sensitivity
        # 计算剩余冷却时间
        now = time.time()
        remaining_cooldown = max(0.0, self.click_cooldown - (now - self.last_click_timestamp))
        status = [
            f"FPS: {self.current_fps:.1f}（目標{TARGET_FPS}FPS）",
            f"點擊: {'開啟' if self.click_enabled else '關閉'}（L鍵控制 | 短按保持0.2秒）",
            f"敏感度: {self.sensitivity}（最小面積: {current_min_area}）",
            f"黑色V值上限: {self.upper_black[2]}",
            f"亮度: {self.brightness}（e/r調整）",
            f"對比度: {self.contrast:.1f}（f/g調整）",
            f"跟蹤物體數: {len(self.tracker.objects)}（僅跟蹤最下方目標）",
            f"點擊延遲: 0ms | 背景重置延遲: {self.background_reset_delay*1000:.0f}ms",
            f"邊緣增強: Canny({EDGE_CANNY_THRESH1},{EDGE_CANNY_THRESH2})",
            f"点击冷却: {remaining_cooldown:.1f}秒（總冷却{self.click_cooldown}秒）",
            f"优先级: 最下方目標（y坐標最大）"  # 新增优先级提示
        ]
        for i, text in enumerate(status):
            y_pos = frame.shape[0] - 30 - i * 25
            frame = self.draw_chinese_text(frame, text, (10, y_pos), 
                                          (0, 255, 0) if self.click_enabled else (0, 0, 255))
        return frame

    # 按鍵處理（不變）
    def handle_key(self, key):
        if key == 27:
            return True
        elif key == ord('l') or key == ord('L'):
            self.click_enabled = not self.click_enabled
            print(f"點擊功能: {'開啟' if self.click_enabled else '關閉'}（短按保持0.2秒）")
        elif key == ord('q'):
            self.sensitivity = max(self.sensitivity - 1, 1)
            print(f"敏感度: {self.sensitivity}（當前最小檢測面積: {MIN_AREA*self.sensitivity}）")
        elif key == ord('w'):
            self.sensitivity = min(self.sensitivity + 1, 20)
            print(f"敏感度: {self.sensitivity}（當前最小檢測面積: {MIN_AREA*self.sensitivity}）")
        elif key == ord('z'):
            new_v = max(10, self.upper_black[2] - 5)
            self.upper_black = np.array([180, 255, new_v])
            print(f"黑色V值上限調整為: {new_v}（更嚴格）")
        elif key == ord('x'):
            new_v = min(100, self.upper_black[2] + 5)
            self.upper_black = np.array([180, 255, new_v])
            print(f"黑色V值上限調整為: {new_v}（更寬鬆）")
        elif key == ord('e'):
            self.brightness = min(self.brightness + 5, 100)
            print(f"亮度調整為: {self.brightness}")
        elif key == ord('r'):
            self.brightness = max(self.brightness - 5, -100)
            print(f"亮度調整為: {self.brightness}")
        elif key == ord('f'):
            self.contrast = min(round(self.contrast + 0.1, 1), 3.0)
            print(f"對比度調整為: {self.contrast:.1f}")
        elif key == ord('g'):
            self.contrast = max(round(self.contrast - 0.1, 1), 0.1)
            print(f"對比度調整為: {self.contrast:.1f}")
        elif key == ord('p'):
            print("重新校準...")
            if not self.calibrate():
                return True
        elif key == ord('b') or key == ord('B'):
            print("手動快速重置背景...")
            self.reset_background()
            self.tracker.reset()
        return False


if __name__ == "__main__":
    game = ProjectionGame()
    game.run()