# Computer Vision Processing Pipeline
# Implements YOLOv11 object detection, optical flow analysis, and visual overlay rendering

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    # Fallback if ultralytics is not properly installed
    YOLO = None

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Detection result data structure as per design document"""
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str

@dataclass
class MotionAnalysis:
    """Optical flow result data structure as per design document"""
    flow_magnitude: float
    is_stationary: bool
    confidence: float
    motion_vectors: np.ndarray

class VisionProcessor:
    """
    Computer vision processing pipeline implementing:
    - YOLOv11 object detection for obstacle identification
    - Optical flow analysis for motion detection
    - Visual overlay rendering for safe path visualization
    """
    
    def __init__(self, model_name: str = "yolo11n.pt", 
                 stationary_threshold: float = 2.0,
                 confidence_threshold: float = 0.5):
        """
        Initialize vision processor with pretrained YOLOv11 model from ultralytics
        
        Args:
            model_name: Pretrained model name (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
            stationary_threshold: Motion magnitude threshold for stationary detection
            confidence_threshold: Minimum confidence for object detection
        """
        # Use pretrained model from ultralytics - no local file needed
        self.model_name = model_name
        
        self.stationary_threshold = stationary_threshold
        self.confidence_threshold = confidence_threshold
        
        # Initialize YOLOv11 model
        self.yolo_model: Optional[Any] = None
        self._initialize_yolo_model()
        
        # Optical flow tracking
        self.previous_frame: Optional[np.ndarray] = None
        self.optical_flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Grid-based path calculation parameters
        self.grid_rows = 6
        self.grid_cols = 8
        self.safety_margin = 20  # pixels around obstacles
        
        logger.info(f"VisionProcessor initialized with pretrained model: {self.model_name}")
    
    def configure_thresholds(self, stationary_threshold: Optional[float] = None,
                           confidence_threshold: Optional[float] = None):
        """
        Configure detection thresholds dynamically
        Implements configurable thresholds as per requirement 1.2, 6.3, 6.6
        
        Args:
            stationary_threshold: Motion magnitude threshold for stationary detection
            confidence_threshold: Minimum confidence for object detection
        """
        if stationary_threshold is not None:
            self.stationary_threshold = stationary_threshold
            logger.info(f"Updated stationary threshold to: {stationary_threshold}")
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            logger.info(f"Updated confidence threshold to: {confidence_threshold}")
    
    def _initialize_yolo_model(self):
        """
        Initialize pretrained YOLOv11 model from ultralytics with automatic download
        Requirement 3.1: Initialize YOLOv11 model loading and configuration
        """
        try:
            logger.info(f"Loading pretrained YOLOv11 model: {self.model_name}")
            logger.info("Model will be automatically downloaded from ultralytics if not cached")
            
            # Load pretrained model - ultralytics will download automatically
            if YOLO is not None:
                self.yolo_model = YOLO(self.model_name)
            else:
                logger.warning("YOLO not available, using fallback detection")
            
            # Verify model loaded successfully
            if self.yolo_model is not None:
                logger.info("[OK] YOLOv11 pretrained model loaded successfully")
                # Log model information
                logger.info(f"[INFO] Model classes: {len(self.yolo_model.names)} classes")
                logger.info(f"[INFO] Model device: {self.yolo_model.device}")
                logger.info(f"[INFO] Model name: {self.model_name}")
                
                # Log some example classes for navigation
                example_classes = []
                for class_id, class_name in list(self.yolo_model.names.items())[:10]:
                    example_classes.append(f"{class_id}:{class_name}")
                logger.info(f"[INFO] Example classes: {', '.join(example_classes)}")
                
                # Test model with dummy input to ensure it's working
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                test_results = self.yolo_model(test_frame, verbose=False)
                logger.info("[OK] YOLOv11 model test inference successful")
                
            else:
                raise RuntimeError("Failed to load YOLOv11 pretrained model")
                
        except Exception as e:
            logger.error(f"❌ Error loading YOLOv11 pretrained model: {e}")
            logger.error("💡 Make sure you have internet connection for model download")
            logger.error("💡 Or check if ultralytics is properly installed: pip install ultralytics")
            self.yolo_model = None
            
            # Initialize fallback detection system
            self._initialize_fallback_detection()
            
            # Don't raise exception - allow system to continue with fallback
            logger.warning("⚠️ YOLOv11 model initialization failed - using fallback detection system")
    
    def _initialize_fallback_detection(self):
        """Initialize fallback object detection using basic computer vision techniques"""
        try:
            # Initialize background subtractor for motion-based detection
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True,
                varThreshold=50,
                history=500
            )
            
            # Initialize contour detection parameters
            self.min_contour_area = 1000  # Minimum area for obstacle detection
            self.max_contour_area = 50000  # Maximum area to filter out noise
            
            logger.info("Fallback detection system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback detection: {e}")
            self.background_subtractor = None
    
    def detect_obstacles(self, frame: np.ndarray) -> List[Detection]:
        """
        Implement frame processing for obstacle detection with fallback handling
        Requirement 3.1: Implement frame processing for obstacle detection
        
        Args:
            frame: Input video frame as numpy array
            
        Returns:
            List of Detection objects for detected obstacles
        """
        # Try YOLOv11 detection first
        if self.yolo_model is not None:
            try:
                return self._detect_obstacles_yolo(frame)
            except Exception as e:
                logger.error(f"YOLOv11 detection failed: {e}")
                # Fall back to basic detection
                logger.warning("Falling back to basic obstacle detection")
                return self._detect_obstacles_fallback(frame)
        else:
            # Use fallback detection system
            logger.debug("Using fallback obstacle detection")
            return self._detect_obstacles_fallback(frame)
    
    def _detect_obstacles_yolo(self, frame: np.ndarray) -> List[Detection]:
        """YOLOv11-based obstacle detection with navigation-specific filtering"""
        # Run YOLOv11 inference on frame
        if self.yolo_model is None:
            return []  # Return empty detections if model not available
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        
        # Define navigation-relevant obstacle classes focusing on banquet hall/event space objects
        navigation_obstacles = {
            # People and animals (highest priority for navigation)
            0: 'person',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
            21: 'bear', 22: 'zebra', 23: 'giraffe',
            
            # BANQUET HALL / EVENT SPACE FURNITURE (key targets like in your image)
            56: 'chair',          # Primary target - chairs in banquet halls
            57: 'couch',          # Lounge seating
            58: 'potted plant',   # Decorative plants common in events
            59: 'bed',            # Hotel/venue furniture
            60: 'dining table',   # Main tables in banquet halls - CRITICAL
            61: 'toilet',         # Restroom facilities
            
            # Display and presentation equipment
            62: 'tv',             # Screens/displays for presentations
            72: 'refrigerator',   # Catering equipment
            74: 'clock',          # Wall clocks in venues
            75: 'vase',          # Decorative items on tables
            
            # Event/meeting room equipment 
            63: 'laptop',         # Presentation equipment
            64: 'mouse', 65: 'remote', 66: 'keyboard',  # AV equipment
            67: 'cell phone',     # Personal items on tables
            
            # Catering and dining items (common in banquet halls)
            40: 'bottle',         # Water bottles, wine on tables
            41: 'wine glass',     # Glassware on dining tables  
            42: 'cup',           # Coffee cups, tea cups
            43: 'fork', 44: 'knife', 45: 'spoon',  # Dining utensils
            46: 'bowl',          # Serving bowls on tables
            
            # Kitchen/catering equipment
            68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            
            # Transportation obstacles (parking lots, venue access)
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
            
            # Personal items and bags (left on chairs/tables at events)
            24: 'backpack',       # Guest bags
            25: 'umbrella',       # Weather protection items  
            26: 'handbag',        # Purses left on tables/chairs
            27: 'tie',           # Formal wear items
            28: 'suitcase',      # Luggage for traveling guests
            
            # Sports and recreational equipment (if venue has multiple uses)
            32: 'sports ball', 37: 'skateboard', 38: 'surfboard',
            39: 'tennis racket',
            
            # Books and documents (common on conference tables)
            73: 'book',
            
            # Personal care items
            76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        
        frame_height, frame_width = frame.shape[:2]
        
        # Process detection results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                # Extract detection data
                for i in range(len(boxes)):
                    # Get bounding box coordinates
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Get confidence score
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # Get class ID and name
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.yolo_model.names[class_id] if self.yolo_model else f"class_{class_id}"
                    
                    # Filter by confidence threshold and navigation relevance
                    if (confidence >= self.confidence_threshold and 
                        class_id in navigation_obstacles):
                        
                        # Additional filtering: ignore objects in upper portion of frame (likely ceiling/roof)
                        object_center_y = (y1 + y2) / 2
                        if object_center_y < frame_height * 0.3:  # Skip objects in top 30% of frame
                            logger.debug(f"Skipping {class_name} in upper portion of frame (y={object_center_y})")
                            continue
                        
                        # Filter out very small objects that might be false positives
                        object_width = x2 - x1
                        object_height = y2 - y1
                        if object_width < 20 or object_height < 20:
                            logger.debug(f"Skipping small {class_name} object ({object_width}x{object_height})")
                            continue
                        
                        detection = Detection(
                            class_id=class_id,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            class_name=class_name
                        )
                        detections.append(detection)
                        logger.debug(f"Detected navigation obstacle: {class_name} at ({x1},{y1},{x2},{y2}) confidence={confidence:.2f}")
        
        logger.debug(f"YOLOv11 detected {len(detections)} navigation obstacles with confidence >= {self.confidence_threshold}")
        return detections
    
    def _detect_obstacles_fallback(self, frame: np.ndarray) -> List[Detection]:
        """Fallback obstacle detection using basic computer vision techniques"""
        detections = []
        
        try:
            if self.background_subtractor is None:
                # Simple edge-based detection as last resort
                return self._detect_obstacles_edge_based(frame)
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours as potential obstacles
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filter by area
                if self.min_contour_area <= area <= self.max_contour_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Create detection object
                    detection = Detection(
                        class_id=0,  # Generic obstacle class
                        confidence=min(0.8, area / self.max_contour_area),  # Confidence based on size
                        bbox=(x, y, x + w, y + h),
                        class_name="obstacle"
                    )
                    detections.append(detection)
            
            logger.debug(f"Fallback detection found {len(detections)} potential obstacles")
            return detections
            
        except Exception as e:
            logger.error(f"Fallback detection failed: {e}")
            # Last resort: edge-based detection
            return self._detect_obstacles_edge_based(frame)
    
    def _detect_obstacles_edge_based(self, frame: np.ndarray) -> List[Detection]:
        """Last resort edge-based obstacle detection"""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours from edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process significant contours
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > 2000:  # Only large edge regions
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Create basic detection
                    detection = Detection(
                        class_id=0,
                        confidence=0.5,  # Low confidence for edge-based detection
                        bbox=(x, y, x + w, y + h),
                        class_name="edge_obstacle"
                    )
                    detections.append(detection)
            
            logger.debug(f"Edge-based detection found {len(detections)} potential obstacles")
            return detections
            
        except Exception as e:
            logger.error(f"Edge-based detection failed: {e}")
            return []
    
    def calculate_optical_flow(self, current_frame: np.ndarray) -> MotionAnalysis:
        """
        Set up OpenCV optical flow calculation between frames
        Requirement 1.2, 6.3, 6.6: Motion magnitude analysis for user movement detection
        
        Args:
            current_frame: Current video frame
            
        Returns:
            MotionAnalysis object with motion information
        """
        try:
            import numpy as np
            # Convert to grayscale for optical flow calculation
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Initialize motion analysis with default values
            motion_analysis = MotionAnalysis(
                flow_magnitude=0.0,
                is_stationary=True,
                confidence=1.0,
                motion_vectors=np.array([])
            )
            
            # Check if we have a previous frame for comparison
            if self.previous_frame is None:
                logger.debug("No previous frame available for optical flow")
                self.previous_frame = gray_current.copy()
                return motion_analysis
            
            # Calculate optical flow using Lucas-Kanade method
            # Create grid of points to track
            h, w = gray_current.shape
            y_points, x_points = np.mgrid[20:h-20:20, 20:w-20:20].reshape(2, -1)
            points_to_track = np.column_stack([x_points, y_points]).astype(np.float32)
            
            if len(points_to_track) == 0:
                logger.warning("No points to track for optical flow")
                self.previous_frame = gray_current.copy()
                return motion_analysis
            
            # Calculate optical flow with explicit parameters
            import numpy as np
            nextPts = np.zeros_like(points_to_track)  # Initialize output array
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.previous_frame,
                gray_current,
                points_to_track,
                nextPts,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Filter good points
            good_mask = status.flatten() == 1
            good_new = new_points[good_mask]
            good_old = points_to_track[good_mask]
            
            if len(good_new) > 0:
                # Calculate motion vectors
                motion_vectors = good_new - good_old
                
                # Calculate motion magnitude
                magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
                avg_magnitude = np.mean(magnitudes)
                
                # Determine if user is stationary
                is_stationary = avg_magnitude < self.stationary_threshold
                
                # Calculate confidence based on number of tracked points
                confidence = min(1.0, len(good_new) / len(points_to_track))
                
                motion_analysis = MotionAnalysis(
                    flow_magnitude=float(avg_magnitude),
                    is_stationary=bool(is_stationary),
                    confidence=confidence,
                    motion_vectors=motion_vectors
                )
                
                logger.debug(f"Optical flow: magnitude={avg_magnitude:.2f}, stationary={is_stationary}, confidence={confidence:.2f}")
            
            # Update previous frame
            self.previous_frame = gray_current.copy()
            
            return motion_analysis
            
        except Exception as e:
            logger.error(f"Error in optical flow calculation: {e}")
            # Return default stationary analysis on error
            import numpy as np
            return MotionAnalysis(
                flow_magnitude=0.0,
                is_stationary=True,
                confidence=0.0,
                motion_vectors=np.array([])
            )
    
    def calculate_safe_path(self, frame_shape: Tuple[int, int], detections: List[Detection]) -> np.ndarray:
        """
        Implement ground-level safe path calculation using grid-based approach
        Focuses on walkable ground paths, ignoring ceiling/roof objects
        
        Args:
            frame_shape: (height, width) of the video frame
            detections: List of detected obstacles
            
        Returns:
            Binary grid where 1 indicates safe path, 0 indicates obstacle/unsafe
        """
        height, width = frame_shape[:2]
        
        # Create grid for path calculation - focus on ground level (bottom 70% of frame)
        grid = np.ones((self.grid_rows, self.grid_cols), dtype=np.uint8)
        
        # Calculate grid cell dimensions
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # Define ground level area (bottom 70% of frame for walking navigation)
        ground_level_start = int(height * 0.3)  # Start from 30% down from top
        
        # Mark obstacle cells as unsafe - only consider ground-level obstacles
        ground_obstacles = 0
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Check if obstacle is in ground level area
            obstacle_bottom = y2
            obstacle_center_y = (y1 + y2) / 2
            
            if obstacle_center_y < ground_level_start:
                logger.debug(f"Ignoring {detection.class_name} - above ground level (y={obstacle_center_y})")
                continue
            
            ground_obstacles += 1
            
            # Add larger safety margin for navigation obstacles
            safety_margin = self.safety_margin * 2  # Double margin for walking safety
            x1 = max(0, x1 - safety_margin)
            y1 = max(0, y1 - safety_margin)
            x2 = min(width, x2 + safety_margin)
            y2 = min(height, y2 + safety_margin)
            
            # Convert pixel coordinates to grid coordinates
            grid_x1 = max(0, x1 // cell_width)
            grid_y1 = max(0, y1 // cell_height)
            grid_x2 = min(self.grid_cols, (x2 + cell_width - 1) // cell_width)
            grid_y2 = min(self.grid_rows, (y2 + cell_height - 1) // cell_height)
            
            # Mark affected grid cells as unsafe
            grid[grid_y1:grid_y2, grid_x1:grid_x2] = 0
            
            logger.debug(f"Marked {detection.class_name} as obstacle in grid cells "
                        f"({grid_x1},{grid_y1}) to ({grid_x2},{grid_y2})")
        
        # Also mark top rows as unsafe (ceiling/roof area)
        top_rows_to_ignore = max(1, self.grid_rows // 3)  # Top 1/3 of grid
        grid[:top_rows_to_ignore, :] = 0
        
        safe_cells = np.sum(grid)
        logger.debug(f"Ground-level path calculation: {ground_obstacles} obstacles detected, "
                    f"{safe_cells} safe cells out of {grid.size} total cells")
        
        return grid
    
    def has_clear_path(self, safe_path_grid: np.ndarray) -> bool:
        """
        Check if there is a clear path from bottom to top of the grid
        
        Args:
            safe_path_grid: Binary grid with safe path information
            
        Returns:
            True if clear path exists, False otherwise
        """
        # Check if there's a clear corridor from bottom center upward
        center_col = self.grid_cols // 2
        
        # Check center column and adjacent columns for clear path
        for col in range(max(0, center_col - 1), min(self.grid_cols, center_col + 2)):
            # Check if this column has a clear path from bottom to top
            column_clear = True
            for row in range(self.grid_rows - 1, -1, -1):  # Bottom to top
                if safe_path_grid[row, col] == 0:
                    column_clear = False
                    break
            
            if column_clear:
                return True
        
        return False
    
    def draw_overlays(self, frame: np.ndarray, detections: List[Detection], 
                     safe_path_grid: np.ndarray) -> np.ndarray:
        """
        Create enhanced visual overlay rendering system exactly like the banquet hall example
        - Green marks/areas = Safe zones for movement (no obstacles)
        - Red marks/areas = Obstacles/blocked areas 
        - Clear bounding boxes around all detected objects (chairs, tables, etc.)
        
        Args:
            frame: Input video frame
            detections: List of detected obstacles
            safe_path_grid: Binary grid with safe path information
            
        Returns:
            Frame with enhanced visual overlays for navigation exactly like requested
        """
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Calculate grid cell dimensions
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # 1. DRAW SAFE AND BLOCKED ZONES (like the green/red marks in the image)
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Calculate center of each grid cell for mark placement
                center_x = x1 + cell_width // 2
                center_y = y1 + cell_height // 2
                
                if safe_path_grid[row, col] == 1:  # Safe zone
                    # Draw bright GREEN marks for safe movement areas
                    # Use filled circles similar to the green marks in your image
                    cv2.circle(overlay_frame, (center_x, center_y), 15, (0, 255, 0), -1)
                    cv2.circle(overlay_frame, (center_x, center_y), 15, (0, 200, 0), 3)  # Border
                    
                    # Add subtle green overlay for the entire safe area
                    cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                else:  # Blocked zone  
                    # Draw bright RED marks for obstacles/blocked areas
                    # Use filled circles similar to the red marks in your image
                    cv2.circle(overlay_frame, (center_x, center_y), 15, (0, 0, 255), -1)
                    cv2.circle(overlay_frame, (center_x, center_y), 15, (0, 0, 200), 3)  # Border
                    
                    # Add subtle red overlay for blocked areas
                    cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 2. DRAW CLEAR BOUNDING BOXES around ALL detected objects (chairs, tables, etc.)
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw thick, clear bounding box around each object
            # Use bright colors for maximum visibility
            if detection.class_name in ['chair', 'dining table', 'couch', 'bed']:
                # Furniture gets bright blue boxes for clarity
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (255, 165, 0), 4)  # Orange
            elif detection.class_name == 'person':
                # People get yellow boxes
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Yellow
            else:
                # Other obstacles get red boxes
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Red
            
            # Add clear object label with enhanced visibility
            label = f"{detection.class_name.upper()}"
            confidence_text = f"{detection.confidence:.1%}"
            
            # Calculate label dimensions
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            conf_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background for maximum readability
            label_bg_x1 = x1
            label_bg_y1 = y1 - label_size[1] - conf_size[1] - 15
            label_bg_x2 = x1 + max(label_size[0], conf_size[0]) + 15
            label_bg_y2 = y1
            
            # Black background with white border for labels
            cv2.rectangle(overlay_frame, (label_bg_x1, label_bg_y1), 
                         (label_bg_x2, label_bg_y2), (0, 0, 0), -1)
            cv2.rectangle(overlay_frame, (label_bg_x1, label_bg_y1), 
                         (label_bg_x2, label_bg_y2), (255, 255, 255), 2)
            
            # Draw white text on black background for maximum contrast
            cv2.putText(overlay_frame, label, (x1 + 5, y1 - conf_size[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay_frame, confidence_text, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 3. NAVIGATION STATUS INDICATOR
        safe_cells = np.sum(safe_path_grid)
        total_cells = safe_path_grid.size
        safe_percentage = (safe_cells / total_cells) * 100
        
        # Status background
        status_bg_x1, status_bg_y1 = 10, 10
        status_bg_x2, status_bg_y2 = 400, 120
        cv2.rectangle(overlay_frame, (status_bg_x1, status_bg_y1), 
                     (status_bg_x2, status_bg_y2), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (status_bg_x1, status_bg_y1), 
                     (status_bg_x2, status_bg_y2), (255, 255, 255), 2)
        
        # Navigation status
        if self.has_clear_path(safe_path_grid):
            status_text = "NAVIGATION: CLEAR PATH"
            status_color = (0, 255, 0)  # Green
            
            # Draw forward navigation arrow
            center_x = width // 2
            arrow_start_y = int(height * 0.85)
            arrow_end_y = int(height * 0.65)
            cv2.arrowedLine(overlay_frame, (center_x, arrow_start_y), 
                          (center_x, arrow_end_y), (0, 255, 0), 8, tipLength=0.3)
        else:
            status_text = "NAVIGATION: OBSTACLES DETECTED"
            status_color = (0, 0, 255)  # Red
        
        # Draw status information
        cv2.putText(overlay_frame, status_text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(overlay_frame, f"OBJECTS DETECTED: {len(detections)}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay_frame, f"SAFE AREAS: {safe_percentage:.1f}%", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay_frame, f"GRID: {safe_cells}/{total_cells} CLEAR", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 4. GROUND LEVEL INDICATOR (like in banquet hall analysis)
        ground_line_y = int(height * 0.3)
        cv2.line(overlay_frame, (0, ground_line_y), (width, ground_line_y), 
                (255, 255, 0), 3)  # Thick yellow line
        cv2.putText(overlay_frame, "GROUND LEVEL ANALYSIS", (width - 300, ground_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 5. LEGEND (like the color coding you showed)
        legend_x = width - 250
        legend_y = height - 120
        
        # Legend background
        cv2.rectangle(overlay_frame, (legend_x - 10, legend_y - 40), 
                     (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (legend_x - 10, legend_y - 40), 
                     (width - 10, height - 10), (255, 255, 255), 2)
        
        # Legend items
        cv2.circle(overlay_frame, (legend_x, legend_y), 8, (0, 255, 0), -1)
        cv2.putText(overlay_frame, "SAFE ZONES", (legend_x + 20, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.circle(overlay_frame, (legend_x, legend_y + 25), 8, (0, 0, 255), -1)
        cv2.putText(overlay_frame, "OBSTACLES", (legend_x + 20, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.rectangle(overlay_frame, (legend_x, legend_y + 50), (legend_x + 15, legend_y + 65), (255, 165, 0), 3)
        cv2.putText(overlay_frame, "OBJECTS", (legend_x + 20, legend_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return overlay_frame
    
    async def process_frame_complete(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Complete frame processing pipeline combining all vision components
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with processing results including detections, motion analysis, and processed frame
        """
        try:
            # Detect obstacles using YOLOv11
            detections = self.detect_obstacles(frame)
            
            # Calculate optical flow for motion analysis
            motion_analysis = self.calculate_optical_flow(frame)
            
            # Calculate safe path grid
            safe_path_grid = self.calculate_safe_path(frame.shape, detections)
            
            # Check if path is clear
            path_clear = self.has_clear_path(safe_path_grid)
            
            # Draw visual overlays
            processed_frame = self.draw_overlays(frame, detections, safe_path_grid)
            
            # Compile results
            results = {
                "detections": [
                    {
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox": d.bbox
                    } for d in detections
                ],
                "motion_analysis": {
                    "flow_magnitude": motion_analysis.flow_magnitude,
                    "is_stationary": motion_analysis.is_stationary,
                    "confidence": motion_analysis.confidence
                },
                "path_analysis": {
                    "safe_path_grid": safe_path_grid.tolist(),
                    "path_clear": path_clear,
                    "safe_cells": int(np.sum(safe_path_grid)),
                    "total_cells": int(safe_path_grid.size)
                },
                "processed_frame": processed_frame,
                "timestamp": datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in complete frame processing: {e}")
            return {
                "error": str(e),
                "detections": [],
                "motion_analysis": {
                    "flow_magnitude": 0.0,
                    "is_stationary": True,
                    "confidence": 0.0
                },
                "path_analysis": {
                    "safe_path_grid": [],
                    "path_clear": False,
                    "safe_cells": 0,
                    "total_cells": 0
                },
                "processed_frame": frame,
                "timestamp": datetime.now().isoformat()
            }

# Global vision processor instance
vision_processor: Optional[VisionProcessor] = None

def get_vision_processor() -> VisionProcessor:
    """Get or create global vision processor instance with pretrained model"""
    global vision_processor
    if vision_processor is None:
        # Use pretrained model from ultralytics - no local file needed
        vision_processor = VisionProcessor(model_name="yolo11n.pt")
    return vision_processor