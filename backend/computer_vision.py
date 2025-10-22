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
        Create clean and reasonable visual overlay system for navigation
        - Selective marking of key safe/blocked zones
        - Clean bounding boxes only for important obstacles
        - Minimal clutter with maximum clarity
        
        Args:
            frame: Input video frame
            detections: List of detected obstacles
            safe_path_grid: Binary grid with safe path information
            
        Returns:
            Frame with clean visual overlays for navigation
        """
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Calculate grid cell dimensions
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # 1. DRAW SELECTIVE SAFE/BLOCKED ZONE INDICATORS
        # Only mark zones that are critical for navigation (not every single cell)
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * cell_width
                y1 = row * cell_height
                center_x = x1 + cell_width // 2
                center_y = y1 + cell_height // 2
                
                # Only draw indicators in key areas (center corridor and edges)
                is_center_path = (col >= self.grid_cols // 3 and col <= 2 * self.grid_cols // 3)
                is_edge_area = (row == 0 or row == self.grid_rows - 1 or col == 0 or col == self.grid_cols - 1)
                
                if is_center_path or is_edge_area:
                    if safe_path_grid[row, col] == 1:  # Safe zone
                        # Small green indicators for safe movement areas
                        cv2.circle(overlay_frame, (center_x, center_y), 8, (0, 255, 0), -1)
                        cv2.circle(overlay_frame, (center_x, center_y), 10, (0, 200, 0), 2)
                    else:  # Blocked zone  
                        # Small red indicators for critical blocked areas
                        cv2.circle(overlay_frame, (center_x, center_y), 8, (0, 0, 255), -1)
                        cv2.circle(overlay_frame, (center_x, center_y), 10, (0, 0, 200), 2)
        
        # 2. DRAW CLEAN BOUNDING BOXES - Only for significant navigation obstacles
        important_objects = []
        
        # Sort detections by confidence and area to prioritize most relevant objects
        sorted_detections = sorted(detections, key=lambda d: (d.confidence * ((d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]))), reverse=True)
        
        for detection in sorted_detections:
            x1, y1, x2, y2 = detection.bbox
            object_width = x2 - x1
            object_height = y2 - y1
            object_area = object_width * object_height
            
            # Only draw boxes for objects that are reasonably sized and relevant
            min_display_area = (width * height) * 0.01   # At least 1% of frame
            max_display_area = (width * height) * 0.25   # At most 25% of frame
            
            # Confidence threshold for display
            min_confidence = 0.3
            
            if (min_display_area <= object_area <= max_display_area and 
                detection.confidence >= min_confidence):
                
                # Filter to only show the most navigation-relevant objects
                navigation_relevant_objects = [
                    'chair', 'dining table', 'couch', 'bed', 'person', 
                    'potted plant', 'tv', 'laptop', 'backpack', 'suitcase'
                ]
                
                if detection.class_name in navigation_relevant_objects:
                    # Check for overlap with existing objects to avoid redundant boxes
                    is_overlapping = False
                    for existing in important_objects:
                        if self._boxes_overlap(detection.bbox, existing.bbox, overlap_threshold=0.5):
                            is_overlapping = True
                            break
                    
                    # Limit to maximum 8 objects to avoid clutter
                    if not is_overlapping and len(important_objects) < 8:
                        important_objects.append(detection)
        
        # Draw clean bounding boxes for important objects only
        for detection in important_objects:
            x1, y1, x2, y2 = detection.bbox
            
            # Choose appropriate colors based on object type
            if detection.class_name in ['chair', 'dining table']:
                # Furniture: Clean blue boxes
                box_color = (255, 120, 0)  # Orange-blue
                label_bg_color = (255, 120, 0)
            elif detection.class_name == 'person':
                # People: Yellow boxes for visibility
                box_color = (0, 255, 255)  # Yellow
                label_bg_color = (0, 200, 200)
            else:
                # Other obstacles: Red boxes
                box_color = (0, 100, 255)  # Red-orange
                label_bg_color = (0, 100, 255)
            
            # Draw clean bounding box with moderate thickness
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Add simple, readable label (no confidence clutter)
            label = detection.class_name.title()
            
            # Calculate compact label size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw compact label background
            label_bg_x1 = x1
            label_bg_y1 = y1 - label_size[1] - 10
            label_bg_x2 = x1 + label_size[0] + 10
            label_bg_y2 = y1
            
            # Ensure label stays within frame
            if label_bg_y1 < 0:
                label_bg_y1 = y2
                label_bg_y2 = y2 + label_size[1] + 10
            
            # Semi-transparent label background for clarity without obstruction
            overlay = overlay_frame.copy()
            cv2.rectangle(overlay, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), label_bg_color, -1)
            cv2.addWeighted(overlay, 0.7, overlay_frame, 0.3, 0, overlay_frame)
            
            # Draw white text for maximum readability
            text_y = label_bg_y1 + label_size[1] + 5 if label_bg_y1 >= 0 else y2 + label_size[1] + 5
            cv2.putText(overlay_frame, label, (x1 + 5, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 3. CLEAN NAVIGATION STATUS (minimal, unobtrusive)
        safe_cells = np.sum(safe_path_grid)
        total_cells = safe_path_grid.size
        
        # Simple status indicator in top-left corner
        status_bg_x1, status_bg_y1 = 10, 10
        status_bg_x2, status_bg_y2 = 200, 50
        
        # Semi-transparent background
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (status_bg_x1, status_bg_y1), (status_bg_x2, status_bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, overlay_frame, 0.4, 0, overlay_frame)
        
        # Navigation status with clean text
        if self.has_clear_path(safe_path_grid):
            status_text = "PATH CLEAR"
            status_color = (0, 255, 0)  # Green
            
            # Simple navigation arrow (less intrusive)
            center_x = width // 2
            arrow_start_y = int(height * 0.9)
            arrow_end_y = int(height * 0.8)
            cv2.arrowedLine(overlay_frame, (center_x, arrow_start_y), 
                          (center_x, arrow_end_y), (0, 255, 0), 4, tipLength=0.2)
        else:
            status_text = "OBSTACLES"
            status_color = (0, 0, 255)  # Red
        
        # Draw clean status text
        cv2.putText(overlay_frame, status_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Show count of important objects only (less clutter)
        if len(important_objects) > 0:
            count_text = f"Objects: {len(important_objects)}"
            cv2.putText(overlay_frame, count_text, (width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
    def _boxes_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int], 
                      overlap_threshold: float = 0.5) -> bool:
        """Check if two bounding boxes overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        intersection_x1 = max(x1_1, x1_2)
        intersection_y1 = max(y1_1, y1_2)
        intersection_x2 = min(x2_1, x2_2)
        intersection_y2 = min(y2_1, y2_2)
        
        if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
            return False  # No overlap
        
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        
        # Calculate areas of both boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate overlap ratio
        overlap_ratio = intersection_area / min(area1, area2)
        
        return overlap_ratio > overlap_threshold
    
    def create_demo_visualization(self, frame_width: int = 800, frame_height: int = 600) -> np.ndarray:
        """Create a demo frame showing the clean visualization system"""
        import numpy as np
        
        # Create a demo frame (simulating a banquet hall view)
        demo_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 50  # Dark background
        
        # Simulate some furniture detections for demo
        demo_detections = [
            Detection(class_id=56, confidence=0.85, bbox=(100, 200, 200, 300), class_name="chair"),
            Detection(class_id=60, confidence=0.92, bbox=(250, 180, 400, 320), class_name="dining table"),
            Detection(class_id=56, confidence=0.78, bbox=(450, 210, 550, 310), class_name="chair"),
            Detection(class_id=0, confidence=0.95, bbox=(300, 100, 380, 250), class_name="person"),
            Detection(class_id=56, confidence=0.82, bbox=(600, 190, 700, 290), class_name="chair")
        ]
        
        # Create a demo safe path grid
        demo_safe_grid = np.ones((self.grid_rows, self.grid_cols), dtype=int)
        # Mark some areas as blocked around the furniture
        demo_safe_grid[2:4, 1:3] = 0  # Around table area
        demo_safe_grid[3:5, 6:8] = 0  # Another blocked area
        
        # Apply the clean visualization
        result_frame = self.draw_overlays(demo_frame, demo_detections, demo_safe_grid)
        
        # Add demo title
        cv2.putText(result_frame, "Clean Navigation Visualization Demo", (10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
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