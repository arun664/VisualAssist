# Computer Vision Processing Pipeline
# Implements YOLOv11 object detection, optical flow analysis, and visual overlay rendering

import cv2
import numpy as np
import logging
import math
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
                     safe_path_grid: np.ndarray) -> Tuple[np.ndarray, dict]:
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
            Tuple containing:
                - Frame with clean visual overlays for navigation
                - Dictionary with navigation guidance information
        """
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Calculate grid cell dimensions
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # 1. DRAW PATHWAY GRID TILES
        # Draw tiles for all grid cells to create a clear pathway visualization
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Calculate cell coordinates
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Add small margin to make tiles visually separated
                margin = 2
                x1 += margin
                y1 += margin
                x2 -= margin
                y2 -= margin
                
                # Determine if this is part of the main walkway (center columns)
                is_main_path = (col >= self.grid_cols // 3 and col <= 2 * self.grid_cols // 3)
                
                if safe_path_grid[row, col] == 1:  # Safe zone
                    # Use semi-transparent green squares for walkable areas
                    # Brighter green for main path, softer green for side paths
                    if is_main_path:
                        # Main walkway - brighter and more visible
                        color = (0, 220, 0)  # Bright green
                        alpha = 0.25  # More visible
                    else:
                        # Side paths - softer visualization
                        color = (0, 180, 0)  # Softer green
                        alpha = 0.15  # More subtle
                    
                    # Draw filled rectangle with semi-transparency
                    overlay = overlay_frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, alpha, overlay_frame, 1-alpha, 0, overlay_frame)
                    
                    # Add grid lines
                    cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 1)
                    
                else:  # Blocked zone
                    # Use semi-transparent red squares for non-walkable areas
                    color = (0, 0, 220)  # Red
                    
                    # Only show blocked areas that are important for navigation
                    # (to avoid cluttering the visualization)
                    if is_main_path or row >= self.grid_rows // 2:  # Only in main path or bottom half
                        overlay = overlay_frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        cv2.addWeighted(overlay, 0.2, overlay_frame, 0.8, 0, overlay_frame)
                        cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 1)
        
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
        navigation_guidance = {}
        
        if self.has_clear_path(safe_path_grid):
            status_text = "PATH CLEAR"
            status_color = (0, 255, 0)  # Green
            
            # Draw a clear walking path from bottom to top and get guidance info
            navigation_guidance = self._draw_walking_path(overlay_frame, safe_path_grid, cell_width, cell_height)
            
            # Add voice guidance message to the frame
            if navigation_guidance and navigation_guidance.get("navigation_message"):
                guidance_text = navigation_guidance.get("navigation_message")
                cv2.putText(overlay_frame, guidance_text, (width // 2 - 150, height - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            status_text = "OBSTACLES"
            status_color = (0, 0, 255)  # Red
            
            # Try to find partial paths or suggest directions
            navigation_guidance = self._suggest_alternative_paths(overlay_frame, safe_path_grid, cell_width, cell_height)
            
            # Add turn around guidance to the frame
            guidance_text = "Turn around to find a clear path"
            cv2.putText(overlay_frame, guidance_text, (width // 2 - 150, height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw clean status text
        cv2.putText(overlay_frame, status_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Show count of important objects only (less clutter)
        if len(important_objects) > 0:
            count_text = f"Objects: {len(important_objects)}"
            cv2.putText(overlay_frame, count_text, (width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Return both the overlay frame and the navigation guidance info
        return overlay_frame, navigation_guidance
    
    def _draw_walking_path(self, frame: np.ndarray, safe_path_grid: np.ndarray, 
                           cell_width: int, cell_height: int) -> dict:
        """
        Draw a 3D-like direction path similar to navigation apps
        
        Args:
            frame: The frame to draw on
            safe_path_grid: Binary grid with safe path information
            cell_width: Width of each grid cell
            cell_height: Height of each grid cell
            
        Returns:
            dict: Navigation guidance information including direction and step instructions
        """
        height, width = frame.shape[:2]
        
        # Find the best path through the grid using simple pathfinding
        path_points = self._find_navigation_path(safe_path_grid)
        
        # Navigation guidance information
        guidance_info = {
            "path_found": False,
            "direction": None,
            "next_step": None,
            "distance": 0,
            "navigation_message": "Turn around to find a clear path"
        }
        
        if not path_points:
            return guidance_info  # No path found
        
        # Convert grid cells to pixel coordinates (center of cells)
        pixel_path = []
        for row, col in path_points:
            x = int(col * cell_width + cell_width / 2)
            y = int(row * cell_height + cell_height / 2)
            pixel_path.append((x, y))
        
        if len(pixel_path) < 2:
            return guidance_info  # Need at least 2 points to draw a path
        
        # Create perspective effect - make path wider at bottom, narrower at top
        # This creates a 3D-like effect similar to navigation apps
        path_segments = []
        path_width_bottom = int(cell_width * 0.7)  # Wider at bottom
        path_width_top = int(cell_width * 0.4)     # Narrower at top
        
        # Calculate total path length for gradient effect
        total_path_length = len(pixel_path) - 1
        
        # Draw 3D-like navigation path
        for i in range(len(pixel_path) - 1):
            p1 = pixel_path[i]
            p2 = pixel_path[i + 1]
            
            # Calculate direction vector
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Normalize direction vector
            length = max(1, math.sqrt(dx*dx + dy*dy))
            dx, dy = dx/length, dy/length
            
            # Perpendicular vector for width
            perp_x, perp_y = -dy, dx
            
            # Calculate width for this segment based on position in path (perspective effect)
            # Segments closer to the bottom (higher row value) are wider
            progress = i / total_path_length
            segment_width = int(path_width_bottom * (1 - progress) + path_width_top * progress)
            
            # Calculate the four corners of the path segment (trapezoid)
            # For current point
            p1_left = (int(p1[0] + perp_x * segment_width), int(p1[1] + perp_y * segment_width))
            p1_right = (int(p1[0] - perp_x * segment_width), int(p1[1] - perp_y * segment_width))
            
            # Width for next segment
            next_progress = (i + 1) / total_path_length
            next_segment_width = int(path_width_bottom * (1 - next_progress) + path_width_top * next_progress)
            
            # For next point
            p2_left = (int(p2[0] + perp_x * next_segment_width), int(p2[1] + perp_y * next_segment_width))
            p2_right = (int(p2[0] - perp_x * next_segment_width), int(p2[1] - perp_y * next_segment_width))
            
            # Create a polygon for this segment
            path_segment = np.array([p1_left, p2_left, p2_right, p1_right], np.int32)
            path_segments.append(path_segment)
            
            # Draw filled segment with gradient color (green to blue-green)
            # Base color is green (0, 255, 0), transitions to teal (0, 255, 128) at the top
            g_val = 255
            b_val = int(128 * progress)  # Blue component increases with progress
            color = (0, g_val, b_val)
            
            # Draw filled polygon with semi-transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [path_segment], color)
            
            # Add outer glow/edge effect
            cv2.polylines(overlay, [path_segment], True, (0, 255, 255), 2)
            
            # Apply segment with transparency
            alpha = 0.7 - (0.3 * progress)  # More transparent further away
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add direction arrows on the path
        self._add_direction_arrows(frame, pixel_path)
        
        # Add start and destination markers
        if len(pixel_path) > 1:
            # Start point (current position)
            start_x, start_y = pixel_path[0]
            cv2.circle(frame, (start_x, start_y), int(path_width_bottom * 0.8), (0, 255, 64), -1)
            cv2.circle(frame, (start_x, start_y), int(path_width_bottom * 0.8), (255, 255, 255), 2)
            
            # Destination marker
            dest_x, dest_y = pixel_path[-1]
            
            # Draw a pin-like destination marker
            pin_height = int(cell_height * 0.8)
            pin_width = int(cell_width * 0.5)
            
            # Pin top (circle)
            cv2.circle(frame, (dest_x, dest_y - pin_height//2), pin_width, (0, 128, 255), -1)
            cv2.circle(frame, (dest_x, dest_y - pin_height//2), pin_width, (255, 255, 255), 2)
            
            # Pin stem (triangle)
            pin_bottom = np.array([
                [dest_x - pin_width//2, dest_y - pin_height//2],
                [dest_x + pin_width//2, dest_y - pin_height//2],
                [dest_x, dest_y + pin_height//2]
            ], np.int32)
            cv2.fillPoly(frame, [pin_bottom], (0, 128, 255))
            cv2.polylines(frame, [pin_bottom], True, (255, 255, 255), 2)
            
            # Calculate direction for voice guidance
            if len(path_points) >= 2:
                # Get first two points to determine initial direction
                x1, y1 = path_points[0]
                x2, y2 = path_points[1]
                
                # Calculate angle
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                
                # Determine direction
                direction = None
                if -45 <= angle < 45:
                    direction = "right"
                elif 45 <= angle < 135:
                    direction = "forward"
                elif angle >= 135 or angle < -135:
                    direction = "left"
                elif -135 <= angle < -45:
                    direction = "backward"
                
                # Determine distance (rough estimate)
                distance = len(path_points)
                
                # Create guidance message
                if direction == "forward":
                    navigation_message = "Clear path ahead. Move forward one step."
                elif direction:
                    navigation_message = f"Turn slightly {direction} and move one step."
                else:
                    navigation_message = "Move forward carefully."
                
                # Update guidance info
                guidance_info = {
                    "path_found": True,
                    "direction": direction,
                    "next_step": "move_forward",
                    "distance": distance,
                    "navigation_message": navigation_message
                }
                
                return guidance_info
        
        return guidance_info
    
    def _add_direction_arrows(self, frame: np.ndarray, path_points: List[Tuple[int, int]]) -> None:
        """
        Add direction arrows to the path
        
        Args:
            frame: The frame to draw on
            path_points: List of path points in pixel coordinates
        """
        if len(path_points) < 5:
            return  # Need enough points for meaningful arrows
        
        # Add arrows at intervals along the path
        interval = max(1, len(path_points) // 4)  # Show about 3-4 arrows on the path
        
        for i in range(interval, len(path_points) - interval, interval):
            # Get points before and after for direction
            p_before = path_points[i - interval]
            p_current = path_points[i]
            
            # Calculate direction vector
            dx = p_current[0] - p_before[0]
            dy = p_current[1] - p_before[1]
            
            # Skip if movement is too small
            if abs(dx) < 5 and abs(dy) < 5:
                continue
                
            # Calculate arrow size based on path position (bigger at bottom)
            progress = i / len(path_points)
            arrow_size = int(15 * (1 - progress) + 8 * progress)
            
            # Draw a nice arrow with 3D effect
            # Main arrow
            cv2.arrowedLine(frame, p_before, p_current, (0, 255, 255), arrow_size, tipLength=0.3)
            
            # Inner arrow (for 3D effect)
            cv2.arrowedLine(frame, p_before, p_current, (0, 200, 128), arrow_size//2, tipLength=0.3)
    
    def _find_navigation_path(self, safe_path_grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find a natural-looking path through the safe grid cells
        
        Args:
            safe_path_grid: Binary grid with safe path information
            
        Returns:
            List of (row, col) coordinates forming a path
        """
        rows, cols = safe_path_grid.shape
        
        # Start from bottom center
        start_row = rows - 1
        start_col = cols // 2
        
        # Try to find a suitable start point if center is blocked
        if safe_path_grid[start_row, start_col] == 0:
            for offset in range(1, cols//2):
                if start_col - offset >= 0 and safe_path_grid[start_row, start_col - offset] == 1:
                    start_col = start_col - offset
                    break
                elif start_col + offset < cols and safe_path_grid[start_row, start_col + offset] == 1:
                    start_col = start_col + offset
                    break
        
        # If still no valid start, return empty path
        if safe_path_grid[start_row, start_col] == 0:
            return []
        
        # Find destination (somewhere in the top half of the grid)
        dest_row = rows // 4  # Aim for top quarter
        dest_col = cols // 2   # Center of grid
        
        # Try to find a suitable destination
        if safe_path_grid[dest_row, dest_col] == 0:
            best_dest = None
            # Search top half for safe cells
            for r in range(0, rows//2):
                for c in range(cols):
                    if safe_path_grid[r, c] == 1:
                        if best_dest is None or r < best_dest[0]:
                            best_dest = (r, c)
            
            if best_dest:
                dest_row, dest_col = best_dest
            else:
                # No suitable destination found
                return [(start_row, start_col)]  # Just return starting point
        
        # Simple A* pathfinding
        import heapq
        
        # Initialize data structures for A*
        open_set = [(0, start_row, start_col)]  # Priority queue (f_score, row, col)
        came_from = {}  # To reconstruct the path
        g_score = {(start_row, start_col): 0}  # Cost from start
        f_score = {(start_row, start_col): abs(start_row - dest_row) + abs(start_col - dest_col)}  # Estimated cost to goal
        
        while open_set:
            _, current_row, current_col = heapq.heappop(open_set)
            
            # Check if we reached destination
            if current_row == dest_row and current_col == dest_col:
                # Reconstruct path
                path = [(current_row, current_col)]
                while (current_row, current_col) in came_from:
                    current_row, current_col = came_from[(current_row, current_col)]
                    path.append((current_row, current_col))
                path.reverse()  # Start to destination
                return path
            
            # Check neighbors (4-directional movement)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor_row, neighbor_col = current_row + dr, current_col + dc
                
                # Check if valid and safe
                if (0 <= neighbor_row < rows and 0 <= neighbor_col < cols and 
                    safe_path_grid[neighbor_row, neighbor_col] == 1):
                    
                    # Calculate cost (diagonal movement costs more)
                    move_cost = 1.4 if abs(dr) + abs(dc) > 1 else 1.0
                    tentative_g = g_score[(current_row, current_col)] + move_cost
                    
                    if ((neighbor_row, neighbor_col) not in g_score or 
                        tentative_g < g_score[(neighbor_row, neighbor_col)]):
                        
                        # This path is better
                        came_from[(neighbor_row, neighbor_col)] = (current_row, current_col)
                        g_score[(neighbor_row, neighbor_col)] = tentative_g
                        f_score[(neighbor_row, neighbor_col)] = tentative_g + abs(neighbor_row - dest_row) + abs(neighbor_col - dest_col)
                        
                        # Add to open set if not already there
                        heapq.heappush(open_set, (f_score[(neighbor_row, neighbor_col)], neighbor_row, neighbor_col))
        
        # No path found, return empty list
        return []
    
    def _suggest_alternative_paths(self, frame: np.ndarray, safe_path_grid: np.ndarray, 
                                 cell_width: int, cell_height: int) -> dict:
        """
        Suggest alternative partial paths when the main path is blocked
        
        Args:
            frame: The frame to draw on
            safe_path_grid: Binary grid with safe path information
            cell_width: Width of each grid cell
            cell_height: Height of each grid cell
            
        Returns:
            dict: Navigation guidance information
        """
        # Default guidance information
        guidance_info = {
            "path_found": False,
            "direction": None,
            "next_step": None,
            "distance": 0,
            "navigation_message": "Turn around to find a clear path"
        }
        height, width = frame.shape[:2]
        rows, cols = safe_path_grid.shape
        
        # Find connected safe regions
        from scipy.ndimage import label
        labeled_grid, num_features = label(safe_path_grid)
        
        if num_features == 0:
            return  # No safe areas
        
        # Find the largest connected region
        region_sizes = {}
        for i in range(1, num_features + 1):
            region_sizes[i] = np.sum(labeled_grid == i)
        
        # Sort regions by size
        sorted_regions = sorted(region_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Try to find paths in the largest regions
        for region_id, _ in sorted_regions[:2]:  # Try the two largest regions
            # Create mask for this region
            region_mask = (labeled_grid == region_id).astype(np.uint8)
            
            # Find a path within this region
            partial_path = []
            
            # Find bottom-most point in region
            bottom_points = []
            for c in range(cols):
                for r in range(rows-1, -1, -1):
                    if region_mask[r, c] == 1:
                        bottom_points.append((r, c))
                        break
            
            if not bottom_points:
                continue
                
            # Sort by row (highest row first - closest to bottom)
            bottom_points.sort(reverse=True)
            
            # Start from the bottom center point
            center_idx = min(range(len(bottom_points)), key=lambda i: abs(bottom_points[i][1] - cols//2))
            start_point = bottom_points[center_idx]
            
            # Find top-most point in region
            top_points = []
            for c in range(cols):
                for r in range(rows):
                    if region_mask[r, c] == 1:
                        top_points.append((r, c))
                        break
            
            if not top_points:
                continue
                
            # Sort by row (lowest row first - closest to top)
            top_points.sort()
            
            # Get the top center point
            center_idx = min(range(len(top_points)), key=lambda i: abs(top_points[i][1] - cols//2))
            end_point = top_points[center_idx]
            
            # Simple pathfinding within the region
            import numpy as np
            temp_grid = np.copy(region_mask)
            
            # Find path using BFS
            from collections import deque
            queue = deque([(start_point[0], start_point[1])])
            visited = {(start_point[0], start_point[1]): None}  # (row, col): parent
            
            found_path = False
            while queue and not found_path:
                current_row, current_col = queue.popleft()
                
                if (current_row, current_col) == end_point:
                    found_path = True
                    break
                
                # Check neighbors
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    neighbor_row, neighbor_col = current_row + dr, current_col + dc
                    
                    if (0 <= neighbor_row < rows and 0 <= neighbor_col < cols and 
                        temp_grid[neighbor_row, neighbor_col] == 1 and 
                        (neighbor_row, neighbor_col) not in visited):
                        
                        queue.append((neighbor_row, neighbor_col))
                        visited[(neighbor_row, neighbor_col)] = (current_row, current_col)
            
            # Reconstruct path if found
            if found_path:
                partial_path = []
                current = end_point
                while current is not None:
                    partial_path.append(current)
                    current = visited[current]
                partial_path.reverse()
                
                # Convert to pixel coordinates
                pixel_path = []
                for row, col in partial_path:
                    x = int(col * cell_width + cell_width / 2)
                    y = int(row * cell_height + cell_height / 2)
                    pixel_path.append((x, y))
                
                # Draw dashed path with caution color (yellow/orange)
                if len(pixel_path) >= 2:
                    for i in range(len(pixel_path) - 1):
                        # Draw dashed line segments
                        p1 = pixel_path[i]
                        p2 = pixel_path[i + 1]
                        
                        # Use dashed line for alternative paths
                        dash_length = 10
                        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                        dist = max(1, math.sqrt(dx*dx + dy*dy))
                        dx, dy = dx/dist, dy/dist
                        
                        # Calculate number of segments
                        num_segments = int(dist / dash_length)
                        
                        for j in range(num_segments):
                            if j % 2 == 0:  # Draw every other segment
                                start_x = int(p1[0] + j * dash_length * dx)
                                start_y = int(p1[1] + j * dash_length * dy)
                                end_x = int(min(p1[0] + (j+1) * dash_length * dx, p2[0]))
                                end_y = int(min(p1[1] + (j+1) * dash_length * dy, p2[1]))
                                
                                # Draw thick line with glow effect
                                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 200, 255), 6)  # Outer glow
                                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)  # Inner line
                
                # Add "ALTERNATE ROUTE" indicator at the start
                if len(pixel_path) > 1:
                    alt_text = "ALTERNATE ROUTE"
                    text_x = pixel_path[0][0] + 20
                    text_y = pixel_path[0][1] - 10
                    
                    # Ensure text is within frame
                    text_x = max(10, min(text_x, width - 150))
                    text_y = max(30, min(text_y, height - 10))
                    
                    # Draw text with background
                    text_size = cv2.getTextSize(alt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, 
                                 (text_x - 5, text_y - text_size[1] - 5),
                                 (text_x + text_size[0] + 5, text_y + 5),
                                 (0, 100, 200), -1)
                    cv2.putText(frame, alt_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Determine direction for alternative path
                if len(pixel_path) >= 2:
                    # Get first two points to determine direction
                    x1, y1 = pixel_path[0]
                    x2, y2 = pixel_path[1]
                    
                    # Calculate angle
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    
                    # Determine direction
                    direction = None
                    if -45 <= angle < 45:
                        direction = "right"
                    elif 45 <= angle < 135:
                        direction = "forward"
                    elif angle >= 135 or angle < -135:
                        direction = "left"
                    elif -135 <= angle < -45:
                        direction = "backward"
                    
                    # Create guidance message for alternative path
                    if direction:
                        alt_guidance = f"Try turning {direction} to find a path"
                        guidance_info = {
                            "path_found": True,
                            "direction": direction,
                            "next_step": "turn",
                            "distance": len(pixel_path),
                            "navigation_message": alt_guidance
                        }
                        
                        return guidance_info
                
                # Return default guidance if no direction could be determined
                return guidance_info
        
        # Return default guidance if no alternative paths were found
        return guidance_info
    
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
        result_frame, _ = self.draw_overlays(demo_frame, demo_detections, demo_safe_grid)
        
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
            
            # Draw visual overlays and get navigation guidance
            processed_frame, navigation_guidance = self.draw_overlays(frame, detections, safe_path_grid)
        
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
                "navigation_guidance": navigation_guidance,
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