# Computer Vision Processing Pipeline
# Implements YOLOv11 object detection, optical flow analysis, and visual overlay rendering

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
from ultralytics import YOLO

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
    
    def __init__(self, model_path: str = "yolov11n.pt", 
                 stationary_threshold: float = 2.0,
                 confidence_threshold: float = 0.5):
        """
        Initialize vision processor with YOLOv11 model and configuration
        
        Args:
            model_path: Path to YOLOv11 model file (defaults to nano model)
            stationary_threshold: Motion magnitude threshold for stationary detection
            confidence_threshold: Minimum confidence for object detection
        """
        import os
        
        # Handle model path - check multiple possible locations
        possible_paths = [
            model_path,  # As provided
            os.path.join(os.path.dirname(__file__), model_path),  # Same directory as this file
            os.path.join("backend", model_path),  # Backend subdirectory
            os.path.join("..", "backend", model_path),  # Parent/backend directory
        ]
        
        self.model_path = model_path  # Default
        for path in possible_paths:
            if os.path.exists(path):
                self.model_path = path
                break
        
        self.stationary_threshold = stationary_threshold
        self.confidence_threshold = confidence_threshold
        
        # Initialize YOLOv11 model
        self.yolo_model: Optional[YOLO] = None
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
        
        logger.info(f"VisionProcessor initialized with model: {model_path}")
    
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
        Initialize YOLOv11 model loading and configuration with fallback handling
        Requirement 3.1: Initialize YOLOv11 model loading and configuration
        """
        try:
            logger.info(f"Loading YOLOv11 model: {self.model_path}")
            self.yolo_model = YOLO(self.model_path)
            
            # Verify model loaded successfully
            if self.yolo_model is not None:
                logger.info("YOLOv11 model loaded successfully")
                # Log model information
                logger.info(f"Model classes: {len(self.yolo_model.names)} classes")
                logger.info(f"Model device: {self.yolo_model.device}")
                
                # Test model with dummy input to ensure it's working
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                test_results = self.yolo_model(test_frame, verbose=False)
                logger.info("YOLOv11 model test inference successful")
                
            else:
                raise RuntimeError("Failed to load YOLOv11 model")
                
        except Exception as e:
            logger.error(f"Error loading YOLOv11 model: {e}")
            self.yolo_model = None
            
            # Initialize fallback detection system
            self._initialize_fallback_detection()
            
            # Don't raise exception - allow system to continue with fallback
            logger.warning("YOLOv11 model initialization failed - using fallback detection system")
    
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
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        
        # Define navigation-relevant obstacle classes (COCO dataset classes)
        navigation_obstacles = {
            # People and animals
            0: 'person',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
            21: 'bear', 22: 'zebra', 23: 'giraffe',
            
            # Furniture and objects that block paths
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
            68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
            76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
            
            # Vehicles and large objects
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
            
            # Sports equipment and obstacles
            32: 'sports ball', 37: 'skateboard', 38: 'surfboard',
            
            # Bags and containers
            24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
            28: 'suitcase',
            
            # Other common obstacles
            39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup',
            43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl'
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
                    class_name = self.yolo_model.names[class_id]
                    
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
            
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.previous_frame,
                gray_current,
                points_to_track,
                None,
                **self.optical_flow_params
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
                    is_stationary=is_stationary,
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
        Create enhanced visual overlay rendering system for ground-level navigation
        Shows clear walking paths and navigation obstacles
        
        Args:
            frame: Input video frame
            detections: List of detected obstacles
            safe_path_grid: Binary grid with safe path information
            
        Returns:
            Frame with enhanced visual overlays for navigation
        """
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Calculate grid cell dimensions
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # Create separate overlay for path visualization
        path_overlay = np.zeros_like(frame)
        
        # Draw safe navigation paths with enhanced visibility
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                if safe_path_grid[row, col] == 1:  # Safe cell
                    # Draw bright green path with gradient effect
                    cv2.rectangle(path_overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                    # Add border for clarity
                    cv2.rectangle(path_overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)
                else:  # Blocked cell
                    # Draw subtle red overlay for blocked areas
                    cv2.rectangle(path_overlay, (x1, y1), (x2, y2), (0, 0, 100), -1)
        
        # Blend path overlay with original frame
        alpha = 0.4  # Increased transparency for better visibility
        overlay_frame = cv2.addWeighted(frame, 1 - alpha, path_overlay, alpha, 0)
        
        # Draw navigation direction arrow if clear path exists
        if self.has_clear_path(safe_path_grid):
            # Draw forward arrow in center
            center_x = width // 2
            arrow_start_y = int(height * 0.8)  # Bottom area
            arrow_end_y = int(height * 0.4)    # Middle area
            
            # Draw thick green arrow
            cv2.arrowedLine(overlay_frame, (center_x, arrow_start_y), 
                          (center_x, arrow_end_y), (0, 255, 0), 8, tipLength=0.3)
            
            # Add "CLEAR PATH" text
            cv2.putText(overlay_frame, "CLEAR PATH", (center_x - 80, arrow_end_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Draw warning for blocked path
            cv2.putText(overlay_frame, "PATH BLOCKED", (width//2 - 100, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Draw enhanced obstacle detection boxes
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw thick red bounding box for obstacles
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Add obstacle label with enhanced visibility
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Draw label background for better readability
            label_bg_x1 = x1
            label_bg_y1 = y1 - label_size[1] - 10
            label_bg_x2 = x1 + label_size[0] + 10
            label_bg_y2 = y1
            
            cv2.rectangle(overlay_frame, (label_bg_x1, label_bg_y1), 
                         (label_bg_x2, label_bg_y2), (0, 0, 255), -1)
            
            # Draw white text on red background
            cv2.putText(overlay_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add navigation info overlay
        info_text = f"Objects: {len(detections)} | Safe Cells: {np.sum(safe_path_grid)}/{safe_path_grid.size}"
        cv2.putText(overlay_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add ground level indicator
        ground_line_y = int(height * 0.3)
        cv2.line(overlay_frame, (0, ground_line_y), (width, ground_line_y), 
                (255, 255, 0), 2)
        cv2.putText(overlay_frame, "GROUND LEVEL", (10, ground_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
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
    """Get or create global vision processor instance"""
    global vision_processor
    if vision_processor is None:
        import os
        # Use absolute path to model file in backend directory
        model_path = os.path.join(os.path.dirname(__file__), 'yolo11n.pt')
        vision_processor = VisionProcessor(model_path=model_path)
    return vision_processor