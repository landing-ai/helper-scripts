import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
from typing import Dict, Tuple, List

def thin_line(binary_image: np.ndarray) -> np.ndarray:
    """
    More conservative thinning function
    """
    # Make sure input is binary
    binary = (binary_image > 0).astype(np.uint8) * 255
    
    # Create a copy to avoid modifying original
    result = binary.copy()
    
    # Use smaller kernel for more precise erosion
    kernel = np.ones((2,2), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    
    return eroded

class SegmentationProcessor:
    def __init__(self, json_data: dict):
        self.json_data = json_data
        self.image_height = json_data['predictions']['imageHeight']
        self.image_width = json_data['predictions']['imageWidth']
        self.central_line = ((17, 345), (521, 345))
    
    def decode_rle(self, bitmap: str, encoding_map: dict) -> np.ndarray:
        """Decode RLE encoded bitmap to numpy array"""
        mask = np.zeros((self.image_height * self.image_width), dtype=np.uint8)
        current_pos = 0
        
        # Debug prints
        print("Encoding map:", encoding_map)
        print("First 100 chars of bitmap:", bitmap[:100])
        
        # Create reverse mapping for the encoding
        reverse_map = {}
        for key, value in encoding_map.items():
            if isinstance(value, int):
                reverse_map[key] = value
            else:
                # Handle the case where value might be a string
                reverse_map[key] = int(value) if value.isdigit() else 1
        
        print("Reverse map:", reverse_map)
        
        # Process each character in the bitmap string
        count = ''
        total_pixels = 0
        
        try:
            for char in bitmap:
                if char.isdigit():
                    count += char
                else:
                    if count:
                        pixel_count = int(count)
                        pixel_value = reverse_map.get(char, 0)
                        
                        # Debug print for first few iterations
                        if total_pixels < 1000:
                            print(f"Processing: count={pixel_count}, char={char}, value={pixel_value}")
                        
                        # Ensure we don't exceed array bounds
                        if current_pos + pixel_count <= len(mask):
                            mask[current_pos:current_pos + pixel_count] = pixel_value
                            current_pos += pixel_count
                            total_pixels += pixel_count
                        count = ''
        except Exception as e:
            print(f"Error in RLE decoding: {str(e)}")
            raise
        
        print(f"Total pixels processed: {total_pixels}")
        print(f"Non-zero pixels in mask: {np.sum(mask > 0)}")
        
        # Reshape to 2D array
        reshaped_mask = mask.reshape((self.image_height, self.image_width))
        print(f"Shape of reshaped mask: {reshaped_mask.shape}")
        print(f"Non-zero pixels in reshaped mask: {np.sum(reshaped_mask > 0)}")
        
        return reshaped_mask
    
    
    


    def process_edge_lines(self) -> Tuple[np.ndarray, np.ndarray]:
        """Process edge lines to ensure 1-pixel width and split into top/bottom edges"""
        edge_mask = None
        
        # Get the single Edge mask
        for pred_id, pred in self.json_data['predictions']['bitmaps'].items():
            if pred['labelName'] == 'Edge':
                edge_mask = self.decode_rle(pred['bitmap'], 
                                       self.json_data['predictions']['encoding']['options']['map'])
                      
                break
                #debugging code
                print("Sample of decoded mask values:")
                print(edge_mask[edge_mask > 0][:10])  # Print first 10 non-zero values
        
        if edge_mask is None:
            raise ValueError("No edge mask found in predictions")

        # Convert to uint8 and ensure binary
        edge_mask = (edge_mask > 0).astype(np.uint8) * 255
        
        # Debug print
        print(f"Number of edge pixels before thinning: {np.sum(edge_mask > 0)}")
        
        # Simple thinning or skip thinning if mask is already thin
        if np.sum(edge_mask > 0) > 0:
            thinned_edge = edge_mask
        else:
            raise ValueError("No edge pixels found in original mask")
        
        # Find all edge pixels
        edge_pixels = np.where(thinned_edge > 0)
        if len(edge_pixels[0]) == 0:
            raise ValueError("No edge pixels found after thinning")
        
        # Debug print
        print(f"Number of edge pixels after thinning: {len(edge_pixels[0])}")
        
        # Split based on the central line y-coordinate
        central_y = self.central_line[0][1]
        
        # Create separate masks for top and bottom edges
        edge_top = np.zeros_like(thinned_edge)
        edge_bottom = np.zeros_like(thinned_edge)
        
        # Count pixels for each section
        top_count = 0
        bottom_count = 0
        
        # Fill the masks based on y-coordinate
        for y, x in zip(edge_pixels[0], edge_pixels[1]):
            if y < central_y:
                edge_top[y, x] = 255
                top_count += 1
            else:
                edge_bottom[y, x] = 255
                bottom_count += 1
        
        # Debug print
        print(f"Top edge pixels: {top_count}")
        print(f"Bottom edge pixels: {bottom_count}")
        
        # Verify both masks have content with more informative error
        if not np.any(edge_top):
            raise ValueError("No pixels found in top edge mask")
        if not np.any(edge_bottom):
            raise ValueError("No pixels found in bottom edge mask")
        
        return edge_top > 0, edge_bottom > 0
    
    def process_YOUR_DEFECT(self) -> List[np.ndarray]:
        """Process YOUR_DEFECT masks, returns list of individual YOUR_DEFECT instance masks"""
        YOUR_DEFECT_masks = []
        
        for pred_id, pred in self.json_data['predictions']['bitmaps'].items():
            if pred['labelName'] == 'CHIPPING':
                # Decode the full mask
                full_mask = self.decode_rle(pred['bitmap'],
                                       self.json_data['predictions']['encoding']['options']['map'])
                
                # Find connected components
                num_labels, labels = cv2.connectedComponents(full_mask.astype(np.uint8))
                
                # Create individual masks for each instance
                for label in range(1, num_labels):  # Start from 1 to skip background
                    instance_mask = (labels == label).astype(np.uint8)
                    YOUR_DEFECT_masks.append(instance_mask)
                    
                print(f"Found {len(YOUR_DEFECT_masks)} YOUR_DEFECT instances")
                break
                
        return YOUR_DEFECT_masks if YOUR_DEFECT_masks else None
    
    def measure_distances(self, edge_top: np.ndarray, edge_bottom: np.ndarray, 
                         YOUR_DEFECT_masks: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Measure distances between each YOUR_DEFECT instance and edges.
        Returns list of distances for each YOUR_DEFECT instance.
        """
        all_distances = []
        if YOUR_DEFECT_masks is None:
            return all_distances
            
        for idx, YOUR_DEFECT_mask in enumerate(YOUR_DEFECT_masks):
            distances = {}
            YOUR_DEFECT_coords = np.where(YOUR_DEFECT_mask)
            
            # Split YOUR_DEFECT area by central line
            above_central = YOUR_DEFECT_coords[0] < self.central_line[0][1]
            below_central = YOUR_DEFECT_coords[0] >= self.central_line[0][1]
            
            # Handle top edge distance
            if np.any(above_central):
                min_y_YOUR_DEFECT = np.min(YOUR_DEFECT_coords[0][above_central])
                edge_top_coords = np.where(edge_top)
                if len(edge_top_coords[0]) > 0:
                    edge_top_y = np.max(edge_top_coords[0])
                    if min_y_YOUR_DEFECT < edge_top_y:
                        distances['top_distance'] = -(edge_top_y - min_y_YOUR_DEFECT)
                    else:
                        distances['top_distance'] = edge_top_y - min_y_YOUR_DEFECT
            
            # Handle bottom edge distance
            if np.any(below_central):
                max_y_YOUR_DEFECT = np.max(YOUR_DEFECT_coords[0][below_central])
                edge_bottom_coords = np.where(edge_bottom)
                if len(edge_bottom_coords[0]) > 0:
                    edge_bottom_y = np.min(edge_bottom_coords[0])
                    if max_y_YOUR_DEFECT > edge_bottom_y:
                        distances['bottom_distance'] = max_y_YOUR_DEFECT - edge_bottom_y
                    else:
                        distances['bottom_distance'] = -(edge_bottom_y - max_y_YOUR_DEFECT)
            
            # Add instance index to distances
            distances['instance_id'] = idx
            all_distances.append(distances)
                
        return all_distances
    
    
    
    def calculate_areas(self) -> Dict[str, float]:
        """Calculate areas for each instance"""
        areas = {}
        for pred_id, pred in self.json_data['predictions']['bitmaps'].items():
            mask = self.decode_rle(pred['bitmap'],
                                 self.json_data['predictions']['encoding']['options']['map'])
            areas[pred['labelName']] = float(np.sum(mask))
        return areas
    

    def visualize_results(self, original_image_path: str, output_path: str):
        """Visualize segmentation results"""
        try:
            # Load original image
            image = Image.open(original_image_path)
            draw = ImageDraw.Draw(image)
            
            # Draw edges (red)
            edge_top, edge_bottom = self.process_edge_lines()
            for y, x in zip(*np.where(edge_top)):
                draw.point((x, y), fill='red')
            for y, x in zip(*np.where(edge_bottom)):
                draw.point((x, y), fill='red')
                
            # Draw YOUR_DEFECT instances (different shades of green)
            YOUR_DEFECT_masks = self.process_YOUR_DEFECT()
            if YOUR_DEFECT_masks is not None:
                colors = ['green', 'lime', 'forestgreen', 'seagreen', 'mediumseagreen']
                for idx, mask in enumerate(YOUR_DEFECT_masks):
                    color = colors[idx % len(colors)]  # Cycle through colors
                    for y, x in zip(*np.where(mask)):
                        draw.point((x, y), fill=color)
                
            # Save result
            image.save(output_path)
        except Exception as e:
            print(f"Error in visualization: {str(e)}")

def main(json_path: str, image_path: str):
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            
        # Create processor instance
        processor = SegmentationProcessor(json_data)
        
        # Process edges
        edge_top, edge_bottom = processor.process_edge_lines()
        
        # Process YOUR_DEFECT
        YOUR_DEFECT_masks = processor.process_YOUR_DEFECT()
        
        # Measure distances
        distances = processor.measure_distances(edge_top, edge_bottom, YOUR_DEFECT_masks)
        
        # Calculate areas
        areas = processor.calculate_areas()
        
        # Visualize results
        output_path = image_path.rsplit('.', 1)[0] + '_rendering.' + image_path.rsplit('.', 1)[1]
        processor.visualize_results(image_path, output_path)
        
        return {
            'distances': distances,
            'areas': areas
        }
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return None

if __name__ == "__main__":
    results = main('input/2024-12-20T00-46-32-807Z-FAIL_9N80A110_CH2-2023_12_22_15_19_17_881.jpg.json', 
                  'input/2024-12-20T00-46-32-807Z-FAIL_9N80A110_CH2-2023_12_22_15_19_17_881.jpg')
    if results:
        print("Distances:", results['distances'])
        print("Areas:", results['areas'])
        
        
        
