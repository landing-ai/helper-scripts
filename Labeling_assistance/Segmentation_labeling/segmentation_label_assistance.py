import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from landingai.predict import Predictor
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VOCDatasetConverter:
    def __init__(self, source_image_folder, output_dataset_folder):
        self.source_folder = Path(source_image_folder)
        self.dataset_folder = Path(output_dataset_folder)
        self.images_folder = self.dataset_folder / "Images"
        self.segmentations_folder = self.dataset_folder / "Segmentations"
        self.defect_map_path = self.dataset_folder / "defect_map.json"
        
        self.endpoint_id = "ENTER_YOUR_OWN_ENDPOINT_HERE"
        self.api_key = "ENTER_YOUR_OWN_KEY_HERE"
        self.predictor = Predictor(self.endpoint_id, api_key=self.api_key)
        
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.defect_map = {"0": "ok"}
        self.class_to_id = {"ok": 0}
        logging.info(f"Initial defect_map: {self.defect_map}")
        
    def setup_folders(self):
        self.dataset_folder.mkdir(exist_ok=True)
        self.images_folder.mkdir(exist_ok=True)
        self.segmentations_folder.mkdir(exist_ok=True)
        
    def get_image_files(self):
        return [f for f in self.source_folder.iterdir() 
                if f.suffix.lower() in self.image_extensions]
    
    def update_defect_map(self, class_name):
        if class_name not in self.class_to_id:
            new_id = len(self.class_to_id)
            self.class_to_id[class_name] = new_id
            self.defect_map[str(new_id)] = class_name
 
    def process_single_image(self, image_path):
       image = Image.open(image_path)
       predictions = self.predictor.predict(image)
       logging.info(f"Image size: {image.size}")
       mask = np.zeros(image.size[::-1], dtype=np.uint8)
       
       for pred in predictions:
           # Check prediction structure
           logging.info(f"Processing prediction: {pred}")
           class_name = pred.label_name
           
           # Update class mapping
           if class_name not in self.class_to_id:
               new_id = len(self.class_to_id)
               self.class_to_id[class_name] = new_id
               self.defect_map[str(new_id)] = class_name
               logging.info(f"Added class: {class_name} with ID {new_id}")
           
           # Create mask
           if hasattr(pred, 'encoded_mask'):
               binary_mask = self.decode_mask(
                   pred.encoded_mask,
                   pred.encoding_map,
                   pred.mask_shape
               )
               mask[binary_mask > 0] = self.class_to_id[class_name]
               logging.info(f"Updated mask for {class_name}")

       logging.info(f"Final mask values: {np.unique(mask)}")
       logging.info(f"Current defect_map: {self.defect_map}")
       return image, mask
       
   
    def decode_mask(self, encoded_mask, encoding_map, mask_shape):
        try:
            decoded = ''
            parts = encoded_mask.replace('Z', 'Z ').replace('N', 'N ').split()
            for item in parts:
                count = int(item[:-1])
                char = item[-1]
                decoded += str(encoding_map[char]) * count
            
            mask = np.array(list(map(int, decoded))).reshape(mask_shape)
            logging.info(f"Decoded mask shape: {mask.shape}, values: {np.unique(mask)}")
            return mask
        except Exception as e:
            logging.error(f"Mask decoding error: {e}")
            raise     
            
    def save_processed_data(self, image_path, image, mask):
        new_image_path = self.images_folder / image_path.name
        shutil.copy2(image_path, new_image_path)
        
        mask_name = image_path.stem + '.png'
        mask_path = self.segmentations_folder / mask_name
        mask_image = Image.fromarray(mask)
        mask_image.save(mask_path)
    
    def save_defect_map(self):
        with open(self.defect_map_path, 'w') as f:
            json.dump(self.defect_map, f)
    
    def convert(self):
        self.setup_folders()
        image_files = self.get_image_files()
        logging.info(f"Found image files: {[f.name for f in image_files]}")
        
        for image_path in image_files:
            try:
                image, mask = self.process_single_image(image_path)
                self.save_processed_data(image_path, image, mask)
            except Exception as e:
                logging.error(f"Error processing {image_path.name}: {str(e)}")
                continue
        
        self.save_defect_map()
        logging.info(f"Final defect_map saved: {self.defect_map}")

    # [Rest of the methods remain the same]

def main():
    source_folder = "images"
    output_folder = "dataset"
    converter = VOCDatasetConverter(source_folder, output_folder)
    converter.convert()

if __name__ == "__main__":
    main()