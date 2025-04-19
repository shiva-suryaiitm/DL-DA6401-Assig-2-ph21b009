import wandb
from PIL import Image
import numpy as np
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# logging the image into wandb
wandb.login(key = "878d8077f96e35def15932addea71a0302e0dede")
wandb.init(project="DA6401-assig-2", name = "Test_image_logging_3", reinit=True)
test_prediction_image = Image.open(root_dir + "/images/test_data_predictions.png") 

# Log the image to wandb
wandb.log({
    "test_pred_image_3": wandb.Image(test_prediction_image, caption="Test prediction image 3")
    
})

wandb.finish()
print(f"finished logging image")