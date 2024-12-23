import os
import cv2

def create_video_from_images(input_dir, output_file, fps=30):
    # Get list of all PNG files in the directory
    image_files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.png')]
    if not image_files:
        raise ValueError("No PNG files found in the specified directory.")
    
    # Read the first image to get the frame size
    first_image_path = os.path.join(input_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Font settings for the filename text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black background for text
    
    for file_name in image_files:
        image_path = os.path.join(input_dir, file_name)
        img = cv2.imread(image_path)
        
        # Add filename at the bottom right
        text_size = cv2.getTextSize(file_name, font, font_scale, font_thickness)[0]
        text_x = width - text_size[0] - 10
        text_y = height - 10
        cv2.rectangle(img, 
                      (text_x - 5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + 5), 
                      bg_color, -1)
        cv2.putText(img, file_name, (text_x, text_y), font, font_scale, text_color, font_thickness)
        
        # Write the frame to the video
        out.write(img)
    
    # Release the video writer
    out.release()
    print(f"Video saved to {output_file}")

# Example usage:
input_directory = "/net/ens/am4ip/datasets/project-dataset/rainy_images/"  # Replace with your directory path
output_video_file = "output_video_rainy.mp4"  # Replace with your desired output filename
create_video_from_images(input_directory, output_video_file)
