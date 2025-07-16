from typing import Tuple

def find_def_location_grid(matrix_dims: Tuple[int, int], 
                           bounding_box: Tuple[int, int, int, int], 
                           image_dims: Tuple[int, int]) -> Tuple[int, int]:
    """
    Determines the grid cell location of a bounding box within an image.

    This function calculates which cell of a defined matrix grid the top-left corner of
    the bounding box occupies in an image. The image is divided into a grid of cells based 
    on the specified matrix dimensions, and the bounding box coordinates are used to 
    determine its position.

    Args:
        matrix_dims (Tuple[int, int]): The dimensions of the grid, where the first value is the 
            number of columns and the second value is the number of rows (matrix_x, matrix_y).
        bounding_box (Tuple[int, int, int, int]): The bounding box coordinates as 
            (x1, y1, x2, y2), where (x1, y1) are the top-left corner and (x2, y2) are the 
            bottom-right corner.
        image_dims (Tuple[int, int]): The dimensions of the image as (image_width, image_height).

    Returns:
        Tuple[int, int]: The (x_slice, y_slice) representing the column and row of the grid 
            where the top-left corner of the bounding box is located. The indices are 1-based 
            (starting from 1).

    Example:
        >>> find_def_location_grid((4, 4), (50, 50, 100, 100), (400, 400))
        (1, 1)
    
    """
    # Unpacking the parameters
    matrix_x, matrix_y = matrix_dims
    x1, y1, x2, y2 = bounding_box  # Landing AI's Object Detection (OD) coordinate results
    image_width, image_height = image_dims

    # Calculating the dimensions of each grid slice
    slice_width = image_width / matrix_x
    slice_height = image_height / matrix_y

    # Find the grid cell based on the top-left corner of the bounding box
    # Adding 1 to adjust the index so it starts from 1
    x_slice = int(x1 // slice_width) + 1
    y_slice = int(y1 // slice_height) + 1

    return (x_slice, y_slice)
