import numpy as np
import cv2 as cv
import rasterio as rio
from rasterio.windows import Window
import os
import shutil

def get_tiles(orthomosaic, output_path):
    columns, rows = orthomosaic.width, orthomosaic.height
    tile_size = 512
    
    print(f'Orthomosaic shape: {rows}x{columns}')
    print(f'Tile size: {tile_size}x{tile_size}')
    tiles_x = int(np.ceil(rows/tile_size))
    tiles_y = int(np.ceil(columns/tile_size))
    num_tiles = tiles_x * tiles_y
    print(f'Number of tiles: {tiles_x}x{tiles_y} = {num_tiles}')
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    tiles = 0
    for tile_y in range(0, rows, tile_size):
        for tile_x in range(0, columns, tile_size):
            lrc_x = min(tile_x + tile_size, columns)
            lrc_y = min(tile_y + tile_size, rows)

            window_location = Window(tile_x, tile_y, lrc_x - tile_x, lrc_y - tile_y)
            img = orthomosaic.read(window=window_location)
            
            temp = img.transpose(1, 2, 0)
            t2 = cv.split(temp)
            img_cv = cv.merge([t2[2], t2[1], t2[0]])
            # printing uses a lot of cpu time so it is commented out
            # print(f'Loaded tile at ({tile_x}, {tile_y}) with shape: {img_cv.shape}')
            
            tiles += 1
            # save tile as png with its tile index (tile_0, tile_1, etc.)
            cv.imwrite(f'{output_path}/tile_{tiles}.png', img_cv)
    return


if __name__ == '__main__':
    orthomosaic_path = "mini_project_1/inputs/segmented_orthomosaic.tif"
    output_path = "mini_project_1/outputs"

    with rio.open(orthomosaic_path) as src:
        tiles = get_tiles(src, output_path)
        print (f'Loaded {tiles} tiles from {orthomosaic_path}')