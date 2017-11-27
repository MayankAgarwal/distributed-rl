import numpy as np
from skimage.transform import resize

class Preprocess(object):
    
    def __init__(self, rescale_size):
        self.rescale_size = rescale_size
        
    def __pixel_max(self, frames):
        
        result = []
        
        for i in xrange(1, len(frames)):
            result.append(
                np.maximum(frames[i-1], frames[i]))
            
        return result
                
    def __extract_luminance(self, frames):
        
        # RGB channel has 3 weights for each channel
        luminance_wts = np.zeros((1, 1, 3))
        
        # Refer: https://en.wikipedia.org/wiki/YUV#Conversion_to.2Ffrom_RGB
        luminance_wts[0, 0, 0] = 0.299
        luminance_wts[0, 0, 1] = 0.587
        luminance_wts[0, 0, 2] = 0.114
        
        result = map(lambda x: np.sum(x*luminance_wts, axis=-1), 
                     frames)
        
        return result
    
    def __scale_images(self, frames):
        
        result = map(lambda x: resize(x, self.rescale_size, mode='symmetric'), frames)
        return result
    
    def process_images(self, frames):
        
        frames = self.__pixel_max(frames)
        frames = self.__extract_luminance(frames)
        frames = self.__scale_images(frames)
        
        return np.array(frames)