#include "color_conversion.h"
#include "read_image.h"

GrayscaleImage* convertToGrayscale(RGBImage* img) 
{
    GrayscaleImage* ret = new GrayscaleImage(img->m_w, img->m_h);
    
    int w = img->m_w;
    int h = img->m_h;
    
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            ret->m_data[y*w+x] = (unsigned char) (0.299 * img->m_data[y*w+x].r +
                             0.587 * img->m_data[y*w+x].g +
                             0.114 * img->m_data[y*w+x].b);            
        }
        
    return ret;
}
