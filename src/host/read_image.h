#pragma once

#include "config.h"

#include <string>
#include <stdbool.h>

using namespace std;

//Global work buffers
struct Pixel { unsigned char r,g,b; };



class RGBImage
{
public:
    RGBImage(int w, int h);
    virtual ~RGBImage();
    
    Pixel* m_data;
    bool m_own;
    int m_w;
    int m_h;
};

class GrayscaleImage
{
public:
    GrayscaleImage(int w, int h);
    GrayscaleImage(int w, int h, unsigned char* data);
    virtual ~GrayscaleImage();
    
    unsigned char* m_data;
    bool m_own;
    int m_w;
    int m_h;
};

bool isPPM(const string& filename);
bool isPGM(const string& filename);

RGBImage* readPPM(const string &filename);
RGBImage* readPGM(const string &filename);

GrayscaleImage* readImage(const string &filename);
