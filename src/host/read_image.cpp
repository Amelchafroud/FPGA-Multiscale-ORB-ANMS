#include "read_image.h"
#include "color_conversion.h"

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

/*
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}*/

RGBImage::RGBImage(int w, int h)
{
    m_data = (Pixel*) malloc(sizeof(Pixel) * w*h);
    m_own = true;
    m_w = w;
    m_h = h;
}

RGBImage::~RGBImage()
{
    if (m_own)
        free(m_data);
}

GrayscaleImage::GrayscaleImage(int w, int h, unsigned char* data)
{
    m_data = data;
    m_own = false;
    m_w = w;
    m_h = h;
}


GrayscaleImage::GrayscaleImage(int w, int h)
{
    m_data = (unsigned char*) malloc(sizeof(unsigned char) * w*h);
    m_own = true;
    m_w = w;
    m_h = h;
}

GrayscaleImage::~GrayscaleImage()
{
    if (m_own)
        free(m_data);
}



bool isPPM(const std::string& filename) {
    // Check if the filename ends with ".ppm"
    if (filename.length() >= 4) {
        return filename.substr(filename.length() - 4) == ".ppm";
    }
    return false;
}

bool isPGM(const std::string& filename) {
    // Check if the filename ends with ".pgm"
    if (filename.length() >= 4) {
        return filename.substr(filename.length() - 4) == ".pgm";
    }
    return false;
}

/**
 * Load a PPM image
 *
 * return a new image pointer or NULL if an error 
 */
RGBImage* readPPM(const string &filename) 
{
    ifstream file(filename, ios::binary);
    if (!file.is_open()) 
        return NULL;

    string line; getline(file, line);
    if (line != "P6") 
        return NULL;

    while (file.peek() == '#') getline(file, line);
    int maxval; 
    int width;
    int height;
    file >> width >> height >> maxval; file.ignore();
    
    RGBImage* img = new RGBImage(width, height);

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) 
        {
            Pixel* p = &img->m_data[y*width+x];
            file.read((char*) &p->r,1);
            file.read((char*) &p->g,1);
            file.read((char*) &p->b,1);
        }
        
    return img;
}

/**
 * Load a PGM image
 *
 * return a new image pointer or NULL if an error 
 */
RGBImage* readPGM(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        cerr << "failed to open file" << filename << "\n";
        return NULL;
    }

    std::string format;
    std::getline(file, format);
    
    // Check the magic number for P2 (ASCII) or P5 (binary)
    if (format != "P2" && format != "P5") 
    {
        cerr << "format:" << format << "\n";
        return NULL;
    }

    // Skip comments
    while (file.peek() == '#') {
        std::string comment;
        std::getline(file, comment);
    }
    
    int maxval;
    int width;
    int height;
    file >> width >> height >> maxval;
    file.ignore(); // Consume the newline character

    RGBImage* img = new RGBImage(width, height);
    
    //cout << "pgm format " << format << " maxval " << maxval << std::endl;
    
    // Read pixel data based on format
    if (format == "P5") { // Binary
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                char c;
                file.read(&c, 1);
                img->m_data[y*width+x].r = c;
                img->m_data[y*width+x].g = c;
                img->m_data[y*width+x].b = c;
            }
        }
    } else { // P2 ASCII
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int gray_val;
                file >> gray_val;
                Pixel* p = &img->m_data[y*width+x];

                p->r = static_cast<unsigned char>(gray_val);
                p->g = p->r;
                p->b = p->r;
            }
        }
    }
    
    return img;
}

/**
 * Read image trying either PPM or PGM extensions
 */
GrayscaleImage* readImage(const std::string &filename)
{
    stringstream ppm_filename;
    stringstream pgm_filename;
    
    ppm_filename << filename << ".ppm";
    pgm_filename << filename << ".pgm";
    
    RGBImage* img = readPPM(ppm_filename.str());
    
    if (img == NULL)  
        img = readPGM(pgm_filename.str());   
        
    if (img == NULL)
        return NULL;
        
    GrayscaleImage* ret = convertToGrayscale(img);
    
    delete img;
    
    return ret;
}
