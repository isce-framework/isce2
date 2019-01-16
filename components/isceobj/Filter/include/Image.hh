#include <sys/types.h>
#include <sys/stat.h>
#ifdef sun
    #include <fcntl.h>
#else
    #include <sys/fcntl.h>
    #include <unistd.h>
#endif
#include <sys/mman.h>
#include <iostream>
#include <string>

template <typename T>
class Image
{
 private:
  int width;
  int height;
  char *filename;
  int openFlags;
  int mapFlags;
  int fd;
  T *image;
  void createMap();
  void testCoordinates(int x, int y);
 public:
  Image(char *filename, const char *mode, int width, int height);
  ~Image();
  int getHeight();
  int getWidth();
  T getValue(int x, int y);
  void setValue(int x, int y, T val);
};

template <typename T>
Image<T>::Image(char *filename,const char *mode,int width,int height)
{
  this->filename = filename;
  this->width = width;
  this->height = height;

  std::string read = "r";
  std::string write = "w";
  // Convert the mode to an oflag for open and a flag for mmap
  if (read.compare(mode) == 0)
    {
      this->openFlags = O_RDONLY;
      this->mapFlags = PROT_READ;
    }
  else if (write.compare(mode) == 0)
    {
      this->openFlags = (O_RDWR | O_CREAT);
      this->mapFlags = (PROT_READ | PROT_WRITE);
    }
  try {
	  this->createMap();
  } catch (const char *e) {
	  std::cerr << e << std::endl;
  }
}

template <typename T>
Image<T>::~Image()
{
  size_t size = (size_t)(this->width*this->height*sizeof(T));

  munmap(this->image,size);
  close(this->fd);
}

template <typename T>
int Image<T>::getWidth()
{
  return this->width;
}

template <typename T>
int Image<T>::getHeight()
{
  return this->height;
}

template <typename T>
void Image<T>::createMap()
{
  size_t size = (size_t)(this->width*this->height*sizeof(T));

  // If we are creating this image for the first time, we need to "create" space
  // for it on the drive
  if ( this->openFlags == (O_RDWR | O_CREAT) )
    {
      this->fd = open(this->filename, this->openFlags, (mode_t)0600);
      int status = ftruncate(this->fd,size);
      if (status == -1) {throw "Unable to create file";}
    }
  else
    {
      this->fd = open(this->filename, this->openFlags);
    }
  this->image = (T *)mmap(0, size, this->mapFlags, MAP_SHARED, this->fd,0);
  if (this->image == MAP_FAILED)
    {
      throw "Memory mapping failed";
    }
}

template <typename T>
T Image<T>::getValue(int x, int y)
{
 this->testCoordinates(x,y);
 return this->image[y*this->width + x];
}

template <typename T>
void Image<T>::setValue(int x, int y, T val)
{
  this->testCoordinates(x,y);
  this->image[y*this->width + x] = val;
}

template <typename T>
void Image<T>::testCoordinates(int x, int y)
{
 if (x > this->width)
   {
     throw "X coordinate out of bounds";
   }
 if (y > this->height)
   {
     throw "Y coordinate out of bounds";
   }
}
