CC=gcc
CXX=g++
RM=rm -f
SRCS=autofocus.cc
OBJS=$(subst .cc,.o,$(SRCS))

all: $(SRCS)
	$(CXX) -std=c++11 -Wall -I/usr/lib/opencv -I/usr/include/opencv4 $< -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -o autofocus

run: autofocus
	LD_PRELOAD=/usr/lib/aarch64-linux-gnu/tegra/libnvjpeg.so ./autofocus

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) autofocus
