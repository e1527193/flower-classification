CC=gcc
CXX=g++
SRCS=capture.cc
OBJS=$(subst .cc,.o,$(SRCS))

all: $(SRCS)
	$(CXX) -std=c++11 -Wall -I/usr/lib/opencv -I/usr/include/opencv4 $< -L/usr/lib -lopencv_core -lopencv_videoio -lopencv_imgcodecs -o capture

run: capture
	LD_PRELOAD=/usr/lib/aarch64-linux-gnu/tegra/libnvjpeg.so ./capture

clean:
	rm -f $(OBJS)

distclean: clean
	rm -f capture
