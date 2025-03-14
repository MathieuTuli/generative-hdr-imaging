# -- main variables
EXE := main
SRC_DIRS := ./src
LIBS_DIR := ./libs
BUILD_DIR := ./build
UNAME_S := $(shell uname -s)

# -- search variables
# Find all the C and C++ files we want to compile
SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.s')
# Prepends BUILD_DIR and appends .o to every src file
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
# String substitution (suffix version without %).
DEPS := $(OBJS:.o=.d)
# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS := $(shell find $(SRC_DIRS) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# -- build variables
# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CPPFLAGS := $(INC_FLAGS) -MMD -MP
CXXFLAGS = -std=c++17 -g -Wall -Wformat
LIBS :=

# -- system based args
ifeq ($(UNAME_S), Linux) #LINUX
	LIBS += -L/usr/local/lib
	CXXFLAGS += -I/usr/local/include
	LIBS += -lGL # $(shell pkg-config --static --libs glfw3)
	# LIBS += $(shell pkg-config --libs opencv4)
	# LIBS += ${LIBS_DIR}/nfd_linux/libnfd.a
	# CXXFLAGS += $(shell pkg-config --cflags glfw3)
	# CXXFLAGS += $(shell pkg-config --cflags opencv4)
	# CXXFLAGS += $(shell pkg-config --cflags --libs gtk+-3.0)
	CFLAGS = $(CXXFLAGS)
endif

ifeq ($(UNAME_S), Darwin) #APPLE
	LIBS += -L/usr/local/lib -L/opt/homebrew/lib 
	CXXFLAGS += -I/usr/local/include -I/opt/homebrew/include
	# LIBS += -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
	# LIBS += -lglfw
	# LIBS += ${LIBS_DIR}/nfd/libnfd.a
	# LIBS += $(shell pkg-config --libs opencv4)
	# LIBS += -L /opt/homebrew/opt/ffmpeg/lib -lavformat -lpostproc -lavcodec -lswscale -lavfilter -lavutil -lswresample -lavdevice
	# LIBS += -framework AppKit -framework UniformTypeIdentifiers
	#LIBS += -lglfw3
	# CXXFLAGS += $(shell pkg-config --cflags opencv4)
	CFLAGS = $(CXXFLAGS)
endif

# -- build rules

# The final build step.
$(BUILD_DIR)/$(EXE): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

# Build step for C source
$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CXXFLAGS) $(CFLAGS) -c -o $@ $< 

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $< 

all: $(EXE)
	@echo Build complete for $(UNAME_S)

clean:
	rm -f $(EXE) $(OBJS)
