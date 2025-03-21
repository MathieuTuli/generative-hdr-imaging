# -- main variables
EXE := radiantflow
TEST_EXE := test_runner
SRC_DIRS := ./src
TEST_DIRS := ./tests
LIBS_DIR := ./libs
BUILD_DIR := ./build
UNAME_S := $(shell uname -s)
DEPS_DIR := ./dependencies/
export UNAME_S

# Find all the C and C++ files we want to compile
SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.s')
TEST_SRCS := $(shell find $(TEST_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.s')
# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS := $(shell find $(SRC_DIRS) $(TEST_DIRS) -type d)
INC_DIRS += ./include
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# -- build variables
# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CXXFLAGS = -std=c++17 -gdwarf-4 -Wall -Wformat
CXXFLAGS += $(INC_FLAGS) -MMD -MP
LIBS :=

# -- system based args
ifeq ($(UNAME_S), Linux) #LINUX
	LIBS += -L/usr/local/lib
	CXXFLAGS += -I/usr/local/include
endif

ifeq ($(UNAME_S), Darwin) #APPLE
	LIBS += -L/usr/local/lib -L/opt/homebrew/lib 
	CXXFLAGS += -I/usr/local/include -I/opt/homebrew/include
endif

# -- dependencies
ifeq ($(UNAME_S), Linux) #LINUX
	# OpenGL and GLFW first
	LIBS += -lGL $(shell pkg-config --static --libs glfw3)
	CXXFLAGS += $(shell pkg-config --cflags glfw3)
	
	# GTK
	CXXFLAGS += $(shell pkg-config --cflags --libs gtk+-3.0)
	
	# OpenBLAS and LAPACK dependencies
	LIBS += -lopenblas -llapack -lgfortran -lquadmath
	
	# OpenCV last since it might depend on BLAS/LAPACK
	LIBS += $(shell pkg-config --libs opencv4)
	CXXFLAGS += $(shell pkg-config --cflags opencv4)

	# catch2
	LIBS += $(shell pkg-config --libs catch2-with-main)
	CXXFLAGS += $(shell pkg-config --cflags catch2-with-main)

	# catch2
	LIBS += $(shell pkg-config --libs libpng)
	CXXFLAGS += $(shell pkg-config --cflags libpng)
endif

ifeq ($(UNAME_S), Darwin) #APPLE
	# imgui
	CXXFLAGS += -fno-omit-frame-pointer

	LIBS += -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
	LIBS += -lglfw
	LIBS += -L /opt/homebrew/opt/ffmpeg/lib -lavformat -lpostproc -lavcodec -lswscale -lavfilter -lavutil -lswresample -lavdevice
	LIBS += -framework AppKit -framework UniformTypeIdentifiers
	# LIBS += -lglfw3
	LIBS += $(shell pkg-config --libs opencv4)
	CXXFLAGS += $(shell pkg-config --cflags opencv4)

	# catch2
	LIBS += $(shell pkg-config --libs catch2-with-main)
	CXXFLAGS += $(shell pkg-config --cflags catch2-with-main)

	# catch2
	LIBS += $(shell pkg-config --libs libpng)
	CXXFLAGS += $(shell pkg-config --cflags libpng)
endif

IMGUI_DIR := $(DEPS_DIR)/imgui
SRCS += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
SRCS += $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
CXXFLAGS += -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends

EXIF_DIR := $(DEPS_DIR)/exiftool
SRCS += $(shell find $(EXIF_DIR)/src -name '*.cpp' -or -name '*.c')
CXXFLAGS += -I$(EXIF_DIR)/inc

# -- build rules
# Prepends BUILD_DIR and appends .o to every src file
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
# For tests, we want all source files except main.cpp, plus all test files
SRC_OBJS_NO_MAIN := $(filter-out %/main.cpp.o,$(OBJS))
TEST_OBJS := $(SRC_OBJS_NO_MAIN) $(TEST_SRCS:%=$(BUILD_DIR)/%.o)
# String substitution (suffix version without %).
DEPS := $(OBJS:.o=.d)
CFLAGS = $(CXXFLAGS)

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

all: $(BUILD_DIR)/$(EXE)
	@echo "OBJS: $(OBJS)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo All build complete for $(UNAME_S)

$(BUILD_DIR)/$(TEST_EXE): $(TEST_OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

.PHONY: clean test debug

test: $(BUILD_DIR)/$(TEST_EXE)
	@echo Test build complete for $(UNAME_S)

clean:
	rm -f $(BUILD_DIR)/$(EXE) $(BUILD_DIR)/$(TEST_EXE) $(OBJS) $(TEST_OBJS)

debug:
	@echo "Include directories:"
	@echo $(INC_DIRS)
	@echo "Include flags:"
	@echo $(INC_FLAGS)
