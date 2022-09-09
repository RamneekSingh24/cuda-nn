INCLUDES := -I./include
LDFLAGS  := -lm

SRC := ./src
INCLUDE := ./include
OBJ := ./obj
CC  := gcc
CXXFLAGS := -O3
NVCC := nvcc
NVCCFLAGS := -O3 -m64 -rdc=true


SOURCES := $(wildcard $(SRC)/*.cpp)
HEADERS := $(wildcard $(INCLUDE)/*.h)
OBJECTS := $(patsubst $(SRC)/%.cpp, $(OBJ)/%.o, $(SOURCES))

CU_SOURCES := $(wildcard $(SRC)/*.cu)
CU_HEADERS := $(wildcard $(INCLUDE)/*.cuh)
CU_OBJECTS := $(patsubst $(SRC)/%.cu, $(OBJ)/%.o, $(SOURCES))


all: nn
	./nn

nn: $(OBJECTS) $(CU_OBJECTS) $(SOURCES) $(CU_SOURCES) $(HEADERS) $(CU_HEADERS) main.cpp
	$(NVCC) -std c++11 main.cpp -o nn $(INCLUDES) $(SOURCES) $(CU_SOURCES) $(NVCCFLAGS) $(LDFLAGS) 

$(OBJ)/%.o: $(SRC)/%.cpp
	$(CC) -std c++11 $(INCLUDES) -c $< -o $@ $(CXXFLAGS) $(LDFLAGS)


$(OBJ)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@




