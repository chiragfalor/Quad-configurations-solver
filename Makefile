CC=g++
CFLAGS=-W -Wall -O3
INCLUDES=-I.
LDFLAGS=
LDLIBS=-lm
EXEC=potentials
SRC=
OBJ=$(SRC:.cpp=.o)

runtest = echo "Running $(1)";./$(1);

.PHONY: clean dist-clean

all: $(EXEC)

%.o : %.cpp
	@echo "CC $^"
	@$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $^

clean:
	@echo "Cleaning object files"
	@rm -rf $(OBJ)

potentials: potentials.cpp $(OBJ)
	@echo "MAKE $@"
	@$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LDLIBS)

dist-clean: clean
	@rm -f $(EXEC)

test: $(EXEC)
	@$(foreach prog,$(EXEC),$(call runtest,$(prog)))
