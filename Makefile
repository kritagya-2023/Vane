CXX      ?= g++
CXXSTD   := -std=c++20
WARN     := -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
INC      := -Iinclude -Itests
OPT      ?= -O2
CXXFLAGS := $(CXXSTD) $(WARN) $(INC) $(OPT)

BUILD := build
OBJ   := $(BUILD)/obj

SRC  := src/core.cpp src/pricer.cpp src/market.cpp src/flow.cpp src/desk.cpp \
        src/policy.cpp src/simulator.cpp
OBJS := $(patsubst src/%.cpp,$(OBJ)/%.o,$(SRC))

TOOLS := vane-quote vane-sim vane-backtest
TESTS := test_unit test_property test_sim test_policy

.PHONY: all test bench sim sanitize mutate phase3 phase4 figures clean

all: $(addprefix $(BUILD)/,$(TOOLS) $(TESTS))

$(OBJ):
	@mkdir -p $(OBJ)

$(OBJ)/%.o: src/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/vane-quote: $(OBJS) tools/vane_quote.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/vane-sim: $(OBJS) tools/vane_sim.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/vane-backtest: $(OBJS) tools/vane_backtest.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/test_%: $(OBJS) tests/test_%.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(addprefix $(BUILD)/,$(TESTS))
	@for t in $(TESTS); do ./$(BUILD)/$$t || exit 1; done

bench: $(BUILD)/vane-quote
	@./$(BUILD)/vane-quote --bench

sim: $(BUILD)/vane-sim
	@./$(BUILD)/vane-sim --weeks 4 --by-tier

mutate: all
	@python3 tests/mutate.py

# Phase 3: needs numpy, pandas, scipy and scikit-learn.
phase3: $(BUILD)/vane-sim
	@mkdir -p data
	@test -f data/train_events.csv || \
		./$(BUILD)/vane-sim --weeks 26 --jitter 0.22 --out data/train >/dev/null
	@python3 analysis/test_phase3.py
	@python3 analysis/run_phase3.py --data data/train

# Address + UB sanitizers across every suite.
sanitize:
	@mkdir -p $(BUILD)
	@for t in $(TESTS); do \
		$(CXX) $(CXXSTD) $(WARN) $(INC) -O1 -g -fsanitize=address,undefined \
			-fno-omit-frame-pointer $(SRC) tests/$$t.cpp -o $(BUILD)/$${t}_asan || exit 1; \
		./$(BUILD)/$${t}_asan || exit 1; \
	done

# Phase 4: export the fitted model, backtest it, then run the closed loop.
phase4: $(BUILD)/vane-backtest
	@mkdir -p data models
	@test -f data/train_events.csv || \
		./$(BUILD)/vane-sim --weeks 26 --jitter 0.22 --out data/train >/dev/null
	@python3 analysis/export_model.py --data data/train --out models/logistic.txt
	@./$(BUILD)/test_policy
	@./$(BUILD)/vane-backtest --model models/logistic.txt --seeds 5 --ablate
	@python3 analysis/run_phase4.py --generations 5

# Static figures for the README. Read-only over whatever the CLI has already
# produced; missing inputs are skipped with a hint rather than fabricated.
figures:
	@python3 analysis/make_figures.py --out docs
	@python3 analysis/test_figures.py

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d)
