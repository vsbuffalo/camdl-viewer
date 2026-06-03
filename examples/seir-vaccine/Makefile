# Vaccine SEIR fixture for camdl-viewer.
#
#   make           generate results/ (3 coverage scenarios × seeds) + held-out truth obs
#   make view      launch the viewer on this fixture
#   make check     type-check the model
#   make clean     remove generated results/ and data/
#
# Generation is deterministic and offline (no network). camdl must be on PATH,
# or pass an explicit binary:  make CAMDL=~/.local/bin/camdl
#
# One model, one backend, one base parameter set — the tree branches only at
# the `coverage` scenario (×3) and `seed`, so the viewer's TOC is genuinely
# tree-shaped.

CAMDL      ?= camdl
MODEL      := seir_vaccine.camdl
RESULTS    := results
OBS        := data/observed
SCENARIOS  := no_vaccine vaccine_50 vaccine_80
SEEDS      ?= 1:12
TRUTH_SEED ?= 99
SIM         = $(CAMDL) simulate $(MODEL) --backend chain_binomial --dt 1 \
                --output-dir $(RESULTS) --no-progress

.DEFAULT_GOAL := all
.PHONY: all view check clean help

all: $(RESULTS)/index.json ## generate the fixture (sims + truth obs + index)

$(RESULTS)/index.json: $(MODEL)
	@for s in $(SCENARIOS); do \
		echo "  simulate $$s  (seeds $(SEEDS))"; \
		$(SIM) --scenario $$s --seeds $(SEEDS); \
	done
	@echo "  truth obs  (held-out, seed $(TRUTH_SEED))"
	$(SIM) --scenario no_vaccine --seed $(TRUTH_SEED) --obs-only-dir $(OBS)
	$(CAMDL) reindex $(RESULTS)

check: $(MODEL) ## type-check the model
	$(CAMDL) check $(MODEL)

view: all ## launch the viewer on this fixture
	cd ../.. && uv run streamlit run app.py -- --results examples/seir-vaccine/$(RESULTS)

clean: ## remove generated data
	rm -rf $(RESULTS) $(OBS)

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-8s %s\n", $$1, $$2}'
