# camdl-viewer demo: generate measles simulations + observed data, then view.
#
#   make        # build sims + observed data (the default)
#   make view   # launch the Streamlit viewer on the generated data
#   make clean  # remove generated artifacts

CAMDL      ?= camdl
MODEL      := he_measles.camdl
OUT        := output
RUNS       := $(OUT)/sims
OBS        := data/observed
TRUTH_SEED ?= 99

.DEFAULT_GOAL := all
.PHONY: all sims obs view clean help

all: sims obs ## Generate simulations and observed data
	@echo
	@echo "Data ready. Launch the viewer with:  make view"

sims: $(RUNS)/manifest.json ## Run the batch: 30 seeds x 3 scenarios -> CAS runs tree

$(RUNS)/manifest.json: $(MODEL) batch.toml params.toml
	$(CAMDL) batch run batch.toml --output-dir $(OUT)

obs: $(OBS)/I.tsv ## Generate held-out "truth" observations via camdl's observation model

$(OBS)/I.tsv: $(MODEL) params.toml
	$(CAMDL) simulate $(MODEL) --params params.toml --backend chain_binomial --dt 1 \
		--seed $(TRUTH_SEED) --obs-dir $(OBS) -o /dev/null

view: all ## Launch the Streamlit viewer on the generated CAS + observed data
	uv run streamlit run app.py -- --runs $(RUNS) --obs $(OBS)/I.tsv

clean: ## Remove generated output/ and data/
	rm -rf $(OUT) $(OBS)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-8s %s\n", $$1, $$2}'
