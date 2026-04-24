.PHONY: install pull-models ingest serve clean

PY := python

install:
	$(PY) -m pip install -r requirements.txt

# Pull the Ollama models used by config.yaml. Adjust here if you change models.
pull-models:
	ollama pull llama3.1:8b
	ollama pull nomic-embed-text

ingest:
	$(PY) -m src.ingest

serve:
	$(PY) -m src.server

clean:
	rm -rf storage/chroma
