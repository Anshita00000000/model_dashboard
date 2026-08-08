.PHONY: run tunnel test

run:
	streamlit run app/ui/main.py

tunnel:
	ngrok http 8501

test:
	pytest
