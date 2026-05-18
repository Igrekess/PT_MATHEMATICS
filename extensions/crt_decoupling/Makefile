all:
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -c

watch:
	latexmk -pvc -pdf main.tex

tests:
	cd scripts && python test_crt_decoupling.py
