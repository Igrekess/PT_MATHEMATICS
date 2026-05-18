all: en fr

en:
	latexmk -pdf -interaction=nonstopmode main_en.tex

fr:
	latexmk -pdf -interaction=nonstopmode main_fr.tex

clean:
	latexmk -c

watch-en:
	latexmk -pvc -pdf main_en.tex

watch-fr:
	latexmk -pvc -pdf main_fr.tex
