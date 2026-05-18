TARGET = ramanujan_mihailescu

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex
	pdflatex -interaction=nonstopmode $(TARGET).tex
	pdflatex -interaction=nonstopmode $(TARGET).tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg

distclean: clean
	rm -f $(TARGET).pdf

.PHONY: all clean distclean
