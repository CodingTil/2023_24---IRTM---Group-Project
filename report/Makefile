DOCNAME = main
LUALATEX = lualatex
BIBTEX = bibtex

all: $(DOCNAME).pdf

$(DOCNAME).pdf: $(DOCNAME).tex $(DOCNAME).bbl
	$(LUALATEX) $(DOCNAME).tex
	$(LUALATEX) $(DOCNAME).tex

$(DOCNAME).bbl: $(DOCNAME).bib
	$(LUALATEX) $(DOCNAME).tex
	$(BIBTEX) $(DOCNAME)
	$(LUALATEX) $(DOCNAME).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc $(DOCNAME).pdf

.PHONY: all clean
