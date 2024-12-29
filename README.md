# Analiza Podobieństwa Klasyfikacji NACE-COICOP

## Opis projektu
Projekt służy do analizy semantycznego podobieństwa między klasyfikacjami NACE (PKWiU) i COICOP, wykorzystując zaawansowane modele języka naturalnego dostosowane do języka polskiego. Głównym celem jest utworzenie macierzy podobieństwa między kategoriami obu klasyfikacji oraz wizualizacja tych powiązań.

W kolejnym kroku zamierzam zbudować automatyczną macierz przejścia między klasyfikacjami w oparciu o największą zbieżność. 

## Funkcjonalności
- Ekstrakcja kodów i opisów COICOP z metodologii BBGD
- Generowanie embedingów dla opisów kategorii z wykorzystaniem modeli językowych
- Obliczanie macierzy podobieństwa między kategoriami
- Wizualizacja wyników w formie map cieplnych i przestrzeni embedingów
- Szczegółowa analiza dopasowań między klasyfikacjami

## Struktura projektu
```
.
├── data/
│   ├── raw/
│   │   ├── pkwiu2015.xls
│   │   └── zeszyt_metodologiczny._badanie_budzetow_gospodarstw_domowych.pdf
│   └── processed/
│       ├── kody_COICOP.csv
│       └── similarity_matrix_embed.npy
└── src/
    ├── extract_COICOP.py
    └── stworz_macierz_podobienstwa.py
```

## Wymagania
- Python 3.12
- Biblioteki Python:
  - sentence-transformers
  - pandas
  - numpy
  - camelot-py
  - plotly
  - scikit-learn
  - seaborn
  - matplotlib
  - alive-progress

## Dostępne modele językowe
Projekt obsługuje następujące modele do analizy semantycznej:
- `roberta_large`: sdadas/mmlw-roberta-large
- `herbert`: allegro/herbert-base-cased
- `multilingual`: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- `polish_roberta`: sdadas/polish-roberta-base-v2

## Instalacja
```bash
pip install -r requirements.txt
```

## Użycie

### 1. Ekstrakcja kodów COICOP

Należy pobrać najnowszą matodologię BBGD ze strony GUS i zapisać w folderze `data/raw`.
Dokument można pobrać [stąd](https://stat.gov.pl/obszary-tematyczne/warunki-zycia/dochody-wydatki-i-warunki-zycia-ludnosci/zeszyt-metodologiczny-badanie-budzetow-gospodarstw-domowych,10,3.html). 

```bash
python src/extract_COICOP.py
```
Skrypt ekstrahuje kody i opisy COICOP z dokumentu PDF i zapisuje je w formacie CSV.

### 2. Generowanie macierzy podobieństwa
```bash
python src/stworz_macierz_podobienstwa.py
```
Skrypt:
- Wczytuje dane COICOP i PKWiU
- Generuje embedingi dla opisów kategorii
- Tworzy macierz podobieństwa
- Generuje wizualizacje
- Zapisuje wyniki do plików

Wyagany jest plik `pkwiu2015.xls` dostępny na [stronie GUS](https://stat.gov.pl/Klasyfikacje/doc/pkwiu_15/pdf/pkwiu2015.xls).

## Główne komponenty

### PolishSemanticMatcher
Klasa odpowiedzialna za:
- Inicjalizację modelu językowego
- Generowanie embedingów dla tekstów
- Obliczanie podobieństwa między embedingami

### SimilarityVisualizer
Zapewnia narzędzia do wizualizacji:
- Mapy cieplnej podobieństwa między kategoriami
- Przestrzeni embedingów z wykorzystaniem t-SNE

### CorrespondenceAnalyzer
Oferuje zaawansowane narzędzia analityczne:
- Identyfikacja najlepszych dopasowań między kategoriami
- Analiza pokrycia klasyfikacji
- Statystyki podobieństwa

## Wyniki
Projekt generuje:
- Macierz podobieństwa między kategoriami (`.npy`)
- Interaktywne wizualizacje w formie map cieplnych
- Szczegółowe analizy dopasowań między klasyfikacjami
- Wizualizację przestrzeni embedingów

## Uwagi
- Dla najlepszych wyników zalecane jest używanie modelu `roberta_large`
- Proces generowania embedingów może być czasochłonny dla dużych zbiorów danych
- Wizualizacje są generowane w formacie interaktywnym przy użyciu biblioteki plotly