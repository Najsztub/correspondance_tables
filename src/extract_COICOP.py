import camelot
import pandas as pd
from typing import List, Union, Tuple

def extract_multipage_tables(
    pdf_path: str,
    page_ranges: List[Union[int, Tuple[int, int]]],
    flavor: str = 'lattice'
) -> List[pd.DataFrame]:
    """
    Extract tables from specific pages or page ranges in a PDF file.

    Args:
        pdf_path (str): Path to the PDF file
        page_ranges (List[Union[int, Tuple[int, int]]]): List of page numbers or ranges
            Example: [1, (3,5), 7] will extract from page 1, pages 3-5, and page 7
        flavor (str): Table extraction method - 'lattice' for tables with borders,
            'stream' for tables without clear borders. Defaults to 'lattice'.

    Returns:
        List[pd.DataFrame]: List of extracted tables as pandas DataFrames
    """
    # Convert page ranges to string format required by camelot
    formatted_pages = []
    for page_range in page_ranges:
        if isinstance(page_range, tuple):
            formatted_pages.append(f"{page_range[0]}-{page_range[1]}")
        else:
            formatted_pages.append(str(page_range))

    pages = ",".join(formatted_pages)

    # Extract tables
    try:
        # Read tables from PDF
        tables = camelot.read_pdf(
            pdf_path,
            pages=pages,
            flavor=flavor,
            strip_text='\n'
        )

        print(f"Found {len(tables)} tables in the specified pages")

        # Convert to list of pandas DataFrames
        extracted_tables = []
        for idx, table in enumerate(tables):
            df = table.df
            # Add metadata about the extraction
            df.attrs['page_number'] = table.page
            df.attrs['table_number'] = idx + 1
            df.attrs['accuracy'] = table.accuracy
            extracted_tables.append(df)

        return extracted_tables

    except Exception as e:
        print(f"Error extracting tables: {str(e)}")
        return []

def merge_split_tables(tables: List[pd.DataFrame], headers: List[str]) -> pd.DataFrame:
    """
    Merge tables that span multiple pages into a single DataFrame.

    Args:
        tables (List[pd.DataFrame]): List of extracted tables
        headers (List[str]): Expected column headers for the table

    Returns:
        pd.DataFrame: Merged table
    """

    merged_table = pd.DataFrame(columns=headers)

    for table in tables:
        # Filtruj pozostawiając tylko 
        # Check if this is a header row or continuation
        if table.iloc[0].tolist() == headers:
            # Remove header row if it's a continuation
            table = table.iloc[1:]

        # Reset the index and drop it
        table.columns = headers
        table = table.reset_index(drop=True)

        # Append to the merged table
        merged_table = pd.concat([merged_table, table], ignore_index=True)

    return merged_table

# Example usage
if __name__ == "__main__":
    # Example with multiple page ranges
    pdf_path = "data/raw/zeszyt_metodologiczny._badanie_budzetow_gospodarstw_domowych.pdf"
    page_ranges = [(64, 86)]
    expected_headers = ["kod", "opis"]

    # Extract tables
    extracted_tables = extract_multipage_tables(pdf_path, page_ranges, flavor = "stream")

    if extracted_tables:
        # Merge tables if they span multiple pages
        merged_table = merge_split_tables(extracted_tables, expected_headers)

        # Filtruj tylko te obserwacje, które są kodami COICOP
        merged_table = merged_table[merged_table['kod'].str.match('^\d')]
        merged_table = merged_table[merged_table['opis'] != '']

        # Save to CSV
        merged_table.to_csv("data/processed/kody_COICOP.csv", index=False)
        print("Table extracted and saved successfully!")

        # Print table info
        print("\nTable Information:")
        print(f"Total rows: {len(merged_table)}")
        print("\nFirst few rows:")
        print(merged_table.head())
