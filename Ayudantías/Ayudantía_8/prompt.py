from langchain_core.prompts import PromptTemplate


poem_extraction = PromptTemplate.from_template("""
You are an expert in analyzing and extracting poetry from raw literary text.

Your task is to **identify and extract ALL distinct poems** from the provided text that corresponds to the same author. For each poem, you must extract its `title`, `author`, and `content`.

---

## Poem Identification Guidelines

1. **Poem Structure**: Look for blocks of text that follow poetic structure—line breaks, stanzas, rhythm, or figurative language. These blocks are often separated by whitespace or page breaks.
2. **Title Extraction**:
   - If the poem has a clear title, use it as is.
   - If no title is present, use the **first line** of the poem as its title.
   - If it's a short fragment with no clear start, use a descriptive fallback like "Poem beginning with..." or "Fragmento de poema".
3. **Author Attribution**:
   - Is the same name for all te text.
   - Otherwise, use "Unknown".
4. **Content Preservation**:
   - Preserve **all** original line breaks, indentation, punctuation, and stanzas.
   - Do not modify or "prettify" the poem content.
5. **Ignore non-poetic content**:
   - Ignore page numbers, footers, headers, publication details, table of contents, or anything clearly not part of a poem.

---

## Output Format Instructions (Strict JSON only)

- Your output must be a **valid JSON object only**. Do **NOT** include explanations, markdown syntax, or commentary.
- **Do not include code block markers** such as ```json or ```.
- Output must be **parsable JSON**, without any additional wrapping or text.
- Return a single object with the top-level key "poemas_encontrados" that maps to an array of poem objects.
- Each poem object must contain only the keys: "title", "author", and "content".

---

Text to analyze:
{page_content}

---

Return the result in this exact format:

{{
  "poemas_encontrados": [
    {{
      "title": "Poem Title",
      "author": "Author Name",
      "content": "Poem text with\\nline breaks exactly\\nas they appear."
    }},
    {{
      "title": "Second Poem Title",
      "author": "Unknown",
      "content": "Another complete\\npoem in correct\\nformat."
    }}
  ]
}}
"""
)