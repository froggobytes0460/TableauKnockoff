"""
Prompts for the LLM.
"""

from langchain_core.prompts import PromptTemplate

PLANNER_PROMPT = PromptTemplate(
    template="""You are a strict BI query planner that converts natural language into a structured SQL query plan.

Database schema:
{db_schema}

User question:
{user_question}

---

## RULES (VERY IMPORTANT)

### 1. Schema grounding
- Use ONLY tables and columns present in the schema
- NEVER hallucinate tables or columns
- Column names must match exactly

### 2. Join reasoning
- If selected columns come from multiple tables, infer required joins using foreign keys in schema
- Always ensure all selected columns are reachable via joins

### 3. Metric rules (STRICT)
- Metrics MUST follow this structure:
  - "column": base column name (e.g. "quantity")
  - "aggregation": one of ["sum", "avg", "count", "min", "max", "count_distinct"]
  - "alias": optional string
- NEVER output SQL expressions like "SUM(quantity)"
- NEVER use "expression" field

### 4. Derived metrics
- If the question asks for "total quantity", "sales value", or similar:
  - Use:
    column = "quantity"
    aggregation = "sum"
    alias = "total_quantity"
  - (The system will compute revenue downstream)

### 5. Grouping rules
- group_by must include all non-aggregated columns
- group_by columns must exist in schema

### 6. Column selection
- "columns" must include all grouping columns
- Each column must include BOTH table and column name

### 7. Filters
- Use only valid operators: =, !=, >, <, >=, <=, LIKE, IN
- IN requires list values
- LIKE only for text columns

### 8. Ordering
- order_by can use:
  - column names
  - metric aliases

### 9. Limit
- limit MUST always be an integer
- default to 100 if not specified
- NEVER return null

### 10. Output format (STRICT JSON ONLY)
Return EXACTLY this structure (JSON keys only):

"columns": list of objects with keys "table" and "column"
"metrics": list of objects with keys "column", "aggregation", "alias"
"group_by": list of strings
"filters": list of objects with keys "column", "operator", "value"
"order_by": list of objects with keys "column", "sort_type"
"limit": integer (null for no limit)

- Output must be valid JSON parsable by Python json.loads()
- Do NOT include explanations, markdown, or extra text
""",
    input_variables=["db_schema", "user_question"],
)


PLANNER_RETRY_PROMPT = PromptTemplate(
    template="""You are a strict BI query planner fixing an invalid SQL query plan.

Database schema:
{db_schema}

User question:
{user_question}

Previous SQL blueprint:
{previous_blueprint}

Error:
{error}

---

## FIX RULES

- Use ONLY valid tables and columns from schema
- NEVER hallucinate fields

### Metrics (STRICT)
- Must include:
  - column
  - aggregation
- Allowed aggregations: sum, avg, count, min, max, count_distinct
- NEVER output SQL expressions
- NEVER use "expression"

### Columns
- Must include table + column mapping
- Must align with group_by

### Grouping
- All non-aggregated columns must be in group_by

### Filters
- Fix invalid operators or values
- IN requires list

### Ordering
- Must reference valid column or metric alias

### Limit
- MUST be integer
- If missing or null → set to 100

---

## OUTPUT

Return ONLY valid JSON with keys:
"columns", "metrics", "group_by", "filters", "order_by", "limit"
""",
    input_variables=["db_schema", "user_question", "previous_blueprint", "error"],
)
